import torch
import torch.nn.functional as F
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, Denoise, LatentEncoder, LatentDecoder
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix
import wandb
import time


class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
		
		# Track inference time statistics
		self.total_inference_time = 0.0
		self.inference_count = 0

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = 0
		ndcgMax = 0
		precisionMax = 0
		bestEpoch = 0

		log('Model Initialized')

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			
			train_log = {f"train/{k}": v for k, v in reses.items()}
			train_log["epoch"] = ep
			wandb.log(train_log)
			
			if tstFlag:
				reses = self.testEpoch()
				# Accumulate inference time statistics
				if 'Inference Time (s)' in reses:
					self.total_inference_time += reses['Inference Time (s)']
					self.inference_count += 1
				
				if (reses['Recall'] > recallMax):
					recallMax = reses['Recall']
					ndcgMax = reses['NDCG']
					precisionMax = reses['Precision']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))

				test_log = {f"test/{k}": v for k, v in reses.items()}
				test_log["epoch"] = ep
				wandb.log(test_log)

			print()
		print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax, ' , Precision', precisionMax)

		# Calculate and log inference time statistics
		if self.inference_count > 0:
			avg_inference_time = self.total_inference_time / self.inference_count
			print(f'Total Inference Time: {self.total_inference_time:.4f} seconds')
			print(f'Average Inference Time per Step: {avg_inference_time:.4f} seconds')
			print(f'Total Inference Steps: {self.inference_count}')
			
			# Log to wandb summary
			wandb.summary["inference/total_time_seconds"] = self.total_inference_time
			wandb.summary["inference/avg_time_per_step_seconds"] = avg_inference_time
			wandb.summary["inference/total_steps"] = self.inference_count
		else:
			print('No inference steps recorded')

		wandb.summary["best/Recall"] = recallMax
		wandb.summary["best/NDCG"] = ndcgMax
		wandb.summary["best/Precision"] = precisionMax
		wandb.summary["best/Epoch"] = bestEpoch


	def prepareModel(self):
		if args.data == 'tiktok':
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach(), self.handler.audio_feats.detach()).cuda()
		else:
			self.model = Model(self.handler.image_feats.detach(), self.handler.text_feats.detach()).cuda()
		
		# Use different learning rates for Transformer if enabled
		if args.use_transformer_fusion and self.model.transformer_fusion is not None:
			# Separate parameters: Transformer uses smaller LR, others use normal LR
			transformer_params = list(self.model.transformer_fusion.parameters())
			other_params = [p for name, p in self.model.named_parameters() 
			               if 'transformer_fusion' not in name]
			transformer_lr = args.lr * args.transformer_lr_scale
			print(f"Using separate learning rates: Main={args.lr}, Transformer={transformer_lr}")
			self.opt = torch.optim.Adam([
				{'params': other_params, 'lr': args.lr},
				{'params': transformer_params, 'lr': transformer_lr}
			], weight_decay=0)
		else:
			self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
		
		# Initialize latent space encoder and decoder if using latent space
		if args.use_latent_space:
			self.encoder_image = LatentEncoder(args.item, args.latent_dim).cuda()
			self.decoder_image = LatentDecoder(args.latent_dim, args.item).cuda()
			self.encoder_opt_image = torch.optim.Adam(list(self.encoder_image.parameters()) + list(self.decoder_image.parameters()), lr=args.lr, weight_decay=0)
			
			self.encoder_text = LatentEncoder(args.item, args.latent_dim).cuda()
			self.decoder_text = LatentDecoder(args.latent_dim, args.item).cuda()
			self.encoder_opt_text = torch.optim.Adam(list(self.encoder_text.parameters()) + list(self.decoder_text.parameters()), lr=args.lr, weight_decay=0)
			
			if args.data == 'tiktok':
				self.encoder_audio = LatentEncoder(args.item, args.latent_dim).cuda()
				self.decoder_audio = LatentDecoder(args.latent_dim, args.item).cuda()
				self.encoder_opt_audio = torch.optim.Adam(list(self.encoder_audio.parameters()) + list(self.decoder_audio.parameters()), lr=args.lr, weight_decay=0)
			
			# Denoise model works in latent space
			diffusion_dim = args.latent_dim
		else:
			self.encoder_image = None
			self.decoder_image = None
			self.encoder_opt_image = None
			self.encoder_text = None
			self.decoder_text = None
			self.encoder_opt_text = None
			if args.data == 'tiktok':
				self.encoder_audio = None
				self.decoder_audio = None
				self.encoder_opt_audio = None
			# Denoise model works in original space
			diffusion_dim = args.item
		
		out_dims = eval(args.dims) + [diffusion_dim]
		in_dims = out_dims[::-1]
		self.denoise_model_image = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_image = torch.optim.Adam(self.denoise_model_image.parameters(), lr=args.lr, weight_decay=0)

		out_dims = eval(args.dims) + [diffusion_dim]
		in_dims = out_dims[::-1]
		self.denoise_model_text = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt_text = torch.optim.Adam(self.denoise_model_text.parameters(), lr=args.lr, weight_decay=0)

		if args.data == 'tiktok':
			out_dims = eval(args.dims) + [diffusion_dim]
			in_dims = out_dims[::-1]
			self.denoise_model_audio = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
			self.denoise_opt_audio = torch.optim.Adam(self.denoise_model_audio.parameters(), lr=args.lr, weight_decay=0)

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss = 0
		epDiLoss_image, epDiLoss_text = 0, 0
		if args.data == 'tiktok':
			epDiLoss_audio = 0
		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader
		# epGradNorm = 0.0

		for i, batch in enumerate(diffusionLoader):
			batch_item, batch_index = batch
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			iEmbeds = self.model.getItemEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			image_feats = self.model.getImageFeats().detach()
			text_feats = self.model.getTextFeats().detach()
			if args.data == 'tiktok':
				audio_feats = self.model.getAudioFeats().detach()

			self.denoise_opt_image.zero_grad()
			self.denoise_opt_text.zero_grad()
			if args.data == 'tiktok':
				self.denoise_opt_audio.zero_grad()
			
			if args.use_latent_space:
				if self.encoder_opt_image is not None:
					self.encoder_opt_image.zero_grad()
				if self.encoder_opt_text is not None:
					self.encoder_opt_text.zero_grad()
				if args.data == 'tiktok' and self.encoder_opt_audio is not None:
					self.encoder_opt_audio.zero_grad()

			diff_loss_image, gc_loss_image = self.diffusion_model.training_losses(
				self.denoise_model_image, batch_item, iEmbeds, batch_index, image_feats,
				self.encoder_image, self.decoder_image
			)
			diff_loss_text, gc_loss_text = self.diffusion_model.training_losses(
				self.denoise_model_text, batch_item, iEmbeds, batch_index, text_feats,
				self.encoder_text, self.decoder_text
			)
			if args.data == 'tiktok':
				diff_loss_audio, gc_loss_audio = self.diffusion_model.training_losses(
					self.denoise_model_audio, batch_item, iEmbeds, batch_index, audio_feats,
					self.encoder_audio, self.decoder_audio
				)

			loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			loss_text = diff_loss_text.mean() + gc_loss_text.mean() * args.e_loss
			if args.data == 'tiktok':
				loss_audio = diff_loss_audio.mean() + gc_loss_audio.mean() * args.e_loss

			epDiLoss_image += loss_image.item()
			epDiLoss_text += loss_text.item()
			if args.data == 'tiktok':
				epDiLoss_audio += loss_audio.item()

			if args.data == 'tiktok':
				loss = loss_image + loss_text + loss_audio
			else:
				loss = loss_image + loss_text

			loss.backward()
			# grad_norm = calcGradNorm(self.model)
			# epGradNorm += grad_norm.item()
	
			self.denoise_opt_image.step()
			self.denoise_opt_text.step()
			if args.data == 'tiktok':
				self.denoise_opt_audio.step()
			
			if args.use_latent_space:
				if self.encoder_opt_image is not None:
					self.encoder_opt_image.step()
				if self.encoder_opt_text is not None:
					self.encoder_opt_text.step()
				if args.data == 'tiktok' and self.encoder_opt_audio is not None:
					self.encoder_opt_audio.step()

			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

		log('')
		log('Start to re-build UI matrix')

		with torch.no_grad():

			u_list_image = []
			i_list_image = []
			edge_list_image = []

			u_list_text = []
			i_list_text = []
			edge_list_text = []

			if args.data == 'tiktok':
				u_list_audio = []
				i_list_audio = []
				edge_list_audio = []

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

				# image
				denoised_batch = self.diffusion_model.p_sample(
					self.denoise_model_image, batch_item, args.sampling_steps, args.sampling_noise,
					self.encoder_image, self.decoder_image
				)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_image.append(int(batch_index[i].cpu().numpy()))
						i_list_image.append(int(indices_[i][j].cpu().numpy()))
						edge_list_image.append(1.0)

				# text
				denoised_batch = self.diffusion_model.p_sample(
					self.denoise_model_text, batch_item, args.sampling_steps, args.sampling_noise,
					self.encoder_text, self.decoder_text
				)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]): 
						u_list_text.append(int(batch_index[i].cpu().numpy()))
						i_list_text.append(int(indices_[i][j].cpu().numpy()))
						edge_list_text.append(1.0)

				if args.data == 'tiktok':
					# audio
					denoised_batch = self.diffusion_model.p_sample(
						self.denoise_model_audio, batch_item, args.sampling_steps, args.sampling_noise,
						self.encoder_audio, self.decoder_audio
					)
					top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

					for i in range(batch_index.shape[0]):
						for j in range(indices_[i].shape[0]): 
							u_list_audio.append(int(batch_index[i].cpu().numpy()))
							i_list_audio.append(int(indices_[i][j].cpu().numpy()))
							edge_list_audio.append(1.0)

			# image
			u_list_image = np.array(u_list_image)
			i_list_image = np.array(i_list_image)
			edge_list_image = np.array(edge_list_image)
			self.image_UI_matrix = self.buildUIMatrix(u_list_image, i_list_image, edge_list_image)
			self.image_UI_matrix = self.model.edgeDropper(self.image_UI_matrix)

			# text
			u_list_text = np.array(u_list_text)
			i_list_text = np.array(i_list_text)
			edge_list_text = np.array(edge_list_text)
			self.text_UI_matrix = self.buildUIMatrix(u_list_text, i_list_text, edge_list_text)
			self.text_UI_matrix = self.model.edgeDropper(self.text_UI_matrix)

			if args.data == 'tiktok':
				# audio
				u_list_audio = np.array(u_list_audio)
				i_list_audio = np.array(i_list_audio)
				edge_list_audio = np.array(edge_list_audio)
				self.audio_UI_matrix = self.buildUIMatrix(u_list_audio, i_list_audio, edge_list_audio)
				self.audio_UI_matrix = self.model.edgeDropper(self.audio_UI_matrix)

		log('UI matrix built!')

		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()

			if args.data == 'tiktok':
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]
			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			# Use numerical stable version: -log(sigmoid(x)) = log(1 + exp(-x))
			# This is equivalent to -log(sigmoid(x)) but more stable
			bprLoss = F.softplus(-scoreDiff).sum() / args.batch
			regLoss = self.model.reg_loss() * args.reg
			
			# Alignment loss for Transformer (if enabled)
			alignLoss = torch.tensor(0.0, device=bprLoss.device)
			if args.use_transformer_fusion and args.transformer_align_weight > 0:
				alignLoss = self.model.transformer_align_loss() * args.transformer_align_weight
			
			loss = bprLoss + regLoss + alignLoss
			
			# Debug: check for NaN, inf, or zero loss
			if i == 0:  # Only print once at the beginning of each epoch
				print(f"\n=== Debug Info (First Step of Epoch) ===")
				print(f"scoreDiff stats: min={scoreDiff.min().item():.4f}, max={scoreDiff.max().item():.4f}, mean={scoreDiff.mean().item():.4f}")
				print(f"bprLoss: {bprLoss.item():.6f}, regLoss: {regLoss.item():.6f}, total loss: {loss.item():.6f}")
				print(f"usrEmbeds stats: min={usrEmbeds.min().item():.4f}, max={usrEmbeds.max().item():.4f}, mean={usrEmbeds.mean().item():.4f}")
				print(f"itmEmbeds stats: min={itmEmbeds.min().item():.4f}, max={itmEmbeds.max().item():.4f}, mean={itmEmbeds.mean().item():.4f}")
				if args.use_transformer_fusion:
					print(f"Using Transformer fusion: layers={args.transformer_layers}, heads={args.transformer_heads}")
				print("=" * 40 + "\n")
			
			if torch.isnan(loss) or torch.isinf(loss):
				print(f"ERROR: NaN/Inf detected in loss! bprLoss: {bprLoss.item()}, regLoss: {regLoss.item()}")
				print(f"scoreDiff stats: min={scoreDiff.min().item():.4f}, max={scoreDiff.max().item():.4f}, mean={scoreDiff.mean().item():.4f}")
				# Skip this batch
				continue
			
			epRecLoss += bprLoss.item()
			epLoss += loss.item()

			if args.data == 'tiktok':
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
			else:
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)
			if args.data == 'tiktok':
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			else:
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			if args.data == 'tiktok':
				clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss_ = clLoss1 + clLoss2 + clLoss3
			else:
				clLoss_ = clLoss1 + clLoss2

			if args.cl_method == 1:
				clLoss = clLoss_

			loss += clLoss

			epClLoss += clLoss.item()

			loss.backward()
			
			# Gradient clipping to prevent explosion (especially important for Transformer)
			# Use adaptive clipping: clip all params, but be stricter with Transformer
			if args.use_transformer_fusion:
				# First clip all parameters
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
				# Then apply stricter clipping to Transformer specifically
				if self.model.transformer_fusion is not None:
					torch.nn.utils.clip_grad_norm_(
						self.model.transformer_fusion.parameters(), 
						max_norm=args.transformer_grad_clip
					)
			else:
				# Standard gradient clipping
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
			
			# Check gradients before update (only first step)
			if i == 0:
				grad_norm = 0.0
				param_count = 0
				for name, param in self.model.named_parameters():
					if param.grad is not None:
						grad_norm += param.grad.data.norm(2).item() ** 2
						param_count += 1
				grad_norm = grad_norm ** 0.5
				print(f"Gradient norm: {grad_norm:.6f}, Parameters with grad: {param_count}")
				if args.use_transformer_fusion and self.model.transformer_fusion is not None:
					transformer_grad_norm = 0.0
					transformer_param_count = 0
					for name, param in self.model.transformer_fusion.named_parameters():
						if param.grad is not None:
							transformer_grad_norm += param.grad.data.norm(2).item() ** 2
							transformer_param_count += 1
					transformer_grad_norm = transformer_grad_norm ** 0.5
					print(f"Transformer gradient norm: {transformer_grad_norm:.6f}, Transformer params with grad: {transformer_param_count}")
			
			self.opt.step()

			log('Step %d/%d: bpr : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i, 
				steps,
				bprLoss.item(),
        regLoss.item(),
				clLoss.item()
				), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['Di image loss'] = epDiLoss_image / (diffusionLoader.dataset.__len__() // args.batch)
		ret['Di text loss'] = epDiLoss_text / (diffusionLoader.dataset.__len__() // args.batch)
		if args.data == 'tiktok':
			ret['Di audio loss'] = epDiLoss_audio / (diffusionLoader.dataset.__len__() // args.batch)
			
		# ret['GradNorm'] = epGradNorm / steps
		return ret

	def testEpoch(self):
		# Record inference start time
		inference_start_time = time.perf_counter()
		
		tstLoader = self.handler.tstLoader
		epRecall, epNdcg, epPrecision = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		if args.data == 'tiktok':
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix, self.audio_UI_matrix)
		else:
			usrEmbeds, itmEmbeds = self.model.forward_MM(self.handler.torchBiAdj, self.image_UI_matrix, self.text_UI_matrix)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, args.topk)
			recall, ndcg, precision = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			epPrecision += precision
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f , precision = %.2f   ' % (i, steps, recall, ndcg, precision), save=False, oneline=True)
		
		# Record inference end time and calculate duration
		inference_end_time = time.perf_counter()
		inference_time = inference_end_time - inference_start_time
		
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		ret['Precision'] = epPrecision / num
		ret['Inference Time (s)'] = inference_time
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = allPrecision = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = precision = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
					precision += 1
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			precision = precision / args.topk
			allRecall += recall
			allNdcg += ndcg
			allPrecision += precision
		return allRecall, allNdcg, allPrecision

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	# Build run name
	# If user provides a custom run_name, use it; otherwise auto-generate
	if args.run_name:
		run_name = args.run_name
	else:
		# Auto-generate run name
		run_name = f"{args.data}_lat{args.latdim}_gnn{args.gnn_layer}"
		if args.use_latent_space:
			run_name += f"_latent{args.latent_dim}"
	
	# Initialize wandb
	# entity=None: Use the current logged-in user's personal workspace (recommended)
	# If logging in with API key, it will automatically use the user workspace corresponding to that key
	# You can also specify entity name via environment variable WANDB_ENTITY
	wandb.init(
        project="enhancing_diffmm",     # You can change this to your own project name
        entity=None,  # None = Use the current logged-in user's personal workspace
        config=vars(args),
        name=run_name
    )

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()

	wandb.finish()
