import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.edgeDropper = SpAdjDropEdge(args.keepRate)

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None

		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
		self.softmax = nn.Softmax(dim=0)

		self.dropout = nn.Dropout(p=0.1)

		self.leakyrelu = nn.LeakyReLU(0.2)
		
		# Transformer fusion module
		if args.use_transformer_fusion:
			self.transformer_fusion = TransformerFusion(
				d_model=args.latdim,
				n_layers=args.transformer_layers,
				n_heads=args.transformer_heads,
				dropout=args.transformer_dropout
			)
		else:
			self.transformer_fusion = None
				
	def getItemEmbeds(self):
		return self.iEmbeds
	
	def getUserEmbeds(self):
		return self.uEmbeds
	
	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			return self.image_trans(self.image_embedding)
	
	def getTextFeats(self):
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			return self.text_trans(self.text_embedding)

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return audio_feats

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		weight = self.softmax(self.modal_weight)

		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(adj, embedsImage)

		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
		embedsImage_ = torch.spmm(adj, embedsImage_)
		embedsImage += embedsImage_
		
		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(adj, embedsText)

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)
		embedsText += embedsText_

		if audio_adj != None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)
			embedsAudio += embedsAudio_

		embedsImage += args.ris_adj_lambda * embedsImageAdj
		embedsText += args.ris_adj_lambda * embedsTextAdj
		if audio_adj != None:
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj
		
		# Multi-modal fusion: Transformer or weighted sum
		if args.use_transformer_fusion and self.transformer_fusion is not None:
			# Split into user and item parts
			embedsImage_u = embedsImage[:args.user]  # (num_users, d)
			embedsImage_i = embedsImage[args.user:]  # (num_items, d)
			embedsText_u = embedsText[:args.user]
			embedsText_i = embedsText[args.user:]
			
			if audio_adj != None:
				embedsAudio_u = embedsAudio[:args.user]
				embedsAudio_i = embedsAudio[args.user:]
				# Transformer fusion for users
				embedsModal_u = self.transformer_fusion([embedsImage_u, embedsText_u, embedsAudio_u])
				# Transformer fusion for items
				embedsModal_i = self.transformer_fusion([embedsImage_i, embedsText_i, embedsAudio_i])
			else:
				# Transformer fusion for users
				embedsModal_u = self.transformer_fusion([embedsImage_u, embedsText_u])
				# Transformer fusion for items
				embedsModal_i = self.transformer_fusion([embedsImage_i, embedsText_i])
			
			# Check for NaN/Inf in transformer output
			if torch.isnan(embedsModal_u).any() or torch.isinf(embedsModal_u).any():
				print("Warning: NaN/Inf detected in transformer output (users)!")
				embedsModal_u = torch.nan_to_num(embedsModal_u, nan=0.0, posinf=1.0, neginf=-1.0)
			if torch.isnan(embedsModal_i).any() or torch.isinf(embedsModal_i).any():
				print("Warning: NaN/Inf detected in transformer output (items)!")
				embedsModal_i = torch.nan_to_num(embedsModal_i, nan=0.0, posinf=1.0, neginf=-1.0)
			
			# Concatenate back
			embedsModal_transformer = torch.cat([embedsModal_u, embedsModal_i], dim=0)
			
			# Compute original weighted sum as residual connection
			if audio_adj == None:
				embedsModal_weighted = weight[0] * embedsImage + weight[1] * embedsText
			else:
				embedsModal_weighted = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio
			
			# Align transformer output with weighted sum (cosine similarity alignment)
			# This helps Transformer learn in the same semantic space
			with torch.no_grad():
				# Normalize both for cosine similarity
				transformer_norm = F.normalize(embedsModal_transformer, p=2, dim=1)
				weighted_norm = F.normalize(embedsModal_weighted, p=2, dim=1)
				# Project transformer output to be more aligned with weighted sum
				# This is a soft alignment that preserves Transformer's learned information
				cosine_sim = (transformer_norm * weighted_norm).sum(dim=1, keepdim=True)
				# If cosine similarity is low, align more; if high, keep as is
				align_factor = (1 - cosine_sim.clamp(-1, 1)) * 0.3  # Max 30% alignment adjustment
			
			# Apply alignment: move transformer output closer to weighted sum direction
			aligned_transformer = embedsModal_transformer + align_factor * (embedsModal_weighted - embedsModal_transformer)
			
			# Residual connection: Transformer learns to refine the weighted sum
			# Scale transformer output to be a "residual" (smaller magnitude)
			transformer_scale = 0.1  # Transformer output is a small refinement
			embedsModal = embedsModal_weighted + transformer_scale * aligned_transformer
			
			# Light normalization: only if needed to prevent extreme values
			embedsModal_norm = embedsModal.norm(p=2, dim=1, keepdim=True)
			expected_norm = (args.latdim ** 0.5) * 0.5
			normalize_threshold = expected_norm * 3.0
			
			# Only normalize samples where norm exceeds threshold
			needs_normalize = embedsModal_norm > normalize_threshold
			if needs_normalize.any():
				normalized = F.layer_norm(embedsModal, (embedsModal.shape[-1],))
				normalized = normalized * expected_norm
				embedsModal = torch.where(needs_normalize, normalized, embedsModal)
			
			# Store for alignment loss computation (keep gradients for learning)
			self._transformer_output = embedsModal_transformer
			self._weighted_output = embedsModal_weighted
		else:
			# Original weighted sum fusion
			if audio_adj == None:
				embedsModal = weight[0] * embedsImage + weight[1] * embedsText
			else:
				embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio

		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)

		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)

		return embeds[:args.user], embeds[args.user:]

	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if audio_adj != None:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		if audio_adj != None:
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)

		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		if audio_adj != None:
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		if audio_adj == None:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		else:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret
	
	def transformer_align_loss(self):
		"""Alignment loss to ensure Transformer output is in similar semantic space as weighted sum"""
		if not args.use_transformer_fusion or self.transformer_fusion is None:
			return torch.tensor(0.0, device=self.uEmbeds.device)
		
		if not hasattr(self, '_transformer_output') or not hasattr(self, '_weighted_output'):
			return torch.tensor(0.0, device=self.uEmbeds.device)
		
		# Normalize both outputs
		transformer_norm = F.normalize(self._transformer_output, p=2, dim=1)
		weighted_norm = F.normalize(self._weighted_output, p=2, dim=1)
		
		# Cosine similarity loss: encourage high similarity (but not identical)
		cosine_sim = (transformer_norm * weighted_norm).sum(dim=1).mean()
		# We want similarity around 0.7-0.9 (similar but not identical)
		target_sim = 0.8
		align_loss = (cosine_sim - target_sim).square()
		
		return align_loss

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class TransformerFusionBlock(nn.Module):
	"""Single Transformer block for multi-modal fusion"""
	def __init__(self, d_model, nhead, dropout=0.1):
		super(TransformerFusionBlock, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
		self.ffn = nn.Sequential(
			nn.Linear(d_model, d_model * 4),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_model * 4, d_model),
			nn.Dropout(dropout)
		)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		# x shape: (seq_len, batch_size, d_model)
		# Self-attention with residual
		attn_out, _ = self.self_attn(x, x, x)
		x = self.norm1(x + self.dropout(attn_out))
		
		# FFN with residual
		ffn_out = self.ffn(x)
		x = self.norm2(x + ffn_out)
		
		return x

class TransformerFusion(nn.Module):
	"""Transformer encoder for multi-modal fusion"""
	def __init__(self, d_model, n_layers, n_heads, dropout=0.1):
		super(TransformerFusion, self).__init__()
		self.d_model = d_model
		self.n_layers = n_layers
		
		# CLS token (learnable) - initialize to zeros
		# This makes Transformer start as identity (output ≈ 0, so residual = weighted_sum)
		# Then it learns small refinements to the weighted sum
		self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
		
		# Positional embedding for modal tokens
		# Max 4 positions: [CLS, Image, Text, Audio] (or [CLS, Image, Text] for 2-modal)
		max_seq_len = 4  # CLS + up to 3 modalities
		self.pos_embedding = nn.Parameter(init(torch.empty(max_seq_len, 1, d_model)) * 0.1)
		
		# Transformer blocks
		self.transformer_blocks = nn.ModuleList([
			TransformerFusionBlock(d_model, n_heads, dropout) 
			for _ in range(n_layers)
		])
	
	def forward(self, modal_embeds):
		"""
		Args:
			modal_embeds: list of tensors, each of shape (batch_size, d_model)
						  e.g., [embedsImage, embedsText] or [embedsImage, embedsText, embedsAudio]
		Returns:
			fused_embed: tensor of shape (batch_size, d_model) - the CLS token output
		"""
		batch_size = modal_embeds[0].shape[0]
		num_modals = len(modal_embeds)
		
		# Stack modal embeddings: (num_modals, batch_size, d_model)
		modal_stack = torch.stack(modal_embeds, dim=0)  # (num_modals, batch_size, d_model)
		
		# Add CLS token: (1, batch_size, d_model)
		cls_tokens = self.cls_token.expand(1, batch_size, -1)
		
		# Concatenate: (1+num_modals, batch_size, d_model)
		# Sequence: [CLS, Image, Text, Audio] or [CLS, Image, Text]
		x = torch.cat([cls_tokens, modal_stack], dim=0)
		
		# Add positional embedding
		# pos_embedding shape: (max_seq_len, 1, d_model)
		# Positions: [0 (CLS), 1 (Image), 2 (Text), 3 (Audio)]
		seq_len = 1 + num_modals  # CLS + num_modals
		pos_emb = self.pos_embedding[:seq_len]  # (seq_len, 1, d_model)
		pos_emb = pos_emb.expand(-1, batch_size, -1)  # (seq_len, batch_size, d_model)
		x = x + pos_emb
		
		# Apply transformer blocks
		for transformer_block in self.transformer_blocks:
			x = transformer_block(x)
		
		# Extract CLS token (first position)
		fused_embed = x[0]  # (batch_size, d_model)
		
		# No normalization here - let the residual connection handle it
		# The output will be scaled down in the residual connection anyway
		return fused_embed

class LatentEncoder(nn.Module):
	"""Encoder to map from original space to latent space"""
	def __init__(self, input_dim, latent_dim):
		super(LatentEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, (input_dim + latent_dim) // 2),
			nn.LeakyReLU(0.2),
			nn.Linear((input_dim + latent_dim) // 2, latent_dim),
			nn.LeakyReLU(0.2)
		)
		self.init_weights()
	
	def init_weights(self):
		for layer in self.encoder:
			if isinstance(layer, nn.Linear):
				size = layer.weight.size()
				std = np.sqrt(2.0 / (size[0] + size[1]))
				layer.weight.data.normal_(0.0, std)
				layer.bias.data.normal_(0.0, 0.001)
	
	def forward(self, x):
		return self.encoder(x)

class LatentDecoder(nn.Module):
	"""Decoder to map from latent space back to original space"""
	def __init__(self, latent_dim, output_dim):
		super(LatentDecoder, self).__init__()
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, (latent_dim + output_dim) // 2),
			nn.LeakyReLU(0.2),
			nn.Linear((latent_dim + output_dim) // 2, output_dim),
			nn.LeakyReLU(0.2)
		)
		self.init_weights()
	
	def init_weights(self):
		for layer in self.decoder:
			if isinstance(layer, nn.Linear):
				size = layer.weight.size()
				std = np.sqrt(2.0 / (size[0] + size[1]))
				layer.weight.data.normal_(0.0, std)
				layer.bias.data.normal_(0.0, 0.001)
	
	def forward(self, x):
		return self.decoder(x)
		
class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)
		if self.norm:
			x = F.normalize(x)
		if mess_dropout:
			x = self.drop(x)
		h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas) 

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, steps, sampling_noise=False, encoder=None, decoder=None):
		if args.use_latent_space and encoder is not None and decoder is not None:
			# Encode to latent space
			x_start_latent = encoder(x_start)
			if steps == 0:
				x_t_latent = x_start_latent
			else:
				t = torch.tensor([steps-1] * x_start_latent.shape[0]).cuda()
				x_t_latent = self.q_sample(x_start_latent, t)
			
			indices = list(range(self.steps))[::-1]

			for i in indices:
				t = torch.tensor([i] * x_t_latent.shape[0]).cuda()
				model_mean, model_log_variance = self.p_mean_variance(model, x_t_latent, t)
				if sampling_noise:
					noise = torch.randn_like(x_t_latent)
					nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t_latent.shape)-1))))
					x_t_latent = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
				else:
					x_t_latent = model_mean
			
			# Decode back to original space
			return decoder(x_t_latent)
		else:
			# Original behavior without latent space
			if steps == 0:
				x_t = x_start
			else:
				t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
				x_t = self.q_sample(x_start, t)
			
			indices = list(range(self.steps))[::-1]

			for i in indices:
				t = torch.tensor([i] * x_t.shape[0]).cuda()
				model_mean, model_log_variance = self.p_mean_variance(model, x_t, t)
				if sampling_noise:
					noise = torch.randn_like(x_t)
					nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1))))
					x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
				else:
					x_t = model_mean
			return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, t):
		model_output = model(x, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats, encoder=None, decoder=None):
		batch_size = x_start.size(0)

		if args.use_latent_space and encoder is not None and decoder is not None:
			# Encode to latent space
			x_start_latent = encoder(x_start)
			
			ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
			noise = torch.randn_like(x_start_latent)
			if self.noise_scale != 0:
				x_t_latent = self.q_sample(x_start_latent, ts, noise)
			else:
				x_t_latent = x_start_latent

			model_output_latent = model(x_t_latent, ts)
			
			# Decode back to original space for loss computation
			model_output = decoder(model_output_latent)

			mse = self.mean_flat((x_start - model_output) ** 2)

			weight = self.SNR(ts - 1) - self.SNR(ts)
			weight = torch.where((ts == 0), 1.0, weight)

			diff_loss = weight * mse

			usr_model_embeds = torch.mm(model_output, model_feats)
			usr_id_embeds = torch.mm(x_start, itmEmbeds)

			gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

			return diff_loss, gc_loss
		else:
			# Original behavior without latent space
			ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()
			noise = torch.randn_like(x_start)
			if self.noise_scale != 0:
				x_t = self.q_sample(x_start, ts, noise)
			else:
				x_t = x_start

			model_output = model(x_t, ts)

			mse = self.mean_flat((x_start - model_output) ** 2)

			weight = self.SNR(ts - 1) - self.SNR(ts)
			weight = torch.where((ts == 0), 1.0, weight)

			diff_loss = weight * mse

			usr_model_embeds = torch.mm(model_output, model_feats)
			usr_id_embeds = torch.mm(x_start, itmEmbeds)

			gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

			return diff_loss, gc_loss
		
	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))
	
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])