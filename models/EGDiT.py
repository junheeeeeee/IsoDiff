import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
from einops import rearrange
from functools import partial
from timm.models.vision_transformer import Attention, Mlp
from models.ROPE import RopeND
from utils.eval_utils import eval_decorator
from utils.train_utils import lengths_to_mask
from diffusions.diffusion import create_diffusion
from diffusions.transport import create_transport, Sampler

#################################################################################
#                                      DM                                    #
#################################################################################
class DM(nn.Module):
    def __init__(self, input_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0, clip_dim=512,
                 diff_model='Flow', cond_drop_prob=0.1, max_length=49, num_joint=22,
                 clip_version='ViT-B/32', cross = True, end = True,con = False,**kargs):
        super(DM, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.cross = cross
        self.end = end
        self.con = con
        self.patches_per_frame = 1
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
            self.num_actions = kargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        # --------------------------------------------------------------------------
        # Diffusion
        self.diff_model = diff_model
        if self.diff_model == 'Flow':
            self.train_diffusion = create_transport()  # default to linear, velocity prediction
            self.gen_diffusion = Sampler(self.train_diffusion)
        else:
            self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="linear")
            self.gen_diffusion = create_diffusion(timestep_respacing="", noise_schedule="linear")
        # --------------------------------------------------------------------------
        self.t_embedder = TimestepEmbedder(self.latent_dim)

        # Patchification
        self.x_embedder = nn.Linear(self.input_dim, self.latent_dim)

        # Positional Encoding
        if self.con:
            max_length +=1
        self.max_lens = [max_length]
        self.rope = RopeND(nd=1, nd_split=[1], max_lens=self.max_lens)
        self.position_ids_precompute = torch.arange(max_length).unsqueeze(0)


        self.DiT = nn.ModuleList([
            DiTBlock(self.latent_dim, num_heads, mlp_size=ff_size, rope=self.rope, qk_norm=True, end= self.end) for _ in range(num_layers)
        ])

        if self.cond_mode == 'text':
            # self.y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
            # self.clip_proj = nn.Linear(self.clip_dim, self.latent_dim)

            self.y_embedder = zero_padding(self.clip_dim, self.latent_dim)
            self.clip_proj = zero_padding(self.clip_dim, self.latent_dim)

        elif self.cond_mode == 'action':
            self.y_embedder = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.y_embedder = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.layer = AdaLN(self.latent_dim, self.input_dim, end = self.end)

        self.initialize_weights()

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in ACMDM blocks:
        for block in self.DiT:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.layer.linear.weight, 0)
        nn.init.constant_(self.layer.linear.bias, 0)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)
        assert torch.cuda.is_available()
        clip.model.convert_weights(clip_model)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)

        text_len = text.argmax(dim=-1)
        
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype) + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        word_tokens = (self.clip_model.ln_final(x) @ self.clip_model.text_projection).type(torch.float32)
        eot_token = word_tokens[torch.arange(word_tokens.shape[0]), text_len]
        
        mask = torch.arange(text.shape[1], device=device)[None, :] <= text_len[:, None]
        word_tokens = word_tokens * mask[:, :, None]

        return word_tokens, eot_token, mask.unsqueeze(1).unsqueeze(1)

    def mask_cond(self, cond, full_cond,force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond), torch.zeros_like(full_cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask), full_cond * (1. - mask.unsqueeze(-1))
        else:
            return cond, full_cond

    def forward(self, x, t, conds, attention_mask, full_cond, text_mask,force_mask=False):
        t = self.t_embedder(t, dtype=x.dtype)
        conds, full_cond = self.mask_cond(conds, full_cond, force_mask=force_mask)
        
        x = self.x_embedder(x)

        conds = self.y_embedder(conds)
        if self.con:
            x = torch.concat([conds.unsqueeze(1), x], dim=1)
            attention_mask = torch.concat([torch.ones_like(attention_mask[:, :, :, :1]).bool(), attention_mask], dim=-1)
        b, f, d = x.shape
        
        if self.end:
            y = torch.concat([conds, t], dim=-1).unsqueeze(1)
        else:
            y = t.unsqueeze(1)

        position_ids = self.position_ids_precompute[:, :x.shape[1]]
        if not self.cross:
            text_mask = torch.zeros_like(text_mask).bool()
        else:
            full_cond = self.clip_proj(full_cond)

        for block in self.DiT:
            x = block(x, y, attention_mask, position_ids=position_ids, full_cond=full_cond, text_mask=text_mask)
        if self.con:
            x = x[:, 1:, :]
        x = self.layer(x, y)
        return x

    def forward_with_CFG(self, x, t, conds, attention_mask, full_cond, text_mask, cfg=1.0):
        if not cfg == 1.0:
            half = x[: len(x) // 2]
            x = torch.cat([half, half], dim=0)

        x = self.forward(x, t, conds, attention_mask, full_cond, text_mask)
        if not cfg == 1.0:
            cond_eps, uncond_eps = torch.split(x, len(x) // 2, dim=0)
            half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
            x = torch.cat([half_eps, half_eps], dim=0)
        return x

    def forward_loss(self, latents, y, m_lens):
        b, l, _ = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1), latents, torch.zeros_like(latents))

        target = latents.clone().detach()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                full_cond, cond_vector, text_mask  = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        attention_mask = non_pad_mask.unsqueeze(-1).repeat(1, 1, self.patches_per_frame).flatten(1).unsqueeze(1).unsqueeze(1)

        model_kwargs = dict(conds=cond_vector, force_mask=force_mask, attention_mask=attention_mask, full_cond=full_cond, text_mask=text_mask)
        if self.diff_model == "Flow":
            loss_dict = self.train_diffusion.training_losses(self.forward, target, model_kwargs)
        else:
            t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
            loss_dict = self.train_diffusion.training_losses(self.forward, target, t, model_kwargs)
        loss = loss_dict["loss"]
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()

        return loss

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 cond_scale: int,
                 temperature=1,
                 j=22,
                 ):
        device = next(self.parameters()).device
        l = max(m_lens)
        b = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                full_cond, cond_vector, text_mask = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        noise = torch.randn(b, l, self.input_dim).to(device)
        # noise = noise[:,:,:1].repeat_interleave(l, dim=2)
        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=0)
            full_cond = torch.cat([full_cond, torch.zeros_like(full_cond)], dim=0)
            noise = torch.cat([noise, noise], dim=0)

        attention_mask = (~padding_mask).unsqueeze(-1).repeat(1,1,self.patches_per_frame).flatten(1).unsqueeze(1).unsqueeze(1)
        model_kwargs = dict(conds=cond_vector, attention_mask=attention_mask, full_cond= full_cond, text_mask = text_mask ,cfg=cond_scale)
        sample_fn = self.forward_with_CFG
        

        if not cond_scale == 1:
            model_kwargs["attention_mask"] = attention_mask.repeat(2, 1, 1, 1)
            model_kwargs["text_mask"] = text_mask.repeat(2, 1, 1, 1)

        if self.diff_model == "Flow":
            model_fn = self.gen_diffusion.sample_ode(sampling_method = "euler")  # {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint", "heun2", "heun3", "rk4", "explicit_adams", "implicit_adams", "fixed_adams", "scipy_solver"}
            sampled_token_latent = model_fn(noise, sample_fn, **model_kwargs)[-1]
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs,
                progress=False,
                temperature=temperature
            )
        if not cond_scale == 1:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)

        # sampled_token_latent = sampled_token_latent.permute(0,2,3,1)
        latents = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(sampled_token_latent), sampled_token_latent)
        return latents

#################################################################################
#                                     DM Zoos                                #
#################################################################################
def dm_raw(**kwargs):
    layer = 8
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, **kwargs)
def dm_xl(**kwargs):
    layer = 16
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, **kwargs)

def dm_raw_nocross(**kwargs):
    layer = 8
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, cross = False, **kwargs)

def dm_raw_crossonly(**kwargs):
    layer = 8
    return DM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196, end = False, **kwargs)

DM_models = { 
    'DM-Raw': dm_raw, "DM-Raw_nocross" :dm_raw_nocross , "DM-Raw_crossonly" : dm_raw_crossonly, 'DM-XL': dm_xl
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class CrossAttention(nn.Module):
    def __init__(
        self,
        head_dim,
        num_heads=8,
        qk_norm=True,
        norm_layer=None,
    ):
        super(CrossAttention, self).__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.q = nn.Linear(self.head_dim, self.head_dim)
        self.k = nn.Linear(self.head_dim, self.head_dim)
        self.v = nn.Linear(self.head_dim, self.head_dim)
        self.proj = nn.Linear(self.head_dim, self.head_dim)
        self.q_norm = norm_layer(self.head_dim// num_heads) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim// num_heads) if qk_norm else nn.Identity()
        self.proj_drop = nn.Dropout(0.)
        self.attn_drop = nn.Dropout(0.)

    def forward(self, x, cond, attention_mask=None):
        B, N, C = x.shape
        _, M, _ = cond.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(cond).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(cond).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)
        
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop.p
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention_r(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        rope=None,
        qk_norm=True,
        **block_kwargs,
    ):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, **block_kwargs)
        self.rope = rope

    def forward(self, x, position_ids=None, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop.p
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, rope=None, qk_norm=True, end = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention_r(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=nn.LayerNorm,
                                        qk_norm=qk_norm, rope=rope)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross = CrossAttention(hidden_size, num_heads=num_heads, norm_layer=nn.LayerNorm,
                                        qk_norm=qk_norm)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(2 / 3 * mlp_size), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        if end:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size*2, 9 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 9 * hidden_size, bias=True)
            )

    def forward(self, x, c ,attention_mask=None, full_cond = None, text_mask = None, position_ids=None):
        dtype = x.dtype
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=-1)
        norm_x1 = self.norm1(x.to(torch.float32)).to(dtype)
        attn_input_x = modulate(norm_x1, shift_msa, scale_msa)
        attn_output_x = self.attn(attn_input_x, attention_mask=attention_mask, position_ids=position_ids)
        x = x + gate_msa * attn_output_x

        if full_cond is not None:
            norm_x2 = self.norm2(x.to(torch.float32)).to(dtype)
            cross_input_x = modulate(norm_x2, shift_mca, scale_mca)
            cross_output_x = self.cross(cross_input_x, full_cond, attention_mask=text_mask)
            x = x + gate_mca * cross_output_x

        norm_x3 = self.norm3(x.to(torch.float32)).to(dtype)
        gate_input_x = modulate(norm_x3, shift_mlp, scale_mlp)
        gate_output_x = self.mlp(gate_input_x)
        x = x + gate_mlp * gate_output_x
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=torch.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class GroupedFeatureTransformer(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        
        self.groups = [
            [0],
            [2, 5, 8, 11],
            [1, 4, 7, 10],
            [3, 6, 9, 12, 15],
            [14, 17, 19, 21],
            [13, 16, 18, 20]
        ]
        
        self.linears = nn.ModuleList()
        for group in self.groups:
            input_size_for_group = len(group) * 3 * 4
            self.linears.append(nn.Linear(input_size_for_group, output_dim))
        
        self._initialize_weights()

    def _initialize_weights(self):
        for linear_layer in self.linears:
            input_dim = linear_layer.in_features
            output_dim = linear_layer.out_features
            
            weight_matrix = torch.zeros(output_dim, input_dim)
            min_dim = min(input_dim, output_dim)
            weight_matrix[:min_dim, :min_dim] = torch.eye(min_dim)
            
            linear_layer.weight.data = weight_matrix
            if linear_layer.bias is not None:
                linear_layer.bias.data.zero_()

    def forward(self, x):
        b, t, j, d = x.shape
        # (b, t, 22, 3) -> (b * t/4, 22, 12)
        x = x.reshape(b * t // 4, j, d * 4) 
        outputs = []
        for i, group in enumerate(self.groups):
            group_features = x[..., group, :]
            batch_size = group_features.size(0)
            flattened_features = group_features.view(batch_size, -1)
            output = self.linears[i](flattened_features)
            outputs.append(output)
            
        # (b, t/4, 6, 256)
        out = torch.stack(outputs, dim=1).reshape(b, t // 4, len(self.groups), -1)
        return out

class GroupedFeatureReconstructor(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        
        self.groups = [
            [0],
            [2, 5, 8, 11],
            [1, 4, 7, 10],
            [3, 6, 9, 12, 15],
            [14, 17, 19, 21],
            [13, 16, 18, 20]
        ]
        
        self.linears = nn.ModuleList()
        for group in self.groups:
            output_size_for_group = len(group) * 3 * 4
            self.linears.append(nn.Linear(input_dim, output_size_for_group))

        self.norm_final = nn.LayerNorm(input_dim, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, 2 * input_dim, bias=True)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for linear_layer in self.linears:
            input_dim = linear_layer.in_features
            output_dim = linear_layer.out_features
            
            weight_matrix = torch.zeros(output_dim, input_dim)
            min_dim = min(input_dim, output_dim)
            weight_matrix[:min_dim, :min_dim] = torch.eye(min_dim)
            
            linear_layer.weight.data = weight_matrix
            if linear_layer.bias is not None:
                linear_layer.bias.data.zero_()

    def forward(self, x, c):
        b, t, j, d = x.shape
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        batch_size = b * t
        x = x.reshape(batch_size, j, d)
        
        # 복원된 텐서를 저장할 빈 텐서를 생성합니다.
        # (b * t/4, 22, 12)
        reconstructed_tensor = torch.zeros(batch_size, 22, 3 * 4, device=x.device, dtype=x.dtype)
        
        for i, group in enumerate(self.groups):
            group_input = x[:, i, :]
            
            # 완벽한 복원을 위해 sin 활성화 함수는 사용하지 않습니다.
            transformed_output = self.linears[i](group_input)
            
            reshaped_output = transformed_output.view(batch_size, len(group), -1)
            reconstructed_tensor[:, group, :] = reshaped_output

        return reconstructed_tensor.reshape(b, -1, 22, 3)
    
class AdaLN(nn.Module):
    def __init__(self, hidden_size, output_size, end = False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)

        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        if end:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size*2, 2 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        norm_x = self.norm_final(x.to(torch.float32)).to(x.dtype)
        x = modulate(norm_x, shift, scale)
        x = self.linear(x)
        return x
#################################################################################

class zero_padding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
    def forward(self, x):
        d = x.shape[-1]
        if d >= self.output_dim:
            return x[...,:self.output_dim]
        else:
            padding = torch.zeros(*x.shape[:-1], self.output_dim - d, device=x.device, dtype=x.dtype)
            return torch.cat([x, padding], dim=-1)
        