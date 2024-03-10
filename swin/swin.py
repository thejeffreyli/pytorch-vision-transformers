# modified from https://github.com/WangFeng18/Swin-Transformer
import math
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(w, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output
    
    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0], relation[:,:,1]]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None, activation_func=nn.GELU()):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            activation_func,
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

    
class SwinTransformer(nn.Module):
    """ Implementation of Swin Transformer https://arxiv.org/abs/2103.14030
    In this Implementation, the standard shape of data is (b h w c), which is a similar protocal as cnn.

    Note that this is a modification of the SWIN Transformer such that it upscales the image back up to
    (h w 3)

    Example from Swin_T: 
    - stages configuration = 2, 2, 6, 2 blocks per stage
    - dimension of the image: 
    """
    def __init__(self, num_classes, config=[2,2,6,2], dim=96, drop_path_rate=0.2, 
                 input_resolution=224, reconstruct=False, latent_dim = 512, activation_func=nn.GELU()):
        super(SwinTransformer, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 7
        self.reconstruct = reconstruct
        self.activation_func = activation_func

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        begin = 0
        self.stage1 = [nn.Conv2d(3, dim, kernel_size=4, stride=4),
                       Rearrange('b c h w -> b h w c'),
                       nn.LayerNorm(dim),] + \
                      [Block(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4, self.activation_func) 
                      for i in range(config[0])]
        begin += config[0]
        self.stage2 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(4*dim), nn.Linear(4*dim, 2*dim, bias=False),] + \
                      [Block(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8, self.activation_func)
                      for i in range(config[1])]
        begin += config[1]
        self.stage3 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(8*dim), nn.Linear(8*dim, 4*dim, bias=False),] + \
                      [Block(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//16, self.activation_func)
                      for i in range(config[2])]
        begin += config[2]
        self.stage4 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(16*dim), nn.Linear(16*dim, 8*dim, bias=False),] + \
                      [Block(8*dim, 8*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//32, self.activation_func)
                      for i in range(config[3])]
        if (self.reconstruct):
            # encode to be in latent dimension
            self.latent_mean = nn.Linear(8*dim, latent_dim)
            self.latent_var = nn.Linear(8*dim, latent_dim)
            begin += config[3]
            self.stage5 = [Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2), 
                            nn.LayerNorm(latent_dim // 4), nn.Linear(latent_dim // 4, 4*dim, bias=False),] + \
                            [Block(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8, self.activation_func)
                            for i in range(config[4])]
            begin += config[4]
            self.stage6 = [Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2), 
                            nn.LayerNorm(dim), nn.Linear(dim, 2*dim, bias=False),] + \
                            [Block(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8, self.activation_func)
                            for i in range(config[5])]
            begin += config[5]
            self.stage7 = [Rearrange('b h w (neiw neih c) -> b (h neih) (w neiw) c', neih=2, neiw=2), 
                            nn.LayerNorm(dim//2), nn.Linear(dim//2, dim, bias=False),] + \
                            [Block(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4, self.activation_func)
                            for i in range(config[6])]
            begin += config[6]
            self.stage8 = [nn.LayerNorm(dim),
                           Rearrange('b h w c -> b c h w'),
                            nn.ConvTranspose2d(dim, 3, kernel_size=4, stride=4)
                            ]

        self.stage1 = nn.Sequential(*self.stage1)
        self.stage2 = nn.Sequential(*self.stage2)
        self.stage3 = nn.Sequential(*self.stage3)
        self.stage4 = nn.Sequential(*self.stage4)
        
        if (self.reconstruct):
            self.stage5 = nn.Sequential(*self.stage5)
            self.stage6 = nn.Sequential(*self.stage6)
            self.stage7 = nn.Sequential(*self.stage7)
            self.stage8 = nn.Sequential(*self.stage8)
            self.norm_last = nn.LayerNorm(dim)
        elif (not self.reconstruct):
            self.norm_last = nn.LayerNorm(dim * 8)
            self.mean_pool = Reduce('b h w c -> b c', reduction='mean')
            self.mlp = nn.Sequential(
                nn.Linear(dim * 8, dim * 16),
                self.activation_func,
                nn.Linear(dim * 16, dim * 8),
            )
            self.classifier = nn.Linear(8*dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reparameterization(self, mean, var):
        '''
        Reparametrization trick
        Reference: https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
        '''
        if (torch.cuda.is_available()):
            epsilon = torch.randn_like(var).cuda() #.to(device)      
        else:
            epsilon = torch.randn_like(var).cuda()
        z = mean + var*epsilon
        return z

    def forward(self, x):
        # encoder
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        if (self.reconstruct == True):
            mu = self.latent_mean(x)
            sigma = self.latent_var(x)
            z = self.reparameterization(mu, torch.exp(0.5 * sigma))
            # decoder
            x = self.stage5(z) # Note that this is z, not x
            x = self.stage6(x)
            x = self.stage7(x)
            x = self.norm_last(x)
            x = self.stage8(x)
            # Apply sigmoid ==> No more sigmoid since Sigmoid + BCE is unstable
            # x = torch.sigmoid(x)
            return x, mu, sigma
        elif (not self.reconstruct):
            x = self.norm_last(x)
            x = self.mlp(x) # add MLP block ==> linear layer with activation layers in between
            x = self.mean_pool(x)
            x = self.classifier(x)
            return x

def Swin_T_reconstruct(num_classes, config=[2,2,6,2,2,6,2,2], dim=96, **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, reconstruct=True, **kwargs)

def Swin_T(num_classes, config=[2,2,6,2], dim=96, **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

def Swin_S(num_classes, config=[2,2,18,2], dim=96, **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

def Swin_B(num_classes, config=[2,2,18,2], dim=128, **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

def Swin_L(num_classes, config=[2,2,18,2], dim=192, **kwargs):
    return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

if __name__ == '__main__':
    test_model = Swin_T(1000).cuda()
    n_parameters = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(test_model)
    dummy_input = torch.rand(3,3,224,224).cuda()
    output = test_model(dummy_input)
    print(output.size())
    # flops, params = profile(test_model, inputs=(dummy_input, ))
    # print(params)
    # print(flops)
    print(n_parameters)
