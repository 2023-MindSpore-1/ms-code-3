# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
import numpy as np
from scipy import ndimage

import models.configs as configs


from .modeling_resnet import ResNetV2
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import x2ms_adapter
import x2ms_adapter.nn
import x2ms_adapter.nn_functional
import x2ms_adapter.nn_init
import x2ms_adapter.util_api as util_api


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = x2ms_adapter.tensor_api.transpose(weights, [3, 2, 0, 1])
    return x2ms_adapter.from_numpy(weights)


def swish(x):
    return x * x2ms_adapter.sigmoid(x)


ACT2FN = {"gelu": x2ms_adapter.nn_functional.gelu, "relu": x2ms_adapter.nn_functional.relu, "swish": swish}

class Attention(nn.Cell):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = x2ms_adapter.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = x2ms_adapter.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = x2ms_adapter.nn.Linear(config.hidden_size, self.all_head_size)

        self.out = x2ms_adapter.nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = x2ms_adapter.nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = x2ms_adapter.nn.Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = x2ms_adapter.nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x2ms_adapter.tensor_api.size(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(self, hidden_states, hidden_states_k):
        mixed_query_layer = self.query(hidden_states)

        mixed_key_layer = self.key(hidden_states_k)

        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = x2ms_adapter.matmul(query_layer, x2ms_adapter.tensor_api.transpose(key_layer, -1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = x2ms_adapter.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = x2ms_adapter.tensor_api.size(context_layer)[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Cell):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = x2ms_adapter.nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = x2ms_adapter.nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = x2ms_adapter.nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        x2ms_adapter.nn_init.xavier_uniform_(self.fc1.weight)
        x2ms_adapter.nn_init.xavier_uniform_(self.fc2.weight)
        x2ms_adapter.nn_init.normal_(self.fc1.bias, std=1e-6)
        x2ms_adapter.nn_init.normal_(self.fc2.bias, std=1e-6)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Cell):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            #patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size = (1, 1)
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
            
        else:
            patch_size = util_api.pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = x2ms_adapter.nn.Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = mindspore.Parameter(x2ms_adapter.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = mindspore.Parameter(x2ms_adapter.zeros(1, 1, config.hidden_size))

        self.dropout = x2ms_adapter.nn.Dropout(config.transformer["dropout_rate"])

    def construct(self, x):
        B = x.shape[0]
        cls_tokens = x2ms_adapter.tensor_api.expand(self.cls_token, B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        
        x = x2ms_adapter.tensor_api.flatten(x, 2)
        x = x2ms_adapter.tensor_api.transpose(x, -1, -2)
        
        x = x2ms_adapter.cat((cls_tokens, x), dim=1)

        
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Cell):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm_1 = x2ms_adapter.nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attention_norm_2 = x2ms_adapter.nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = x2ms_adapter.nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def construct(self, x, k):
        h = x
        x = self.attention_norm_1(x)
        k = self.attention_norm_2(k)
        x, weights = self.attn(x, k)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
        out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

        query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
        key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
        value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
        out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

        self.attn.query.weight.copy_(query_weight)
        self.attn.key.weight.copy_(key_weight)
        self.attn.value.weight.copy_(value_weight)
        self.attn.out.weight.copy_(out_weight)
        self.attn.query.bias.copy_(query_bias)
        self.attn.key.bias.copy_(key_bias)
        self.attn.value.bias.copy_(value_bias)
        self.attn.out.bias.copy_(out_bias)

        mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
        mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
        mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
        mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

        self.ffn.fc1.weight.copy_(mlp_weight_0)
        self.ffn.fc2.weight.copy_(mlp_weight_1)
        self.ffn.fc1.bias.copy_(mlp_bias_0)
        self.ffn.fc2.bias.copy_(mlp_bias_1)

        self.attention_norm_1.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
        self.attention_norm_1.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))

        self.attention_norm_2.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
        self.attention_norm_2.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))

        self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
        self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Cell):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = x2ms_adapter.nn.ModuleList()
        self.encoder_norm = x2ms_adapter.nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def construct(self, hidden_states):
        attn_weights = []
        k = hidden_states
        for layer_block in self.layer:
            k1 = hidden_states
            hidden_states, weights = layer_block(hidden_states, k)
            k=k1
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Cell):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def construct(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Cell):
    def __init__(self, config, img_size=(128,512), vis=False):
        super(VisionTransformer, self).__init__()

        self.gs_new_h = img_size[0] // 16#config.patches.grid[0]
        self.gs_new_w = img_size[1] // 16#config.patches.grid[1]
        self.transformer = Transformer(config, img_size, vis)

    def construct(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        out = x[:, 0]
        l2_normalize = ops.L2Normalize(axis=1)
        return l2_normalize(out)
    
    def load_from(self, weights):
        print('loading model' )
        
        self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
        self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
        self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
        
        self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
        self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

        
        posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
        posemb_new = self.transformer.embeddings.position_embeddings
        if x2ms_adapter.tensor_api.size(posemb) == x2ms_adapter.tensor_api.size(posemb_new):
            self.transformer.embeddings.position_embeddings.copy_(posemb)
        else:
            logger.info("load_pretrained: resized variant: %s to %s" % (x2ms_adapter.tensor_api.size(posemb), x2ms_adapter.tensor_api.size(posemb_new)))
            ntok_new = x2ms_adapter.tensor_api.size(posemb_new, 1)
            
            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
            
            gs_old = int(np.sqrt(len(posemb_grid)))
            
            print('load_pretrained: grid-size from (%s %s) to (%s %s)' % (gs_old, gs_old, self.gs_new_h, self.gs_new_w))
            
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

            zoom = (self.gs_new_h / gs_old, self.gs_new_w / gs_old, 1)
            posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
            posemb_grid = posemb_grid.reshape(1, self.gs_new_h * self.gs_new_w, -1)
            posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
            self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
        
        for bname, block in self.transformer.encoder.named_children():
            for uname, unit in block.named_children():
                unit.load_from(weights, n_block=uname)

        if self.transformer.embeddings.hybrid:
            self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            gn_weight = np2th(weights["gn_root/scale"]).view(-1)
            gn_bias = np2th(weights["gn_root/bias"]).view(-1)
            self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
            self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

            for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)
        

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
