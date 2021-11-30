import torch.nn as nn
import torch
import os 

import argparse
import math
from pathlib import Path
import sys

sys.path.append('../data_augmentation/taming-transformers')
sys.path.append('../data_augmentation/')

sys.path.append('./taming-transformers')
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
 

import numpy as np
import imageio
from PIL import ImageFile, Image

import json
ImageFile.LOAD_TRUNCATED_IMAGES = True

 
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    print(config.model.params.first_stage_config.params.ddconfig.dropout)
    #print(config.model.target)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model #parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    return model

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


class GPT2ClfImage(nn.Module):
    def __init__(self, args):
        super(GPT2ClfImage, self).__init__()
        self.args = args
        model=load_vqgan_model('../data_augmentation/imgnt.yaml','../data_augmentation/imgnt.ckpt')
        self.tok_emb=model.transformer.tok_emb
        self.pos_emb=model.transformer.pos_emb
        self.drop= model.transformer.drop
        self.blocks=model.transformer.blocks
        self.ln_f= model.transformer.ln_f
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(1536, args.n_classes,bias=False)
        
    def forward(self,image):
        token_embeddings=self.tok_emb(image)
        t = token_embeddings.shape[1]
        position_embeddings =self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.dropout(x[:,-1,:])
        out = self.classifier(x)
              
        return out