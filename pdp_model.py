import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model.model_base import Bottleneck
from einops.layers.torch import Rearrange
import pdb


class ModelSpatial_PDP(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, args):
        block = Bottleneck
        self.patch_num = 7
        super(ModelSpatial_PDP, self).__init__()
        self.input_resolution = 224
        self.inplanes_scene = 64
        self.inplanes_face = 64
         # common
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_final = nn.AdaptiveMaxPool1d(1, return_indices=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        layers_scene = [3, 4, 6, 3, 2]
        layers_face = [3, 4, 6, 3, 2]
        self.use_patch, self.use_depth = args.use_patch, args.use_depth
        # scene pathway
        if self.use_depth:
            self.conv1_scene = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)            
        else: 
            self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50
        #self.contract_feat = nn.Conv2d(kernel_size=3, stride=2, padding=1)
        # face pathway

        self.conv1_face = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50
      

        # attention
        if self.use_depth:
            self.attn = nn.Linear(2592, 1*7*7)    
        else:
            self.attn = nn.Linear(1808, 1*7*7)

        # encoding for saliency  # modify from 2048 to 1024
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)  # previous: conv2d
        self.outside_embedding = nn.Parameter(torch.randn(1, 512)) # original pdp
        
        self.patch_pos_embedding = nn.Parameter(torch.randn(50, 512)) # original pdp
        
        self.qkv_proj = nn.Linear(512, 512*3)
        self.inout_patch_encode = nn.Conv1d(512,256, kernel_size=1)
        self.inout_patch_pred = nn.Conv1d(256,1, kernel_size=1)

        # decoding
        self.deconv_encode = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)

    def patch_attention(self, input):
        q,k,v = torch.chunk(self.qkv_proj(input), 3, dim=2)
        attn_score_unnorm = torch.bmm(q, k.transpose(1, 2).contiguous())
        attn_scores = F.softmax(attn_score_unnorm, dim=2)
        
        value = torch.bmm(attn_scores, v)
        value = input+value  # for both 
        new_input = value        
        return new_input


    def forward(self, input, hidden_scene=None):
        images, head, face, depth_img = input

        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        bs = images.size()[0]

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
            
        # get and reshape attention weights such that it can be multiplied with scene feature map
        if self.use_depth:
            depth_reduced = self.maxpool(self.maxpool(self.maxpool(depth_img))).view(-1, 784)
            attn_weights = self.attn(torch.cat((head_reduced, depth_reduced, face_feat_reduced), 1))
        else:
            attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        if self.use_depth:
            im = torch.cat((images, depth_img, head), dim=1)
        else:
            im = torch.cat((images, head), dim=1)
            
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)

        # encode scene + face feat
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        
        encoding = encoding.flatten(start_dim=2)
        encoding = torch.transpose(encoding.flatten(start_dim=2), 1, 2).contiguous()
        encoding = torch.cat((encoding, self.outside_embedding.unsqueeze(0).expand(bs, -1, -1)), dim=1) # bsx50x512
        
        encoding = encoding + self.patch_pos_embedding.unsqueeze(0).expand(bs, -1, -1)
        encoding = self.patch_attention(encoding)
        encoding = encoding.transpose(1,2)
    
        inout_encoding = self.inout_patch_encode(encoding)
        inout_encoding = self.relu(inout_encoding)
        pred_inout_patches = self.inout_patch_pred(inout_encoding).view(bs,-1)
    
        _, c, d = encoding.size()
        encoding = encoding[:,:,:-1].clone()
        encoding = encoding.view(-1, c, 7, 7)
        encoding = self.deconv_encode(encoding)
        encoding = self.relu(encoding)
        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x, pred_inout_patches



class ModelSpatialTemporal_PDP(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, args, seq_len=5):
        block = Bottleneck
        super(ModelSpatialTemporal_PDP, self).__init__()
        self.input_resolution = 224
        self.inplanes_scene = 64
        self.inplanes_face = 64
         # common
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_final = nn.AdaptiveMaxPool1d(1, return_indices=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        layers_scene = [3, 4, 6, 3, 2]
        layers_face = [3, 4, 6, 3, 2]
        self.use_patch, self.use_depth = args.use_patch, args.use_depth
        self.use_temporal_att = args.use_temporal_att
        self.seq_len = seq_len
        self.patch_num = 7
        
        # scene pathway
        if self.use_depth:
            self.conv1_scene = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)            
        else: 
            self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway       
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50
        
        # attention
        if self.use_depth:
            self.attn = nn.Linear(2592, 1*7*7)    
        else:
            self.attn = nn.Linear(1808, 1*7*7)

        # encoding for saliency  # modify from 2048 to 1024
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        # modify here,
        self.compress_conv1_inout = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)  # previous: conv2d
        self.outside_embedding = nn.Parameter(torch.randn(1, 512))
        self.patch_pos_embedding = nn.Parameter(torch.randn(50, 512))
        self.fc_inout = nn.Linear(50,1)
        
        self.qkv_proj = nn.Linear(512, 512*3)
        self.inout_patch_encode = nn.Conv1d(512,256, kernel_size=1)
        self.inout_patch_pred = nn.Conv1d(256,1, kernel_size=1)
        self.squeeze_channel=1
        if self.use_temporal_att:
            # attention across temporal dimension
            self.temporal_pos_embed = nn.Parameter(torch.randn(self.seq_len, 1, (self.patch_num*self.patch_num+1)*self.squeeze_channel))
            self.temporal_mapping = nn.Conv1d(512, self.squeeze_channel, kernel_size=1)
            self.temporal_attention = nn.MultiheadAttention((self.patch_num*self.patch_num+1)*self.squeeze_channel, 1)
            self.temporal_proj_1 = nn.Conv1d(512, 512, kernel_size=1)
            self.temporal_norm_1 = nn.LayerNorm([512, self.patch_num*self.patch_num+1])

        # decoding
        self.deconv_encode = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)
    
    def patch_attention(self, input):
        q,k,v = torch.chunk(self.qkv_proj(input), 3, dim=2)
        attn_score_unnorm = torch.bmm(q, k.transpose(1, 2).contiguous())
        attn_scores = F.softmax(attn_score_unnorm, dim=2)
        
        value = torch.bmm(attn_scores, v)
        value = input+value  # for both 
        return value

    def forward(self, input, hidden_scene=None):
        images, head, face, depth_img = input
        bs, T, c, h, w= images.size()
        images, face = images.view(-1,c,h,w), face.view(-1,c,h,w)
        head, depth_img = head.view(-1,1,h,w), depth_img.view(-1, 1, h, w)

        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        
        # get and reshape attention weights such that it can be multiplied with scene feature map
        if self.use_depth:
            depth_reduced = self.maxpool(self.maxpool(self.maxpool(depth_img))).view(-1, 784)
            attn_weights = self.attn(torch.cat((head_reduced, depth_reduced, face_feat_reduced), 1))
        else:
            attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, self.patch_num*self.patch_num)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, self.patch_num, self.patch_num)

        if self.use_depth:
            im = torch.cat((images, depth_img, head), dim=1)
        else:
            im = torch.cat((images, head), dim=1)
            
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
        
        # scene + face feat -> in/out
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)

        encoding = torch.transpose(encoding.flatten(start_dim=2), 1, 2).contiguous()
        encoding = torch.cat((encoding, self.outside_embedding.unsqueeze(0).expand(bs*T, -1, -1)), dim=1) # bsx50x512
        encoding = encoding + self.patch_pos_embedding.unsqueeze(0).expand(bs*T, -1, -1)
        encoding = self.patch_attention(encoding).transpose(1,2)  # original: all encoding_inout are inout_patch_input
        
        if self.use_temporal_att:
            _, c, d = encoding.size()
            out_feat_input = self.temporal_mapping(encoding).squeeze(1).view(bs, T, -1)
            encoding = encoding.reshape(bs, T, c, -1).contiguous().view(bs,T,-1)
            out_feat_input = out_feat_input.transpose(0,1).contiguous() + self.temporal_pos_embed
            out_feat_input, temporalatt_weights = self.temporal_attention(out_feat_input, out_feat_input, out_feat_input, need_weights=True)
            
            encoding = encoding.view(-1,c,d) # added
            encoding_value = self.temporal_proj_1(encoding).view(bs,T,-1) # added
            out_feat_add = torch.bmm(temporalatt_weights, encoding_value).view(bs,T,c,d).view(-1,c,d) # original: encoding
            
            encoding = encoding + out_feat_add
            encoding = self.temporal_norm_1(encoding)
            
        if self.use_patch:
            inout_encoding = self.inout_patch_encode(encoding)
            inout_encoding = self.relu(inout_encoding)
            pred_inout_patches = self.inout_patch_pred(inout_encoding).view(bs*T,-1)
        
        # original inout branch, just for comparison
        encoding_inout = self.compress_conv1_inout(encoding)
        encoding_inout = self.relu(encoding_inout).view(bs*T, -1)
        encoding_inout =  self.fc_inout(encoding_inout)

        # scene + face feat -> encoding -> decoding
        
        _, c, d = encoding.size()
        encoding = encoding[:,:,:-1].clone()
        encoding = encoding.view(-1, c, self.patch_num, self.patch_num)
        encoding = self.deconv_encode(encoding)
        encoding = self.relu(encoding)

        x = self.deconv1(encoding)
        x = self.deconv_bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.deconv_bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.deconv_bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        return x, pred_inout_patches, encoding_inout





