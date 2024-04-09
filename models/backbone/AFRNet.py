import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
import sys
import os
from units import Conv2d
from localization_network import LocalizationNetwork
from vit import ViT
import os
import cv2


class AFRNet(nn.Module):
    def __init__(self,num_classes):
        super(AFRNet, self).__init__()

        self.localization_net = LocalizationNetwork()
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.resnet50.children())[:-2]) 
        self.ViT_head = ViT(
            image_size = 14,   
            patch_size = 1,    
            dim = 384,       
            depth = 12,         
            heads = 12,        
            mlp_dim = 1024,     
            channels = 1024
        )
        self.CNN_head = nn.Sequential(
            Conv2d(2048,1024,5,1,1),
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.CNN_dim_1 = nn.Linear(1024, 384)


    def forward(self, x):
        x = self.localization_net(x)
        aligned_img = x
        x = self.backbone(x)
        x = self.CNN_head(x)
        #CNN_Zc
    
        immediate_feature_map = x

        CNN_Zc = self.adaptive_avg_pool(x)
        CNN_Zc = torch.flatten(CNN_Zc, 1)
        CNN_Zc = self.dropout(CNN_Zc)
        CNN_Zc = self.CNN_dim_1(CNN_Zc)
        CNN_Zc = torch.nn.functional.normalize(CNN_Zc, dim=1)

        #attention_Za
        attention_Za = self.ViT_head(x)
        attention_Za = torch.nn.functional.normalize(attention_Za, dim=1)

        return CNN_Zc, attention_Za, immediate_feature_map, aligned_img
