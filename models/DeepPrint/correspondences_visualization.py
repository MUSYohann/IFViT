import sys
sys.path.append('../../models/')
sys.path.append('../../data_process/')
from DeepPrint import DeepPrint
import os
import numpy as np
import random
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cs
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import cv2
from einops.einops import rearrange
import torch.nn.functional as F
from plotting import make_matching_figure
import matplotlib.cm as cm

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed)
random.seed(seed)

# load the trained_model
def load_DeepPrint(save_last_model_path, classes):
    DeepPrint_model = DeepPrint(num_classes=classes)
    DeepPrint_model = DeepPrint_model.eval().cuda()

    #load the model
    model_checkpoint = None
    if os.path.exists(save_last_model_path):
        model_checkpoint = torch.load(save_last_model_path)
        DeepPrint_model.load_state_dict(model_checkpoint['model_state_dict'])
        print("Load the latest model")
    return DeepPrint_model

def load_input_data(fg_1, fg_2):
    fg_1 = Image.open(fg_1).convert('RGB')
    fg_2 = Image.open(fg_2).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    fg_1 = transform(fg_1)
    fg_2 = transform(fg_2)

    fg_1 = fg_1.unsqueeze(0)
    fg_2 = fg_2.unsqueeze(0)

    return fg_1, fg_2

def model_evaluation(fg_1_path, fg_2_path):


    DeepPrint_model = load_DeepPrint()
    
    fg_1, fg_2 = load_input_data(fg_1_path, fg_2_path)

    Representation_1, texture_output_1, M_output_1, D_output_1, aligned_fingerprint,CNN_feature_map_1 = DeepPrint_model(fg_1.cuda())
    Representation_2, texture_output_2, M_output_2, D_output_2, aligned_fingerprint,CNN_feature_map_2 = DeepPrint_model(fg_2.cuda())
    print(CNN_feature_map_1.shape)
    features1_flattened = CNN_feature_map_1.detach().cpu().view(-1, 384)
    features2_flattened = CNN_feature_map_2.detach().cpu().view(-1, 384)

    vec1_expanded = features1_flattened.unsqueeze(0) 
    vec2_expanded = features2_flattened.unsqueeze(0)

    sim_matrix = torch.einsum("nlc,nsc->nls", vec1_expanded, 
                            vec2_expanded) / 0.1
    conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
    print(sim_matrix)
    print(conf_matrix)
    mask = conf_matrix > 0
    mask = mask \
    * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
    * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v) 
    j_ids = all_j_ids[b_ids, i_ids] 
    mconf = conf_matrix[b_ids, i_ids, j_ids]
    print(i_ids)
    print(j_ids)
    ratio = 53
    scale = 448 / ratio
    scale0 = scale
    scale1 = scale
    mkpts0_c = torch.stack(
        [(i_ids % ratio), (i_ids // ratio)],
        dim=1) * scale0
    mkpts1_c = torch.stack(
        [(j_ids % ratio), (j_ids // ratio)],
        dim=1) * scale1

    mkpts0_c = mkpts0_c[mconf != 0].cpu().numpy()
    mkpts1_c = mkpts1_c[mconf != 0].cpu().numpy()

    mkpts0_c = mkpts0_c[6:]
    mkpts1_c = mkpts1_c[6:]

    img0 = plt.imread(fg_1_path)
    img1 = plt.imread(fg_2_path)
    mconf = mconf.tolist()

    color = cm.jet(mconf,alpha=0.7)
    color_rgb = (38, 138, 255)
    color_scaled = tuple(c/255 for c in color_rgb)

    make_matching_figure(
        img0, img1, mkpts0_c, mkpts1_c, color=color,
        kpts0=None, kpts1=None, text=[''], dpi=250, path='matched_images_combined.jpg')
    print(len(mkpts0_c))