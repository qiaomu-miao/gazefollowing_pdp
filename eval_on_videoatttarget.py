import os, sys
import torch
import torch.nn as nn
import pandas as pd
from pdp_model import ModelSpatialTemporal_PDP
from dataset import VideoAttTarget_video_new
from utils import imutils, evaluation, misc
from utils.log_utils import get_logger
from sklearn.metrics import average_precision_score
import matplotlib
from utils import myutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, cv2
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
import warnings
import argparse
import pdb
warnings.simplefilter(action='ignore', category=FutureWarning)


def custom_collate_fn(batch):
    num_element = len(batch[0])
    stacked = [torch.stack([each[i] for each in batch ])  for i in range(num_element-1)]
    last_element = tuple([each[num_element-1] for each in batch]) # paths
    stacked.append(last_element)
    return tuple(stacked)


parser = argparse.ArgumentParser()
parser.add_argument("--device", default='0')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--not_use_temporal_att', dest='use_temporal_att', action='store_false')
parser.add_argument('--not_use_depth', dest='use_depth', action='store_false')
parser.add_argument('--not_use_patch', dest='use_patch', action='store_false')
parser.add_argument('--config_file', default='./config/config_pdp.yaml')
parser.add_argument('--model_weights', type=str, default='/nfs/bigbrain.cs.stonybrook.edu/add_disk0/qiaomu/ckpts/gaze/videoatt/pdp_videoatt.pt')
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--vis', action='store_true', help='whether visualize result images')
args = parser.parse_args()


cfg = myutils.update_config(args.config_file)
for k in cfg.__dict__:
	setattr(args, k, cfg[k])
# Prepare data
print("Loading Data")
val_dataset = VideoAttTarget_video_new(args.DATA, test=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
											batch_size=args.batch_size,
											shuffle=False,
											num_workers=16,
											collate_fn=custom_collate_fn)


model = ModelSpatialTemporal_PDP(args, seq_len=args.DATA.seq_len)
model.cuda()
myutils.load_pretrained_weights(model, state_dict = None, weight_path=args.model_weights) 

setting_name = os.path.dirname(args.model_weights).split('/')[-1]
logdir = os.path.join('./logs', "eval", setting_name)
if not os.path.exists(logdir):
	os.makedirs(logdir)
log_path = os.path.join(logdir, "{}.log".format('eval'))
logger = get_logger(log_path)
logger.info(args)

eps = 1e-8
print('Evaluation in progress ...')
model.train(False)
AUC = []; in_vs_out_groundtruth = []; in_vs_out_pred = []; distance = []
all_img_names, all_patch_dist, all_hm_dist, all_gt_dist = [],[],[],[]
chunk_size = 5
frame_total = 0
input_resolution, output_resolution = args.DATA.input_resolution, args.DATA.output_resolution


with torch.no_grad():
	for batch_val,(img, face, head_channel, gaze_heatmap, gaze_coords, inout_label, head_coords, inout_logits_patch, depth_img,frame_idx, imsizes, paths) in enumerate(tqdm(val_loader)):
	
		bs, T, c, h, w = img.size()
		images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
		depth_img = depth_img.float().cuda()    
		inout_logits_patch = inout_logits_patch.float()
		inout_logits_patch, gaze_heatmap = inout_logits_patch.view(bs*T, -1), gaze_heatmap.view(bs*T, args.DATA.output_resolution, args.DATA.output_resolution)

		gaze_heatmap_pred, pred_inout_patches, inout_pred = model([images, head, faces, depth_img])
		frame_idx = torch.flatten(frame_idx).bool()
		gaze_heatmap_pred, pred_inout_patches, inout_pred = gaze_heatmap_pred[frame_idx], pred_inout_patches[frame_idx], inout_pred[frame_idx]
		head_coords, gaze_coords, inout_label = head_coords.view(-1, 4), gaze_coords.view(-1,2), inout_label.view(-1)
		inout_logits_patch, gaze_heatmap = inout_logits_patch[frame_idx], gaze_heatmap[frame_idx]
		gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
		head_coords, gaze_coords, inout_label, imsizes = head_coords[frame_idx], gaze_coords[frame_idx], inout_label[frame_idx], imsizes.view(-1,2)[frame_idx]
		paths_new = []
		for i in range(len(frame_idx)):
			row, col = i//T, i%T
			if frame_idx[i]==1:
				paths_new.append(paths[row][col])

		valid_idx = inout_label.cuda().bool()

		if args.use_patch:
			pred_inout_patches_dist = torch.sigmoid(pred_inout_patches) 
			pred_inout_patches_dist = pred_inout_patches_dist / (torch.sum(pred_inout_patches_dist, dim=1, keepdim=True) + eps)

		gaze_heatmap_pred  = gaze_heatmap_pred.cpu().numpy()
		gaze_coords = gaze_coords.cpu()
		# go through each data point and record AUC, min dist, avg dist
		myutils.eval_auc_sample(gaze_coords, inout_label, gaze_heatmap_pred, AUC, distance, output_resolution=args.DATA.output_resolution) # evaluate auc, dist for each sample in the batch
		in_vs_out_groundtruth.extend(inout_label.cpu().numpy())
		if args.use_patch:
			out_prob = pred_inout_patches_dist[:,-1].cpu()
			out_prob = (torch.ones(pred_inout_patches.size()[0])-out_prob).numpy()
		else:
			out_prob = torch.sigmoid(inout_pred).cpu().numpy()
		in_vs_out_pred.extend(out_prob)
		ap = evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)

		print("\t Batch {}/{}"
        "\tAUC:{:.4f}"
		"\tdist:{:.4f}"
		"\tin vs out AP:{:.4f}".
		format( batch_val, len(val_loader),
      			torch.mean(torch.tensor(AUC)),
				torch.mean(torch.tensor(distance)),
				ap
				))
  	
	ap = evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)

	logger.info("\tAUC:{:.4f}"
		"\tdist:{:.4f}"
		"\tin vs out AP:{:.4f}".
		format(torch.mean(torch.tensor(AUC)),
				torch.mean(torch.tensor(distance)),
				ap
				))

