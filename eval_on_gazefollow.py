import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.log_utils import get_logger
import torch
import torch.nn as nn
from pdp_model import ModelSpatial_PDP
from dataset import GazeFollow
from utils import imutils, evaluation, myutils
import numpy as np
from skimage.transform import resize
from utils.torchtools import save_checkpoint

def eval(args):
    eps = 1e-8
    print("Loading Data")
    # Set up log dir
    cfg = myutils.update_config(args.config_file)
    for k in cfg.__dict__:
        setattr(args, k, cfg[k])

    output_resolution = args.DATA.output_resolution 
    model = ModelSpatial_PDP(args)
    myutils.load_pretrained_weights(model, state_dict = None, weight_path=args.model_weights)  
     
    model.cuda()
    setting_name = os.path.dirname(args.model_weights).split('/')[-1]
    logdir = os.path.join('./logs', "eval", setting_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_path = os.path.join(logdir, "{}.log".format('eval'))
    logger = get_logger(log_path)
    logger.info(args)
    
    model.train(False)
    AUC = []; min_dist = []; avg_dist = []
    val_dataset = GazeFollow(args.DATA, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=12)

    with torch.no_grad():
        for val_batch, (img, face, head_channel, gaze_heatmap, cont_gaze, imsize, head_coords, gaze_coords, inout_logits_patch, depth_img, eye_pos, _) in enumerate(val_loader):
            images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
            inout_logits_patch = inout_logits_patch.float().cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            depth_img= depth_img.cuda()
            gaze_heatmap_pred, pred_inout_patches = model([images, head, faces, depth_img])
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
            bs = img.size()[0]
            inout_logits_patch = inout_logits_patch.view(bs, -1)
            
                # cross entropy loss for in vs out
            if args.use_patch:

                out_prob = torch.zeros(images.size()[0], 1).to(images) 
                inout_logits_patch = torch.cat((inout_logits_patch, out_prob), dim=-1)
                inout_logits_patch_dist = inout_logits_patch / inout_logits_patch.sum(dim=1, keepdim=True)
                
                pred_inout_patches_dist = torch.sigmoid(pred_inout_patches) 
                pred_inout_patches_dist = pred_inout_patches_dist / (torch.sum(pred_inout_patches_dist, dim=1, keepdim=True) + eps)
                pred_inout_patches_dist = pred_inout_patches_dist.cpu().numpy()        
            
            gaze_heatmap_pred = gaze_heatmap_pred.cpu().numpy()
            
            
            for b_i in range(len(cont_gaze)):
                # remove padding and recover valid ground truth points
                valid_gaze = cont_gaze[b_i]
                valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
                # AUC: area under curve of ROC
                multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                scaled_heatmap = resize(gaze_heatmap_pred[b_i], (imsize[b_i][1], imsize[b_i][0]))
                auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                AUC.append(auc_score)
                # min distance: minimum among all possible pairs of <ground truth point, predicted point>
                pred_x, pred_y = evaluation.argmax_pts(gaze_heatmap_pred[b_i])
                norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
                all_distances = []
                valid_gaze = valid_gaze.numpy()
                for gt_gaze in valid_gaze:
                    all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                min_dist.append(min(all_distances))
                # average distance: distance between the predicted point and human average point
                mean_gt_gaze = np.mean(valid_gaze, axis=0)
                avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
                avg_dist.append(avg_distance)
        
            
            
            print("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}\t".format(
            torch.mean(torch.tensor(AUC)),
            torch.mean(torch.tensor(min_dist)),
            torch.mean(torch.tensor(avg_dist))))
        
        logger.info("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}\t".format(
            torch.mean(torch.tensor(AUC)),
            torch.mean(torch.tensor(min_dist)),
            torch.mean(torch.tensor(avg_dist))))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0', help="gpu id")
    parser.add_argument('--config_file', default='./config/config_pdp.yaml')
    parser.add_argument('--not_use_depth', dest='use_depth', action='store_false')
    parser.add_argument('--model', default='baseline_single')
    parser.add_argument('--not_use_patch', dest='use_patch', action='store_false')
    parser.add_argument("--model_weights", type=str, default="/nfs/bigbrain.cs.stonybrook.edu/add_disk0/qiaomu/ckpts/gaze/pdp_spatial.pt", help="initial weights")
    parser.add_argument("--batch_size", type=int, default=80, help="batch size")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    eval(args)
