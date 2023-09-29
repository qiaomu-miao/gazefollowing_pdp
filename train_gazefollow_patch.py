import torch
import argparse
import os
from utils.log_utils import get_logger
import pdb
import torch.nn as nn
from pdp_model import ModelSpatial_PDP
from dataset import GazeFollow
from utils import imutils, evaluation, myutils
import random
from datetime import datetime
import shutil
import numpy as np
import torch.nn.functional as F
from skimage.transform import resize
from utils.torchtools import save_checkpoint
from tensorboardX import SummaryWriter
import warnings
from losses import KL_div_modified

warnings.simplefilter(action='ignore', category=FutureWarning)

eps = 1e-8



def train(args):

    # Prepare data
    print("Loading Data")
    # Set up log dir
    cfg = myutils.update_config(args.config_file)
    for k in cfg.__dict__:
        setattr(args, k, cfg[k])
    
    torch.manual_seed(1)
    np.random.seed(1)
    
    
    input_resolution, output_resolution = args.DATA.input_resolution, args.DATA.output_resolution
    train_dataset = GazeFollow(args.DATA, test=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16)
    
    val_dataset = GazeFollow(args.DATA, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size*2,
                                               shuffle=False,
                                               num_workers=16)
    if args.resume==-1:
        setting_name = f"{args.remark}model{args.model}_depth_{args.use_depth}lr{args.lr}bs{args.batch_size}_ampfactor{args.loss_amp_factor}_lambda{args.lambda_}" 
    else:
        setting_name = args.setting_name
    
    logdir = os.path.join('./logs', args.project_name, setting_name)
    ckpt_dir = os.path.join(args.ckpt_dir, args.project_name, setting_name)
    if args.resume==-1 and os.path.exists(logdir):
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    writer = SummaryWriter(logdir)
        
    log_path = os.path.join(logdir, "{}.log".format('train'))
    logger = get_logger(log_path)
    logger.info(args)

    print("Constructing model")
    
    model = ModelSpatial_PDP(args)
    model.cuda()
    
    if len(args.init_weights)>0 and args.init_weights.lower()!='imagenet':
        myutils.load_pretrained_weights(model, state_dict = None, weight_path=args.init_weights) 
    
    # Loss functions
    mseloss_mean = nn.MSELoss(reduction='mean')
    kl_div_loss = KL_div_modified(reduction='batchmean') # modify the kl divergence loss so it can deal with 0 probabilities

    # Optimizer
    att_param_names = ['qkv_proj.weight', 'qkv_proj.bias']
    att_params = list(filter(lambda kv: kv[0] in att_param_names , model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in att_param_names , model.named_parameters())) 
    
    optimizer = torch.optim.Adam([
        {'params': [temp[1] for temp in base_params]},
        {'params': [temp[1] for temp in att_params], 'lr': args.lr/2}
                ], lr=args.lr)
    sche = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 22, 30], gamma=0.2)
    step = 0
    calc_lr = 0
    if args.resume!=-1:
        ckpt_load = os.path.join(args.ckpt_dir, args.project_name, setting_name, 'epoch_%02d_weights.pt' % (args.resume))
        torch_load = torch.load(ckpt_load)
        state_dict = torch_load['state_dict']
        myutils.load_pretrained_weights(model, state_dict = state_dict, weight_path=ckpt_load) 
        print("successfully load {}!".format(ckpt_load))
        step = torch_load['train_step']
        for i in range(args.resume+1):
            sche.step()
    
    multigpu = False
    if len(args.device)>1:
        multigpu=True
        model = nn.DataParallel(model)
    
    loss_amp_factor = args.loss_amp_factor # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()
    print("Training in progress ...")
    for ep in range(max(args.resume+1,0), args.epochs):
        logger.info(f"Epoch {ep}: learning rate: {optimizer.param_groups[0]['lr']}, calc_lr: {calc_lr}") 
        model.train(True) 
        for batch, (img, face, head_channel, gaze_heatmap, gaze_inside, inout_logits_patch, depth_img) in enumerate(train_loader):
            
            images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
            inout_logits_patch = inout_logits_patch.float().cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            depth_img = depth_img.cuda()
            gaze_inside = gaze_inside.cuda().to(torch.float)
            gaze_heatmap_pred, pred_inout_patches  = model([images, head, faces, depth_img])
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
            bs = img.size()[0]
            inout_logits_patch = inout_logits_patch.view(bs, -1)

            # l2 loss computed only for inside case?
 
            l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)

            l2_loss = l2_loss * loss_amp_factor            
            
            if args.use_patch:
                out_prob = (torch.ones(gaze_inside.size()).to(gaze_inside) - gaze_inside).unsqueeze(1)
                inout_logits_patch = torch.cat((inout_logits_patch, out_prob), dim=-1)
                inout_logits_patch_dist = inout_logits_patch / inout_logits_patch.sum(dim=1, keepdim=True)
                pred_inout_patches_dist = torch.sigmoid(pred_inout_patches) 
                pred_inout_patches_dist = pred_inout_patches_dist / (torch.sum(pred_inout_patches_dist, dim=1, keepdim=True) + eps)
                loss_pdp = kl_div_loss(pred_inout_patches_dist, inout_logits_patch_dist)

                Xent_loss = loss_pdp  * args.lambda_
            else:
                Xent_loss = torch.tensor(0.0).cuda()
            total_loss = l2_loss + Xent_loss
            

            writer.add_scalar('train/Loss_step', total_loss.item(), step)
            writer.add_scalar('train/l2_loss_step', l2_loss.item(), step)
            writer.add_scalar('train/PDP_loss_step', Xent_loss.item(), step)

            total_loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if batch % args.print_every == 0:
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (PDP){:.4f}".format(ep, batch+1, max_steps, l2_loss, Xent_loss))
        
        
        logger.info('Validation in progress ...: epoch {}'.format(ep))
        model.train(False)
        AUC = []; min_dist = []; avg_dist = []
        ep_val_loss, ep_val_l2_loss, ep_val_inout_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for val_batch, (img, face, head_channel, gaze_heatmap, cont_gaze, imsize, head_coords, gaze_coords, inout_logits_patch, depth_img, eye_pos, _) in enumerate(val_loader):
                images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
                head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
                inout_logits_patch = inout_logits_patch.float().cuda()
                gaze_heatmap = gaze_heatmap.cuda()
                depth_img= depth_img.cuda()
                gaze_heatmap_pred, pred_inout_patches = model([images, head, faces, depth_img])
                gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
                bs = img.size()[0]
                inout_logits_patch = inout_logits_patch.view(bs, -1)
                
                # Loss
                    # l2 loss computed only for inside case
                l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)*loss_amp_factor
                gaze_inside = torch.ones(images.size()[0]).cuda() 

                    # cross entropy loss for in vs out
                if args.use_patch:
                    out_prob = torch.zeros(images.size()[0], 1).to(gaze_inside) 
                    inout_logits_patch = torch.cat((inout_logits_patch, out_prob), dim=-1)
                    inout_logits_patch_dist = inout_logits_patch / inout_logits_patch.sum(dim=1, keepdim=True)
                    pred_inout_patches_dist = torch.sigmoid(pred_inout_patches) 
                    pred_inout_patches_dist = pred_inout_patches_dist / (torch.sum(pred_inout_patches_dist, dim=1, keepdim=True) + eps)
                    
                    loss_pdp = kl_div_loss(pred_inout_patches_dist, inout_logits_patch_dist)
                else:
                    pred_inout_patches_dist = torch.zeros(bs, 50).cuda()
                    if args.no_inout:
                        loss_pdp = torch.tensor(0.0).cuda()
                    
                Xent_loss = loss_pdp  * args.lambda_
                total_loss = l2_loss + Xent_loss

                ep_val_l2_loss += l2_loss.item()
                ep_val_inout_loss += Xent_loss.item()
                gaze_heatmap_pred = gaze_heatmap_pred.cpu().numpy()
                pred_inout_patches_dist = pred_inout_patches_dist.cpu().numpy()
                
                # go through each data point and record AUC, min dist, avg dist
                for b_i in range(len(cont_gaze)):
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
            
            logger.info("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}\t".format(
                torch.mean(torch.tensor(AUC)),
                torch.mean(torch.tensor(min_dist)),
                torch.mean(torch.tensor(avg_dist))))


        # Tensorboard
        writer.add_scalar('Validation/AUC', torch.mean(torch.tensor(AUC)), global_step=ep)
        writer.add_scalar('Validation/min dist', torch.mean(torch.tensor(min_dist)), global_step=ep)
        writer.add_scalar('Validation/avg dist', torch.mean(torch.tensor(avg_dist)), global_step=ep)
        ep_val_loss, ep_val_l2_loss, ep_val_inout_loss  = ep_val_loss/(val_batch+1), ep_val_l2_loss/(val_batch+1), ep_val_inout_loss/(val_batch+1)
        writer.add_scalar('val/Loss', ep_val_loss, ep)
        writer.add_scalar('val/l2_loss', ep_val_l2_loss, ep)
        writer.add_scalar('val/Inout_loss', ep_val_inout_loss, ep)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], ep)

        
        if ep % args.save_every == 0:
            # save the model
            if multigpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            checkpoint = {'state_dict': state_dict, 'lr':optimizer.param_groups[0]['lr'], 'train_step':step}
            save_checkpoint(checkpoint, ckpt_dir, 'epoch_%02d_weights.pt' % (ep), remove_module_from_keys=True)

        sche.step() 
        writer.flush()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0', help="gpu id")
    parser.add_argument('--project_name', default='test_final')
    parser.add_argument('--setting_name', default='')
    parser.add_argument('--config_file', default='./config/config_pdp.yaml')
    parser.add_argument('--no_inout', action='store_true')
    parser.add_argument('--not_use_depth', dest='use_depth', action='store_false')
    parser.add_argument('--model', default='pdp')
    parser.add_argument('--not_use_patch', dest='use_patch', action='store_false')
    parser.add_argument('--loss_amp_factor', type=float, default=10000)
    parser.add_argument('--lambda_', type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=80, help="batch size")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
    parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
    parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
    parser.add_argument("--resume", type=int, default=-1, help="which epoch to resume from")
    parser.add_argument('--remark', type=str, default='')
    parser.add_argument("--init_weights", type=str, default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/videoatttarget_init/initial_weights_for_spatial_training.pt", help="initial weights")
    parser.add_argument("--ckpt_dir", type=str, default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gazefollow_pdp", help="directory to save log files")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    train(args)
