import torch
import torch.nn as nn
import torch.nn.functional as F

from pdp_model import ModelSpatialTemporal_PDP 
from dataset import VideoAttTarget_video_new
import argparse
import os, shutil
from datetime import datetime
from utils import imutils, myutils, evaluation, misc
from matplotlib import pyplot as plt
import numpy as np
import warnings, pdb
import logging
from skimage.transform import resize
from losses import KL_div_modified
from utils.torchtools import save_checkpoint
from sklearn.metrics import average_precision_score
from tensorboardX import SummaryWriter
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_logger(log_file):
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s- %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')[:-7]


def get_optimize_list(args, model):
    optimize_list=[]

    # Optimizer
    if 'baseline_single' in args.model:
        optimize_list += [
            {'params': model.compress_conv1.parameters(), 'lr': args.lr}, # added for scene encoding
            {'params': model.compress_conv2.parameters(), 'lr': args.lr},
            {'params': model.deconv1.parameters(), 'lr': args.lr},
            {'params': model.deconv2.parameters(), 'lr': args.lr},
            {'params': model.deconv3.parameters(), 'lr': args.lr},
            {'params': model.conv4.parameters(), 'lr': args.lr},
            
            {'params': model.qkv_proj.parameters(), 'lr': args.lr},
            {'params': model.outside_embedding, 'lr':args.lr},
            {'params': model.patch_pos_embedding, 'lr':args.lr},
            
            {'params': model.inout_patch_encode.parameters(), 'lr': args.lr},
            {'params': model.inout_patch_pred.parameters(), 'lr': args.lr},
            {'params': model.deconv_encode.parameters(), 'lr': args.lr}]

    if args.use_temporal_att:
        temporal_att_params = [p for n,p in model.temporal_attention.named_parameters() if 'out_proj' not in n]
        optimize_list+=[
            {'params': model.temporal_mapping.parameters(), 'lr': args.lr},
            {'params': model.temporal_pos_embed,'lr': args.lr},
            {'params': model.temporal_proj_1.parameters(),'lr': args.lr},
            {'params': model.temporal_norm_1.parameters(),'lr': args.lr},
            {'params': temporal_att_params, 'lr': args.lr},
        ]
    return optimize_list

eps = 1e-8
def train(args):
    cfg = myutils.update_config(args.config_file)
    for k in cfg.__dict__:
        setattr(args, k, cfg[k])
    # Prepare data
    print("Loading Data")
    train_dataset = VideoAttTarget_video_new(args.DATA, test=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16,
                                               collate_fn = custom_collate_fn)
    
    val_dataset = VideoAttTarget_video_new(args.DATA, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=16,
                                             collate_fn=custom_collate_fn)

    if args.resume==-1:
        setting_name = f"{args.remark}lr{args.lr}bs{args.batch_size}_ampfactor{args.loss_amp_factor}_lambda{args.lambda_}" 
        if args.debug:
            setting_name += '_debug'
    else:
        setting_name = args.setting_name
    
    logdir = os.path.join('./logs', args.project_name, setting_name)
    ckpt_dir = os.path.join(args.ckpt_dir, args.project_name, setting_name)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(ckpt_dir) and not args.debug:
        os.makedirs(ckpt_dir)

    writer = SummaryWriter(logdir)
    np.random.seed(1)
    log_name = get_date_str()
    log_path = os.path.join(logdir, "{}.log".format(log_name))
    logger = get_logger(log_path)
    logger.info(args)

    # Define device
    print("Constructing model")
    if args.model=='baseline_single':
        model = ModelSpatialTemporal_PDP(args, seq_len=args.DATA.seq_len)
    model.cuda()

    if args.init_weights:
        print("Loading weights")
        snapshot = torch.load(args.init_weights)
        myutils.load_pretrained_weights(model, weight_path=args.init_weights) 

    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    mseloss_mean = nn.MSELoss(reduction='mean')
    bcelogit_loss = nn.BCEWithLogitsLoss()
    kl_div_loss = KL_div_modified(reduction='batchmean') # modify the kl divergence loss so it can deal with 0 probabilities

    optimize_list = get_optimize_list(args, model)
    optimizer = torch.optim.Adam(optimize_list, lr=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.5)
    input_resolution, output_resolution = args.DATA.input_resolution, args.DATA.output_resolution

    if args.resume!=-1:
        weight_path = os.path.join(args.ckpt_dir, args.project_name, args.setting_name, 'epoch_%02d_weights.pt' % (args.resume))
        snapshot = torch.load(weight_path)
        myutils.load_pretrained_weights(model, state_dict = snapshot, weight_path=weight_path)

    
    if len(args.device)>1:
        model = nn.DataParallel(model)
    max_steps = len(train_loader)
    print("Training in progress ...")
    step = 0
    
    for ep in range(args.resume+1, args.epochs):
        
        ep_loss, ep_l2_loss, ep_inout_loss = 0.0, 0.0, 0.0
        # load weight
        model.train(True)
        # freeze batchnorm layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()
        
        logger.info(f"Epoch {ep}: learning rate: {optimizer.param_groups[0]['lr']}")  
        for batch, (img, face, head_channel, gaze_heatmap, gaze_coords, inout_label, head_coords, inout_logits_patch, depth_img, frame_idx, imsizes, img_paths) in enumerate(train_loader):
            # img: BxTxCxHxW

            bs, T, c, h, w = img.size()
            images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
            head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
            depth_img = depth_img.float().cuda()
            inout_logits_patch, gaze_heatmap = inout_logits_patch.float().cuda(), gaze_heatmap.cuda()
            inout_logits_patch, gaze_heatmap = inout_logits_patch.view(bs*T, -1), gaze_heatmap.view(bs*T, output_resolution, output_resolution)
            gaze_heatmap_pred, pred_inout_patches, inout_pred = model([images, head, faces, depth_img])
            frame_idx = torch.flatten(frame_idx).bool().cuda()
            gaze_heatmap_pred, pred_inout_patches, inout_pred = gaze_heatmap_pred[frame_idx], pred_inout_patches[frame_idx], inout_pred[frame_idx]
            head_coords, gaze_coords, inout_label = head_coords.view(-1, 4), gaze_coords.view(-1,2), inout_label.view(-1)
            inout_logits_patch, gaze_heatmap = inout_logits_patch[frame_idx], gaze_heatmap[frame_idx]
            gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
            head_coords, gaze_coords, inout_label = head_coords[frame_idx], gaze_coords[frame_idx], inout_label[frame_idx]
            valid_idx = inout_label.cuda().bool()
            bs = img.size()[0]

            l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)            
            l2_loss = l2_loss * args.loss_amp_factor
            inout_label = inout_label.cuda().to(torch.float)
            
                # cross entropy loss for in vs out
            if args.use_patch:
                out_prob = (torch.ones(inout_label.size()).to(inout_label) - inout_label).unsqueeze(1)
                inout_logits_patch = torch.cat((inout_logits_patch, out_prob), dim=-1)
                inout_logits_patch_dist = inout_logits_patch / inout_logits_patch.sum(dim=1, keepdim=True)
                
                pred_inout_patches_dist = torch.sigmoid(pred_inout_patches) 
                pred_inout_patches_dist = pred_inout_patches_dist / (torch.sum(pred_inout_patches_dist, dim=1, keepdim=True) + eps)
                    
                loss_inout_patch = kl_div_loss(pred_inout_patches_dist, inout_logits_patch_dist) 
                
            else:
                loss_inout_patch = bcelogit_loss(inout_pred.squeeze(), inout_label.squeeze())
            
            Xent_loss = loss_inout_patch  * args.lambda_
            total_loss = l2_loss + Xent_loss
            
            if torch.isnan(loss_inout_patch):
                pdb.set_trace()
            ep_loss += total_loss
            ep_l2_loss += l2_loss
            ep_inout_loss += Xent_loss
            optimizer.zero_grad()
            total_loss.backward() 

            optimizer.step()
            step+=1
    
            if batch % args.print_every == 0:
                print("Epoch:{:04d}\tstep:{:04d}/{:04d}\ttraining loss: (l2){:.4f} (Xent){:.4f}".format(ep, batch+1, max_steps, l2_loss, Xent_loss))
        ep_loss /= batch+1
        ep_l2_loss /= batch+1
        ep_inout_loss /= batch+1
        writer.add_scalar('train/Loss', ep_loss.item(), ep)
        writer.add_scalar('train/l2_loss', ep_l2_loss.item(), ep)
        writer.add_scalar('train/Inout_loss', ep_inout_loss.item(), ep)

            # save the model
        if ep % args.save_every == 0 and not args.debug:
            # save the model
            if len(args.device) > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            checkpoint = {'state_dict': state_dict}
            save_checkpoint(checkpoint, ckpt_dir, 'epoch_%02d_weights.pt' % (ep), remove_module_from_keys=True)
        
        logger.info('Validation in progress ...: epoch {}'.format(ep))
        model.train(False)
        AUC = []; in_vs_out_groundtruth = []; in_vs_out_pred = []; distance = []
        with torch.no_grad():
            ep_loss, ep_l2_loss, ep_inout_loss = 0.0, 0.0, 0.0
            for val_batch, (img, face, head_channel, gaze_heatmap, gaze_coords, inout_label, head_coords, inout_logits_patch, depth_img,frame_idx, imsizes, paths) in enumerate(val_loader):
                bs, T, c, h, w = img.size()
                images, head, faces = img.cuda(), head_channel.cuda(), face.cuda()
                head_coords, gaze_coords = head_coords.float().cuda(), gaze_coords.float().cuda()
                depth_img = depth_img.float().cuda()    
                inout_logits_patch, gaze_heatmap = inout_logits_patch.float().cuda(), gaze_heatmap.cuda()
                inout_logits_patch, gaze_heatmap = inout_logits_patch.view(bs*T, -1), gaze_heatmap.view(bs*T, args.DATA.output_resolution, args.DATA.output_resolution)
                gaze_heatmap_pred, pred_inout_patches, inout_pred = model([images, head, faces, depth_img])
                frame_idx = torch.flatten(frame_idx).bool()
                gaze_heatmap_pred, pred_inout_patches, inout_pred = gaze_heatmap_pred[frame_idx], pred_inout_patches[frame_idx], inout_pred[frame_idx]
                head_coords, gaze_coords, inout_label = head_coords.view(-1, 4), gaze_coords.view(-1,2), inout_label.view(-1)
                inout_logits_patch, gaze_heatmap = inout_logits_patch[frame_idx], gaze_heatmap[frame_idx]
                gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)
                head_coords, gaze_coords, inout_label = head_coords[frame_idx], gaze_coords[frame_idx], inout_label[frame_idx]
                valid_idx = inout_label.cuda().bool()
                
                l2_loss = mseloss_mean(gaze_heatmap_pred, gaze_heatmap)
                
                l2_loss = l2_loss * args.loss_amp_factor

                inout_label = inout_label.cuda().to(torch.float)
                
            
                # cross entropy loss for in vs out
                if args.use_patch:
                    
                    out_prob = (torch.ones(inout_label.size()).to(inout_label) - inout_label).unsqueeze(1)
                    inout_logits_patch = torch.cat((inout_logits_patch, out_prob), dim=-1)
                    inout_logits_patch_dist = inout_logits_patch / inout_logits_patch.sum(dim=1, keepdim=True)
                    
                    pred_inout_patches_dist = torch.sigmoid(pred_inout_patches) 
                    pred_inout_patches_dist = pred_inout_patches_dist / (torch.sum(pred_inout_patches_dist, dim=1, keepdim=True) + eps)
                    loss_inout_patch = kl_div_loss(pred_inout_patches_dist, inout_logits_patch_dist)
                 
                else:
                    loss_inout_patch = bcelogit_loss(inout_pred.squeeze(), inout_label.squeeze())

                Xent_loss = loss_inout_patch  * args.lambda_
                
                total_loss = l2_loss + Xent_loss
                ep_loss += total_loss
                ep_l2_loss += l2_loss
                ep_inout_loss += Xent_loss

                gaze_heatmap_pred  = gaze_heatmap_pred.cpu().numpy()
                gaze_coords = gaze_coords.cpu()
                # go through each data point and record AUC, min dist, avg dist
                myutils.eval_auc_sample(gaze_coords, inout_label, gaze_heatmap_pred, AUC, distance, output_resolution=args.DATA.output_resolution) # evaluate auc, dist for each sample in the batch
                in_vs_out_groundtruth.extend(inout_label.cpu().numpy())
                if args.use_patch:
                    #out_prob = F.softmax(pred_inout_patches, dim=-1)[:,-1].cpu()
                    out_prob = pred_inout_patches_dist[:,-1].cpu()
                    out_prob = (torch.ones(pred_inout_patches.size()[0])-out_prob).numpy()
                else:
                    out_prob = torch.sigmoid(inout_pred).cpu().numpy()
                in_vs_out_pred.extend(out_prob)
            

            ap = evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)
            try:
                logger.info("\tAUC:{:.4f}"
                    "\tdist:{:.4f}"
                    "\tin vs out AP:{:.4f}".
                    format(torch.mean(torch.tensor(AUC)),
                            torch.mean(torch.tensor(distance)),
                            ap
                            ))
            except:
                pass   
            # Tensorboard
            writer.add_scalar('Val/AUC', torch.mean(torch.tensor(AUC)), global_step=ep)
            writer.add_scalar('Val/dist', torch.mean(torch.tensor(distance)), global_step=ep)
            writer.add_scalar('Val/ap', ap, global_step=ep)
            ep_loss /= val_batch+1
            ep_l2_loss /= val_batch+1
            ep_inout_loss /= val_batch+1
    
            writer.add_scalar('Val/Loss', ep_loss.item(), ep)
            writer.add_scalar('Val/l2_loss', ep_l2_loss.item(), ep)
            writer.add_scalar('Val/Inout_loss', ep_inout_loss.item(), ep)
            logger.info("Epoch {} test: loss: {} L2 loss: {}, inout loss: {}".format(ep, ep_loss, ep_l2_loss, ep_inout_loss))
        scheduler.step()
        writer.flush()
    writer.close()


def custom_collate_fn(batch):
    num_element = len(batch[0])
    stacked = [torch.stack([each[i] for each in batch ])  for i in range(num_element-1)]
    last_element = tuple([each[num_element-1] for each in batch]) # paths
    stacked.append(last_element)
    return tuple(stacked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='0', help="gpu id")
    parser.add_argument('--project_name', default='train_videoatt_pdp')
    parser.add_argument('--setting_name', default='')
    parser.add_argument('--model', default='baseline_single')
    parser.add_argument('--not_use_temporal_att', dest='use_temporal_att', action='store_false')
    parser.add_argument('--config_file', default='./config/config_pdp.yaml')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--not_use_depth', dest='use_depth', action='store_false')
    parser.add_argument('--not_use_patch', dest='use_patch', action='store_false')
    parser.add_argument('--optim', default='adam')
    parser.add_argument("--init_weights", type=str, default="/nfs/bigbrain/add_disk0/qiaomu/ckpts/gaze/videoatttarget/initial_weights_for_temporal.pt", help="initial weights")
    parser.add_argument("--ckpt_dir", type=str, default="/nfs/bigrod/add_disk0/qiaomu/ckpts/videoatttarget/pdp", help="directory to save log files")
    parser.add_argument('--loss_amp_factor', type=float, default=5000)
    parser.add_argument('--lambda_', type=float, default=40.0)
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
    parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
    parser.add_argument("--resume", type=int, default=-1, help="which epoch to resume from")
    parser.add_argument('--remark', type=str, default='')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    train(args)

