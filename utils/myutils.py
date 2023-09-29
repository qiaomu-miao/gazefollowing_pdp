import matplotlib as mpl
mpl.use('Agg')
import cv2
from matplotlib import pyplot as plt
from easydict import EasyDict as edict
import yaml
import numpy as np
import pandas as pd
import logging
import torch
import os, pickle
import warnings
import shutil
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter
import matplotlib.patches as patches
from utils import imutils, evaluation, misc
from collections import OrderedDict
from skimage.transform import resize

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

def get_logger(log_file):
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s- %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def setup_logger_tensorboard(project_name, setting_name, ckpt_base_dir, resume=-1):
    logdir = os.path.join('./logs', project_name, setting_name)
    ckpt_dir = os.path.join(ckpt_base_dir, project_name, setting_name)
    if resume==-1 and os.path.exists(logdir):
        shutil.rmtree(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir)
    log_path = os.path.join(logdir, "{}.log".format('train'))
    logger = get_logger(log_path)
    return logdir, ckpt_dir, logger, writer


def smooth_by_conv(window_size, df, col):
    padded_track = pd.concat([pd.DataFrame([[df.iloc[0][col]]]*(window_size//2), columns=[0]),
                     df[col],
                     pd.DataFrame([[df.iloc[-1][col]]]*(window_size//2), columns=[0])])
    smoothed_signals = np.convolve(padded_track.squeeze(), np.ones(window_size)/window_size, mode='valid')
    return smoothed_signals

def eval_auc_sample(gaze_coords, inout_label, gaze_heatmap_pred, AUC, distance, output_resolution=64):
    for b_i in range(len(inout_label)):
        # remove padding and recover valid ground truth points
        valid_gaze = gaze_coords[b_i]
        valid_gaze = valid_gaze[valid_gaze != -1].reshape(-1,2)
        # AUC: area under curve of ROC
        if inout_label[b_i]:
                # AUC: area under curve of ROC
            multi_hot = torch.zeros(output_resolution, output_resolution)  # set the size of the output
            gaze_x = gaze_coords[b_i, 0]
            gaze_y = gaze_coords[b_i, 1]

            multi_hot = imutils.draw_labelmap(multi_hot, [gaze_x * output_resolution, gaze_y * output_resolution], 3, type='Gaussian')
            multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
            multi_hot = misc.to_numpy(multi_hot)

            scaled_heatmap = resize(gaze_heatmap_pred[b_i].squeeze(), (output_resolution, output_resolution))
            auc_score = evaluation.auc(scaled_heatmap, multi_hot)
            AUC.append(auc_score)

            # distance: L2 distance between ground truth and argmax point
            pred_x, pred_y = evaluation.argmax_pts(gaze_heatmap_pred[b_i])
 
            norm_p = [pred_x/output_resolution, pred_y/output_resolution]
            dist_score = evaluation.L2_dist(gaze_coords[b_i], norm_p).item()
            distance.append(dist_score)


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.
    
    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, state_dict = None, weight_path=None):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    if state_dict is None:
        checkpoint = load_checkpoint(weight_path)
        state_dict = checkpoint

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
        
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

def load_submodule_weights(model, module_name, state_dict=None, weight_path=None):
    # module name: the name of the submodule in the main model
    
    if state_dict is None:
        checkpoint = load_checkpoint(weight_path)
        state_dict = checkpoint
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k.startswith(module_name):
            k = k[len(module_name)+1:]
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)
    
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}" for {}'.
            format(weight_path, module_name)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )     

def resume_from_epoch(args, model, scheduler, ckpt_dir):
    
    ckpt_load = os.path.join(ckpt_dir, 'epoch_%02d_weights.pt' % (args.resume))
    torch_load = torch.load(ckpt_load)
    state_dict = torch_load['state_dict']
    load_pretrained_weights(model, state_dict = state_dict, weight_path=ckpt_load) 
    print("successfully load {}!".format(ckpt_load))
    step = torch_load['train_step']
    if not args.no_decay: #and not args.coord_regression:
        for i in range(args.resume+1):
            scheduler.step()
    return step

