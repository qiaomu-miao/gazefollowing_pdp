import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pandas as pd
import pickle
import matplotlib as mpl

mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
import os
import pdb
from utils import imutils
from utils import myutils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



class GazeFollow(Dataset):
    # Use Gazefollow for pretraining
    # add patch logits and eye/head coordinates
    def __init__(self, args,
                 test=False, imshow=False):
        super(GazeFollow, self).__init__()
        self.data_dir = args.gazefollow_base_dir
        self.depth_dir = args.gazefollow_depth_dir
        
        if test:
            csv_path = os.path.join(self.data_dir, 'test_annotations_release.txt')
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta', 'ori_name']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")

            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max']].groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)
        else:
            csv_path = os.path.join(self.data_dir, 'train_annotations_release.txt')
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta', 'ori_name']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df = df[np.logical_and(np.less_equal(df['bbox_x_min'].values,df['bbox_x_max'].values), np.less_equal(df['bbox_y_min'].values, df['bbox_y_max'].values))]
            df.reset_index(inplace=True)
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'inout']]
            self.X_train = df['path']
            self.length = len(df)

        self.test = test
        self.patch_num= 7
        self.input_size = args.input_resolution
        self.output_size = args.output_resolution 
        self.transform = self._get_transform()

    def __getitem__(self, index):
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for i, row in g.iterrows():
                path = row['path']
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout = self.y_train.iloc[index]
            gaze_inside = bool(inout)

        
        x_min_ori, y_min_ori, x_max_ori, y_max_ori = x_min, y_min, x_max, y_max
        # expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        head_img_ori = img.crop((int(x_min_ori), int(y_min_ori), int(x_max_ori), int(y_max_ori))) # unjittered head image
        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        # read depth image
        depth_path = os.path.join(self.depth_dir, path)
        depth_path = depth_path[:-3]+'png'
        depth_img = cv2.imread(depth_path, -1)
        bits = 2
        max_val = (2**(8*bits))-1
        depth_img = depth_img / float(max_val)
        depth_img = Image.fromarray(depth_img)


        if self.test:
            imsize = torch.IntTensor([width, height])        
        else:
            ## data augmentation
            # Jitter (expansion-only) bounding box size
        
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.1
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)
                
            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])
                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)
                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)            
                depth_img = TF.crop(depth_img, crop_y_min, crop_x_min, crop_height, crop_width)

                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                head_img_ori = head_img_ori.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
            
            # Random color change
            if np.random.random_sample() <= 0.5:
                n1, n2, n3 = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), np.random.uniform(0, 1.5)
                img = TF.adjust_brightness(img, brightness_factor=n1)
                img = TF.adjust_contrast(img, contrast_factor=n2)
                img = TF.adjust_saturation(img, saturation_factor=n3)
                if 0 not in head_img_ori.size:
                    head_img_ori = TF.adjust_brightness(head_img_ori, brightness_factor=n1)
                    head_img_ori = TF.adjust_contrast(head_img_ori, contrast_factor=n2)
                    head_img_ori = TF.adjust_saturation(head_img_ori, saturation_factor=n3)
        
        
        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))


        head_coords = torch.tensor([x_min/float(width), y_min/float(height), x_max/float(width), y_max/float(height)])

        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)
        
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if self.test:  # aggregated heatmap
            num_valid = 0
            gaze_coords = []
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_coords.append(torch.tensor([gaze_x, gaze_y]))

            if num_valid>0:
                gaze_coords = torch.stack(gaze_coords).mean(dim=0)
                gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_coords[0].item() * self.output_size, gaze_coords[1].item() * self.output_size],
                                                         3,
                                                         type='Gaussian')
            else:
                gaze_coords = torch.tensor([-1.0, -1.0]) 
        else:
            if gaze_inside:
                gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                    3,
                                                    type='Gaussian')
                gaze_coords = torch.tensor([gaze_x, gaze_y])
            else:
                gaze_coords = torch.tensor([-1.0, -1.0])
        
        # add head heatmap
        eye_pos = torch.tensor([eye_x,eye_y]) # only for test
        
        inout_logits_patch = self.get_inout_patch_logits(gaze_heatmap)
        depth_img = depth_img.resize((self.input_size, self.input_size))
        depth_img = torch.tensor(np.array(depth_img)).unsqueeze(0)

        if self.test:
            return img, face, head_channel, gaze_heatmap, cont_gaze, imsize, head_coords, gaze_coords, inout_logits_patch, depth_img, eye_pos, path
        else:
            return img, face, head_channel, gaze_heatmap, gaze_inside, inout_logits_patch, depth_img

    def __len__(self):
        return self.length

    def get_inout_patch_logits(self, gaze_heatmap):
        patch_size = self.output_size // self.patch_num  # modify here
        steps = self.patch_num
        inout_patch = []
        for i in range(steps):
            for j in range(steps):
                inout_patch.append(gaze_heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].max())
        
        inout_patch = torch.tensor(inout_patch)
        return inout_patch
    
    def _get_transform(self):
        transform_list = []
        transform_list.append(transforms.Resize((self.input_size, self.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)


class VideoAttTarget_video_new(Dataset):
    # for transformer, output patchwise inout logits
    def __init__(self, args, test=False, imshow=False, seq_len=5):
        self.mode='train' if test==False else 'test'
        anno_path = os.path.join(args.vat_base_dir, 'annotations', f'anno_seq_{self.mode}.pickle')
        with open(anno_path, 'rb') as file:
            anno_pkl = pickle.load(file)
        
        self.all_sequences = anno_pkl
        self.data_dir = os.path.join(args.vat_base_dir, 'images')
        self.depth_dir = args.vat_depth_dir
        self.test = test
        self.patch_num= 7
        self.input_size = args.input_resolution
        self.output_size = args.output_resolution 
        self.imshow = imshow
        self.length = len(self.all_sequences)
        self.seq_len = seq_len
        self.transform = self._get_transform()

    def __getitem__(self, index):
        # frame idx: records whether this frame has been loaded before
        show_name, clip, frame_idx, df = self.all_sequences[index]
        this_length = len(df.index)

        # moving-avg smoothing
        window_size = 11 # should be odd number
        df['xmin'] = myutils.smooth_by_conv(window_size, df, 'xmin')
        df['ymin'] = myutils.smooth_by_conv(window_size, df, 'ymin')
        df['xmax'] = myutils.smooth_by_conv(window_size, df, 'xmax')
        df['ymax'] = myutils.smooth_by_conv(window_size, df, 'ymax')
        

        if not self.test:
            # cond for data augmentation
            cond_jitter = np.random.random_sample()
            cond_flip = np.random.random_sample()
            cond_color = np.random.random_sample()
            cond_crop = np.random.random_sample()
            if cond_color < 0.5:
                n1 = np.random.uniform(0.5, 1.5)
                n2 = np.random.uniform(0.5, 1.5)
                n3 = np.random.uniform(0.5, 1.5)

            if cond_crop < 0.5:
                sliced_x_min = df['xmin']
                sliced_x_max = df['xmax']
                sliced_y_min = df['ymin']
                sliced_y_max = df['ymax']
                sliced_gaze_x = df['gazex']
                sliced_gaze_y = df['gazey']

                check_sum = sliced_gaze_x.sum() + sliced_gaze_y.sum()
                all_outside = check_sum == -2*this_length

                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if all_outside:
                    crop_x_min = np.min([sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_y_min.max(), sliced_y_max.max()])
                else:
                    crop_x_min = np.min([sliced_gaze_x.min(), sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_gaze_y.min(), sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_gaze_x.max(), sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_gaze_y.max(), sliced_y_min.max(), sliced_y_max.max()])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Get image size
                path = os.path.join(self.data_dir, show_name, clip, df['path'].iloc[0])
                img = Image.open(path)
                img = img.convert('RGB')
                width, height = img.size

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)


        faces, images, head_channels, heatmaps, paths, gazes, imsizes, gaze_inouts = [], [], [], [], [], [], [], []
        head_coords, inout_logits_patch, depth_images, depth_faces = [],[],[],[]
        for i, row in df.iterrows():

            face_x1 = row['xmin']  # note: Already in image coordinates
            face_y1 = row['ymin']  # note: Already in image coordinates
            face_x2 = row['xmax']  # note: Already in image coordinates
            face_y2 = row['ymax']  # note: Already in image coordinates
            gaze_x = row['gazex']  # note: Already in image coordinates
            gaze_y = row['gazey']  # note: Already in image coordinates

            impath = os.path.join(self.data_dir, show_name, clip, row['path'])
            img = Image.open(impath)
            img = img.convert('RGB')
            head_img_ori = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2))).resize((224,224))
            paths.append(impath)
            depth_path = os.path.join(self.depth_dir, show_name, clip, row['path'])
            depth_path = depth_path[:-3]+'png'
            depth_img = cv2.imread(depth_path, -1)
            bits = 2
            max_val = (2**(8*bits))-1
            depth_img = depth_img / float(max_val)
            depth_img = Image.fromarray(depth_img)
            width, height = img.size
            face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
            gaze_x, gaze_y = map(float, [gaze_x, gaze_y])
            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
            else:
                if gaze_x < 0: # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = True

            if not self.test:
                ## data augmentation
                # Jitter (expansion-only) bounding box size.
                if cond_jitter < 0.5:
                    k = cond_jitter * 0.1
                    face_x1 -= k * abs(face_x2 - face_x1)
                    face_y1 -= k * abs(face_y2 - face_y1)
                    face_x2 += k * abs(face_x2 - face_x1)
                    face_y2 += k * abs(face_y2 - face_y1)
                    face_x1 = np.clip(face_x1, 0, width)
                    face_x2 = np.clip(face_x2, 0, width)
                    face_y1 = np.clip(face_y1, 0, height)
                    face_y2 = np.clip(face_y2, 0, height)

                # Random Crop
                if cond_crop < 0.5:
                    # Crop it
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                    depth_img = TF.crop(depth_img, crop_y_min, crop_x_min, crop_height, crop_width)
                    offset_x, offset_y = crop_x_min, crop_y_min

                    # convert coordinates into the cropped frame
                    face_x1, face_y1, face_x2, face_y2 = face_x1 - offset_x, face_y1 - offset_y, face_x2 - offset_x, face_y2 - offset_y
                    if gaze_inside:
                        gaze_x, gaze_y = (gaze_x- offset_x), \
                                         (gaze_y - offset_y)
                    else:
                        gaze_x = -1; gaze_y = -1

                    width, height = crop_width, crop_height

                # Flip?
                if cond_flip < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
                    x_max_2 = width - face_x1
                    x_min_2 = width - face_x2
                    face_x2 = x_max_2
                    face_x1 = x_min_2

                    head_img_ori = head_img_ori.transpose(Image.FLIP_LEFT_RIGHT)
                    if gaze_x != -1 and gaze_y != -1:
                        gaze_x = width - gaze_x

                # Random color change
                if cond_color < 0.5:
                    img = TF.adjust_brightness(img, brightness_factor=n1)
                    img = TF.adjust_contrast(img, contrast_factor=n2)
                    img = TF.adjust_saturation(img, saturation_factor=n3)
                    head_img_ori = TF.adjust_brightness(head_img_ori, brightness_factor=n1)
                    head_img_ori = TF.adjust_contrast(head_img_ori, contrast_factor=n2)
                    head_img_ori = TF.adjust_saturation(head_img_ori, saturation_factor=n3)
                

            # Face crop
            face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))
            head_coord_this = torch.tensor([face_x1/float(width), face_y1/float(height), face_x2/float(width), face_y2/float(height)])

            # Head channel image
            head_channel = imutils.get_head_box_channel(face_x1, face_y1, face_x2, face_y2, width, height,
                                                        resolution=self.input_size, coordconv=False).unsqueeze(0)
            if self.transform is not None:
                img = self.transform(img)
                face = self.transform(face)

            
            # Deconv output
            if gaze_inside:
                gaze_x /= float(width) # fractional gaze
                gaze_y /= float(height)
                gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                gaze_map = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
                gazes.append(torch.FloatTensor([gaze_x, gaze_y]))

            else:
                gaze_map = torch.zeros(self.output_size, self.output_size)
                gazes.append(torch.FloatTensor([-1, -1]))

    
            depth_img = depth_img.resize((self.input_size, self.input_size))

            inout_logits_this = self.get_inout_patch_logits(gaze_map)
            inout_logits_patch.append(inout_logits_this)
            faces.append(face)
            images.append(img)
            head_channels.append(head_channel)
            heatmaps.append(gaze_map)
            gaze_inouts.append(torch.FloatTensor([int(gaze_inside)]))
            head_coords.append(head_coord_this)
            imsizes.append(torch.tensor([width, height]))
            depth_images.append(torch.tensor(np.array(depth_img)).unsqueeze(0))
        
        faces = torch.stack(faces)
        images = torch.stack(images)
        head_channels = torch.stack(head_channels)
        heatmaps = torch.stack(heatmaps)
        gazes = torch.stack(gazes)
        gaze_inouts = torch.stack(gaze_inouts)
        head_coords = torch.stack(head_coords)
        inout_logits_patch = torch.stack(inout_logits_patch)        
        imsizes = torch.stack(imsizes)
        depth_images = torch.stack(depth_images)
    
    
        return images, faces, head_channels, heatmaps, gazes, gaze_inouts, head_coords, inout_logits_patch, depth_images, frame_idx, imsizes, paths
    
    def __len__(self):
        return self.length
    

    def get_inout_patch_logits(self, gaze_heatmap):
        patch_size = self.output_size // self.patch_num
        steps = self.patch_num
        inout_patch = []
        for i in range(steps):
            for j in range(steps):
                inout_patch.append(gaze_heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].max())
        
        inout_patch = torch.tensor(inout_patch)
        return inout_patch
    
    def _get_transform(self):
        transform_list = []
        transform_list.append(transforms.Resize((self.input_size, self.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)


