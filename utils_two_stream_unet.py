import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import ipdb
import cv2 
from torchvision import transforms

torch.manual_seed(10)

convert_tensor = transforms.ToTensor()
augmentation_transform = None # add augmentation 

PARENT_FOLDER= 'data'
LIST_OF_DATASETS = ['task_positives_11-2022_04_12_18_28_14-segmentation mask 1.1']
IMAGE_LOC  = 'JPEGImages'
MASK_LOC = 'SegmentationClass'


def get_image(i):
    image_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
    return Image.open(image_path)

def get_mask(i):
    mask_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
    return Image.open(mask_path)


def find_needle_mask(mask_arr):
    mask_needle = 255*(np.array([255,255,255]) - np.where(mask_arr == 51,221,255))
    mask_needle = np.concatenate([np.expand_dims(mask_needle[:,:,0], axis=-1)]*3, axis=-1)
    mask_needle = Image.fromarray(mask_needle.astype('uint8'))        

    return mask_needle

def get_data(n_flow = 3):
    
    i = 1
    image_list, flow_list, needle_masks_list, mask_list, flow_concat_list = [], [], [], [], []
    image_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
    # mask_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
    
    # as flow for 1st frame is zero | first frame outisde while 
    img = get_image(i)
    img_tensor = convert_tensor(img)
    image_list.append(img_tensor)
    hsv = np.zeros_like(img)
    cartesian = np.zeros_like(img)
    hsv[..., 1] = 255 # scale should be 1, then conversion from hsv to rgb becomes easy

    # initial flow for frame 1 
    flow_ = torch.zeros_like(img_tensor)[0:1,:,:]
    flow_ = torch.cat([flow_]*3, dim = 0)
    flow_concat_list.append(flow_)

    mask = get_mask(i) #Image.open(mask_path)
    mask_arr = np.array(mask)
    mask_tensor = convert_tensor(mask)
    mask_list.append(mask_tensor)

    mask_needle_ = find_needle_mask(mask_arr)
    needle_masks_list.append(convert_tensor(mask_needle_)[0:1,:,:])    

    img_H = img_tensor.shape[1]
    img_W = img_tensor.shape[2]

    while os.path.exists(image_path):
        
        # next image frame
        print("i = " , i)
        i = i + 1
        image_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
        img_next = get_image(i) #Image.open(image_path)
        img_tensor = convert_tensor(img_next)
        image_list.append(img_tensor)

        # read segmentation masks
        # mask_path = os.path.join(MASK_LOC, 'frame_' + '000{}'.format(i+1).zfill(6) + '.png')
        mask = get_mask(i) #Image.open(mask_path)
        # get just needle masks
        mask_arr = np.array(mask)
        mask_tensor = convert_tensor(mask)
        mask_list.append(mask_tensor)
        # finding just needle mask
        mask_needle_ = find_needle_mask(mask_arr)
        needle_masks_list.append(convert_tensor(mask_needle_)[0:1,:,:])
        
        # compute optical flow and stack in groups of n
        prvs = np.array(img)[:,:,0]
        next = np.array(img_next)[:,:,0]

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)    
        mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])])
        cartesian[..., [0,2]] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
        cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
        
        # cv2.imwrite('flow_cartesian/flow_cartesian_{}.png'.format(i), cartesian)
        flow_list.append(convert_tensor(cartesian_bgr)[0:1,:,:])

        # # EXTRAS
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang*180/np.pi/2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # save combined needle_mask and optical flow
        # flow_mask = cv2.vconcat([mask_needle.astype('uint8'), cartesian])
        # cv2.imwrite('flow_and_mask/{}.png'.format(i), flow_mask)

        # update previous image
        img = img_next

        # ipdb.set_trace()
        # concatenate m flow frames along channel dim
        if len(flow_list) >= 3:
            tmp = torch.cat(flow_list[-3:], dim=0)              
        else:
            tt1 = torch.cat([torch.zeros(1, img_H, img_W)]*(n_flow-len(flow_list)), dim=0)
            tt2 = torch.cat(flow_list, dim=0)
            tmp = torch.cat([tt2,tt1], dim=0)
        flow_concat_list.append(tmp)
        if i > 10:
            print("stopping")
            # ipdb.set_trace()
            break
    
    ipdb.set_trace()
    # convert lists to tensors 
    image_list = torch.stack(image_list)
    flow_list = torch.stack(flow_list) 
    needle_masks_list = torch.stack(needle_masks_list)
    mask_list = torch.stack(mask_list)
    flow_concat_list = torch.stack(flow_concat_list)

    print('check dimensions')

    return image_list, flow_list, needle_masks_list, mask_list, flow_concat_list


if __name__ == '__main__':

    n_flow = 3
    i = 1
    image_list, flow_list, needle_masks_list, mask_list, flow_concat_list = [], [], [], [], []
    image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')

    # as flow needs to be computed first frame outisde while 
    img = Image.open(image_path)
    img_tensor = convert_tensor(img)
    image_list.append(img_tensor)
    hsv = np.zeros_like(img)
    cartesian = np.zeros_like(img)
    hsv[..., 1] = 255

    flow_ = torch.zeros_like(img_tensor)[0:2,:,:]
    flow_ = torch.cat([flow_]*3, dim = 0)
    flow_concat_list.append(flow_)

    img_H = img_tensor.shape[1]
    img_W = img_tensor.shape[2]

    # ipdb.set_trace()
    while os.path.exists(image_path):
        # save images
        # image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')

        # next image frame
        image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i+1).zfill(6) + '.PNG')
        img2 = Image.open(image_path)    
        
        # compute optical flow and stack in groups of n
        
        prvs = np.array(img)[:,:,0]
        next = np.array(img2)[:,:,0]

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        print("mean of flows: " , np.mean(flow[..., 0]), " " , np.mean(flow[..., 1]))
        mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])])
        cartesian[..., 0:2] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
        # cartesian[..., 0] = cv2.normalize(flow[..., 0],None, 0, 255, cv2.NORM_MINMAX)
        # cartesian[..., 2] = cv2.normalize(flow[..., 1],None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('flow_cartesian/flow_cartesian_{}.png'.format(i), cartesian)
        flow_list.append(convert_tensor(cartesian)[0:2,:,:])

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite('flow_polar/flow_polar_{}.png'.format(i), bgr)
        # read segmentation masks
        mask_path = os.path.join(MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
        mask = Image.open(mask_path)
        # mask.save('segmentation_mask.png')

        # get just needle masks
        mask_arr = np.array(mask)
        mask_tensor = convert_tensor(mask)
        mask_list.append(mask_tensor)
        # print("shape mask array", mask_arr.shape) 
        
        # finding just needle mask
        mask_needle = 255*(np.array([255,255,255]) - np.where(mask_arr == 51,221,255))
        mask_needle = np.concatenate([np.expand_dims(mask_needle[:,:,0], axis=-1)]*3, axis=-1)
        mask_needle_ = Image.fromarray(mask_needle.astype('uint8'))
        # print("needle mask shape: ", mask_needle_.size)
        mask_needle_.save('needle_mask/pure_needle_{}.png'.format(i))
        needle_masks_list.append(convert_tensor(mask_needle_)[0:1,:,:])
        
        # save combined needle_mask and optical flow
        # ipdb.set_trace()
        flow_mask = cv2.vconcat([mask_needle.astype('uint8'), cartesian])
        cv2.imwrite('flow_and_mask/{}.png'.format(i), flow_mask)

        i += 1
        image_path = os.path.join(IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')


        # update previous image
        img = img2
        # store image 
        img_tensor = convert_tensor(img)
        image_list.append(img_tensor)

        # ipdb.set_trace()
        # concatenate m flow frames along channel dim
        if len(flow_list) >= 3:
            tmp = torch.cat(flow_list[-3:], dim=0)
            ipdb.set_trace()  
        else:
            tt1 = torch.cat([torch.zeros(2, img_H, img_W)]*(n_flow-len(flow_list)), dim=0)
            tt2 = torch.cat(flow_list, dim=0)
            tmp = torch.cat([tt2,tt1], dim=0)
        flow_concat_list.append(tmp)
        if i > 10:
            print("stopping")
            ipdb.set_trace()
            break