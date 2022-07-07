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
# add augmentation 
transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor()
                                #  transforms.Normalize()
                                    ]) 

IMAGE_LOC  = 'JPEGImages'
MASK_LOC = 'SegmentationClass'
# save_number = '2'

def overlay_mask_with_image(img, mask):
    
    pass

def get_dict_vals(data):

    images_tensor = data['images']
    flows_tensor = data['flows']
    needle_masks_tensor = data['needle_masks']
    masks_tensor = data['masks']
    flow_concats_tensor = data['flow_concats']
    
    return images_tensor, flows_tensor, needle_masks_tensor, masks_tensor, flow_concats_tensor 

def get_image(i, dataset, PARENT_FOLDER):
    image_path = os.path.join(PARENT_FOLDER, dataset, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
    # if not os.path.isabs(image_path):
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(img) #img

def get_mask(i, dataset, PARENT_FOLDER):
    mask_path = os.path.join(PARENT_FOLDER, dataset, MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
    # if not os.path.isabs(mask_path):
    if not os.path.exists(mask_path):
        return None, None    
    mask = cv2.imread(mask_path) #Image.open(mask_path)
    mask_resized = cv2.resize(mask, dsize=(256,256), interpolation=cv2.INTER_NEAREST)    
    return mask, mask_resized #Image.open(mask_path)


def find_needle_mask(mask):
    mask_needle = 255*(np.array([255,255,255]) - np.where(mask == 51,221,255))
    mask_needle_ = np.concatenate([np.expand_dims(mask_needle[:,:,0], axis=-1)]*3, axis=-1).astype('uint8')
    mask_needle_pil = Image.fromarray(mask_needle.astype('uint8'))
    # mask_needle_pil.save('needle_mask/pure_needle_{}.png'.format(i))
    mask_needle_resized = cv2.resize(mask_needle, dsize=(256,256), interpolation=cv2.INTER_NEAREST).astype('uint8')
    # cv2.imwrite('needle_mask/pure_needle_resize_{}.png'.format(i), mask_needle_resized)
    # ipdb.set_trace()

    max_mask = np.max(mask_needle_resized)
    ind_ = np.where(mask_needle_resized == max_mask)[-1][0]
    tmp_ = mask_needle_resized[:,:,ind_] # BGR #np.array(mask_needle_resized)[:,:,0]
    # PUTTING CHECK HERE TO NOT USE IMAGES W/O NEEDLE MASK
    if np.max(tmp_) == 0:
        return None
    mask_needle_arr = tmp_ // np.max(tmp_)
    mask_needle_arr = mask_needle_arr.astype(np.int16)

    return mask_needle_arr

def get_data_all_dataset(n_flow, LIST_OF_DATASETS, PARENT_FOLDER, saved_data_file, type='train'):
    
    # initialise list to append stacked data from each dataset
    image_lists, flow_lists, needle_mask_lists, mask_lists, flow_concat_lists = [], [], [], [], []
    good_pixel_count_lists = []
    for j, dataset_name in enumerate(LIST_OF_DATASETS):
        print("dataset # " , j)
        i = 1
        image_list, flow_list, needle_mask_list, mask_list, flow_concat_list = [], [], [], [], []
        good_pixel_count_list = []
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
        # mask_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
        
        # as flow for 1st frame is zero | first frame outisde while 
        img = get_image(i, dataset_name, PARENT_FOLDER)
        mask, mask_resized = get_mask(i, dataset_name, PARENT_FOLDER) #Image.open(mask_path)    
        # ipdb.set_trace()
        if mask is not None:
            mask_needle_arr_resized = find_needle_mask(mask)
            hsv = np.zeros_like(img)
            cartesian = np.zeros_like(img)
            hsv[..., 1] = 255 # scale should be 1, then conversion from hsv to rgb becomes easy

            # initial flow for frame 1 
            flow_ = torch.zeros_like(img)[:,:, 0:1]
            flow_new = torch.cat([flow_]*n_flow, dim = -1)
            
            # # APPEND EVERYTHING TOGETHER
            # if mask_needle_arr_resized is not None:
            # ipdb.set_trace()
            image_list.append(img[:,:,0:1]) #image_list.append(img)
            flow_list.append(flow_)
            flow_concat_list.append(flow_new)
            mask_list.append(torch.from_numpy(mask_resized))
            needle_mask_list.append(torch.from_numpy(mask_needle_arr_resized).unsqueeze(-1))    
            # finding number of good pixels
            # ipdb.set_trace()
            _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True) 
            if count_[1] != 0:
                good_pixel_count_list.append(count_[0]/count_[1])

        img_H = img.shape[0]
        img_W = img.shape[1]

        while os.path.exists(image_path):
            
            # next image frame
            # print("i = " , i)
            i = i + 1
            # mask_path = os.path.join(PARENT_FOLDER, dataset_name, MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')            
            # image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')            
            img_next = get_image(i, dataset_name, PARENT_FOLDER) #Image.open(image_path)        
            if img_next is None:
                print("done")
                break
            mask, mask_resized = get_mask(i, dataset_name, PARENT_FOLDER) #Image.open(mask_path)
            
            # if mask is none then don't append images
            if mask is not None:                
                mask_needle_arr_resized = find_needle_mask(mask)
                # if needle mask is none don't append
                if mask_needle_arr_resized is not None:
                    image_list.append(img_next[:,:,0:1])
                    # mask_path = os.path.join(MASK_LOC, 'frame_' + '000{}'.format(i+1).zfill(6) + '.png')
                    mask_list.append(torch.from_numpy(mask_resized))
                    # finding just needle mask
                    # mask_needle_arr_resized = find_needle_mask(mask)
                    needle_mask_list.append(torch.from_numpy(mask_needle_arr_resized).unsqueeze(-1))
                    # finding number of good pixels
                    # ipdb.set_trace()
                    _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True) 
                    if count_[1] != 0:
                        good_pixel_count_list.append(count_[0]/count_[1])                        
                    # compute optical flow and stack in groups of n
                    prvs = np.array(img)[:,:,0]
                    next = np.array(img_next)[:,:,0]

                    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)    
                    mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])])
                    cartesian[..., [0,2]] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
                    cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
                    
                    flow_list.append(torch.from_numpy(cartesian_bgr)[:,:, 0:1])
                    # concatenate m flow frames along channel dim
                    if len(flow_list) >= n_flow:
                        tmp = torch.cat(flow_list[-n_flow:], dim=-1)              
                    else:
                        tt1 = torch.cat([torch.zeros(img_H, img_W, 1)]*(n_flow-len(flow_list)), dim=-1)
                        tt2 = torch.cat(flow_list, dim=-1)
                        tmp = torch.cat([tt2,tt1], dim=-1)
                    flow_concat_list.append(tmp)
    
                # update previous image
                img = img_next
        
        # convert lists to tensors 
        image_list = torch.stack(image_list).permute(0,3,1,2)/255 #- 0.5
        flow_list = torch.stack(flow_list).permute(0,3,1,2)/255 #- 0.5
        needle_mask_list = torch.stack(needle_mask_list).permute(0,3,1,2)
        mask_list = torch.stack(mask_list).permute(0,3,1,2)
        flow_concat_list = torch.stack(flow_concat_list).permute(0,3,1,2)/255 #- 0.5

        # append to global lists
        image_lists.append(image_list)
        flow_lists.append(flow_list)
        needle_mask_lists.append(needle_mask_list)
        mask_lists.append(mask_list)
        flow_concat_lists.append(flow_concat_list)
        good_pixel_count_lists.append(np.mean(good_pixel_count_list))

    # ipdb.set_trace()
    image_lists = torch.concat(image_lists, dim=0)
    flow_lists = torch.concat(flow_lists, dim=0)
    needle_mask_lists = torch.concat(needle_mask_lists, dim=0)
    mask_lists = torch.concat(mask_lists, dim=0)
    flow_concat_lists = torch.concat(flow_concat_lists, dim=0)

    print("shapes check")
    print("image_lists", image_lists.shape)
    print("flow_lists", flow_lists.shape)
    print("needle_mask_lists", needle_mask_lists.shape)
    print("mask_lists", mask_lists.shape)
    print("flow_concat_lists", flow_concat_lists.shape)

    print('check dimensions')

    print("average ratio of good to bad: ", np.mean(good_pixel_count_lists))

    data_ = {'images':image_lists, 'flows':flow_lists, 'needle_masks':needle_mask_lists, 'masks':mask_lists, 'flow_concats':flow_concat_lists,
             'ratio_bce_loss':np.mean(good_pixel_count_lists)}
    data_dir = os.path.join('saved_data', saved_data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(data_, os.path.join(data_dir, type+'.pt'))
    ipdb.set_trace()

    return data_ #image_list, flow_list, needle_mask_list, mask_list, flow_concat_list

def get_data(n_flow = 3):
    
    i = 1
    image_list, flow_list, needle_mask_list, mask_list, flow_concat_list = [], [], [], [], []
    image_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
    # mask_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
    
    # as flow for 1st frame is zero | first frame outisde while 
    img = get_image(i)
    # img_tensor = transform(img) #convert_tensor(img)
    # ipdb.set_trace()
    image_list.append(img)
    hsv = np.zeros_like(img)
    cartesian = np.zeros_like(img)
    hsv[..., 1] = 255 # scale should be 1, then conversion from hsv to rgb becomes easy

    # initial flow for frame 1 
    flow_ = torch.zeros_like(img)[:,:, 0:1]
    flow_list.append(flow_)
    flow_ = torch.cat([flow_]*3, dim = -1)
    flow_concat_list.append(flow_)

    mask, mask_resized = get_mask(i) #Image.open(mask_path)    
    mask_list.append(torch.from_numpy(mask_resized))

    mask_needle_arr_resized = find_needle_mask(mask)
    needle_mask_list.append(torch.from_numpy(mask_needle_arr_resized).unsqueeze(-1))    

    img_H = img.shape[0]
    img_W = img.shape[1]

    while os.path.exists(image_path):
        
        # next image frame
        print("i = " , i)
        i = i + 1
        image_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
        img_next = get_image(i) #Image.open(image_path)        
        if img_next is None:
            print("done")
            break
        image_list.append(img_next)

        # read segmentation masks
        # mask_path = os.path.join(MASK_LOC, 'frame_' + '000{}'.format(i+1).zfill(6) + '.png')
        mask, mask_resized = get_mask(i) #Image.open(mask_path)
        mask_list.append(torch.from_numpy(mask_resized))
        # finding just needle mask
        mask_needle_arr_resized = find_needle_mask(mask)
        needle_mask_list.append(torch.from_numpy(mask_needle_arr_resized).unsqueeze(-1))    
        # ipdb.set_trace()
        # compute optical flow and stack in groups of n
        prvs = np.array(img)[:,:,0]
        next = np.array(img_next)[:,:,0]

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)    
        mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])])
        cartesian[..., [0,2]] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
        cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
        
        # cv2.imwrite('flow_cartesian/flow_cartesian_{}.png'.format(i), cartesian)
        flow_list.append(torch.from_numpy(cartesian_bgr)[:,:, 0:1])

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
            tmp = torch.cat(flow_list[-3:], dim=-1)              
        else:
            tt1 = torch.cat([torch.zeros(img_H, img_W, 1)]*(n_flow-len(flow_list)), dim=-1)
            tt2 = torch.cat(flow_list, dim=-1)
            tmp = torch.cat([tt2,tt1], dim=-1)
        flow_concat_list.append(tmp)
        # if i > 10:
        #     print("stopping")
        #     # ipdb.set_trace()
        #     break
    
    # convert lists to tensors 
    image_list = torch.stack(image_list).permute(0,3,1,2)/255
    flow_list = torch.stack(flow_list).permute(0,3,1,2)/255 
    needle_mask_list = torch.stack(needle_mask_list).permute(0,3,1,2)
    mask_list = torch.stack(mask_list).permute(0,3,1,2)
    flow_concat_list = torch.stack(flow_concat_list).permute(0,3,1,2)/255

    print("shapes check")
    print("image_list", image_list.shape)
    print("flow_list", flow_list.shape)
    print("needle_mask_list", needle_mask_list.shape)
    print("mask_list", mask_list.shape)
    print("flow_concat_list", flow_concat_list.shape)

    print('check dimensions')

    data_ = {'images':image_list, 'flows':flow_list, 'needle_masks':needle_mask_list, 'masks':mask_list, 'flow_concats':flow_concat_list}
    data_dir = os.path.join('saved_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(data_, os.path.join(data_dir, 'test.pt'))
    ipdb.set_trace()

    return image_list, flow_list, needle_mask_list, mask_list, flow_concat_list


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