import enum
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
transform_ = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomRotation(15),
                               transforms.RandomVerticalFlip(p=0.5),                               
                            ])

transform_image = transforms.Compose([
                                    transforms.GaussianBlur(kernel_size=3, sigma=(0.01,2)),
                                    # transforms.ColorJitter(brightness=(0,0.1), contrast=(0,0.1))
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
    # mask_needle_ = np.concatenate([np.expand_dims(mask_needle[:,:,0], axis=-1)]*3, axis=-1).astype('uint8')
    # mask_needle_pil = Image.fromarray(mask_needle.astype('uint8'))
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

def concatenate_m_flow_frames(flow_list, n_flow, img_H, img_W):
    if len(flow_list) >= n_flow:
        tmp = torch.cat(flow_list[-n_flow:], dim=-1)              
    else:
        tt1 = torch.cat([torch.zeros(img_H, img_W, 1)]*(n_flow-len(flow_list)), dim=-1)
        tt2 = torch.cat(flow_list, dim=-1)
        tmp = torch.cat([tt2,tt1], dim=-1)

    return tmp  

def concatenate_m_flow_frames_np(flow_list, n_flow, img_H, img_W, gamma=0.5):
    if len(flow_list) >= n_flow:
        tmp = np.concatenate(flow_list[-n_flow:], axis=-1)              
    else:
        tt1 = np.concatenate([np.zeros((img_H, img_W, 1))]*(n_flow-len(flow_list)), axis=-1)
        tt2 = np.concatenate(flow_list, axis=-1)
        tmp = np.concatenate([tt2,tt1], axis=-1)

    # multiply tmp past with weights to have forgetting factor
    tmp = np.concatenate([tmp[:,:,i:i+1]*(gamma**i) for i in range(n_flow)],axis=-1)

    return tmp        

def concatenate_m_flow_frames_raw(flow_xy_list, n_flow, img_H, img_W):
    # two channels per flow
    scale = 2 
    ipdb.set_trace()
    if len(flow_xy_list) >= scale*n_flow:
        tmp = torch.cat(flow_xy_list[-scale*n_flow:], dim=-1)              
    else:
        tt1 = torch.cat([torch.zeros(img_H, img_W, 1)]*(scale*n_flow-len(flow_xy_list)), dim=-1)
        tt2 = torch.cat(flow_xy_list, dim=-1)
        tmp = torch.cat([tt2,tt1], dim=-1)

    return tmp    

def find_flow_history(image_list, img_H, img_W, n_history=1, abs_flag = False):
    # find flow b/w current image and past n_history of images 
    flow_list = []
    flowX_list = []
    flowY_list = []
    cartesian = np.zeros((img_H,img_W,3), dtype=np.uint8)
    for k in range(n_history):    
        if len(image_list) > n_history:        
            prvs = np.array(image_list[-1-k])
            next = np.array(image_list[-1-k-1])

            # ipdb.set_trace()
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)                        
            mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])]) #extra 
            if abs_flag:
                flow = np.abs(flow) 
            cartesian[..., [0,2]] = cv2.normalize(flow,None, 0, 255, cv2.NORM_MINMAX)
            cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)            
            flow_list.append(cartesian_bgr[:,:,0:1])
            flowX_list.append(flow[..., 0:1])
            flowY_list.append(flow[..., 1:2])
        else:
            flow_list.append(np.zeros((img_H, img_W, 1), dtype=np.uint8))
            flowX_list.append(np.zeros((img_H, img_W, 1), dtype=np.uint8))
            flowY_list.append(np.zeros((img_H, img_W, 1), dtype=np.uint8))

    # print("flow shape check ")
    # print(np.concatenate(flow_list, axis=-1).shape)
    # print(np.concatenate(flowX_list, axis=-1).shape)
    return np.concatenate(flow_list, axis=-1), np.concatenate(flowX_list, axis=-1), np.concatenate(flowY_list, axis=-1)



def transform_all_in_same_way(inp, type, n_flow):
    # inp is a list of all things which go in dict
    out_trans = torch.cat(inp, dim=0)
    # inp_ = torch.stack(inp, dim=0)
    if type == 'train':
        # gaussian blur for img
        tmp_ = torch.cat([torch.cat([out_trans[0:1]]*3, dim = 0).unsqueeze(0), 
                            torch.cat([out_trans[1:2]]*3, dim = 0).unsqueeze(0)                            
                        ], dim = 0)
        # tmp_ = torch.cat([out_trans[0:1].unsqueeze(1), out_trans[1:2].unsqueeze(1), out_trans[7:7+n_flow].unsqueeze(1)], dim = 0)
        tmp_ = transform_image(tmp_) #image transform first two entries only 
        out_trans[0:1] = tmp_[0,0:1]
        out_trans[1:2] = tmp_[1,0:1]
        # gaussian blur the flow | #? interesting when transform applied to 1 channel vs 3 channels 
        # ipdb.set_trace()
        tmp_ = transform_image(out_trans[7:7+n_flow].unsqueeze(1))
        out_trans[7:7+n_flow] = tmp_.squeeze(1)

        out_trans = transform_(out_trans)

    return out_trans

def update_max_min_flows(flow, max_flow, min_flow):
    # update x flow
    if np.max(flow[..., 0]) > max_flow[0]:
        max_flow[0] = np.max(flow[..., 0])    
    if np.min(flow[..., 0] < min_flow[0]):
        min_flow[0] = np.min(flow[..., 0])

    # update y flow
    if np.max(flow[..., 1]) > max_flow[1]:
        max_flow[1] = np.max(flow[..., 1])    
    if np.min(flow[..., 1] < min_flow[1]):
        min_flow[1] = np.min(flow[..., 1])

    return max_flow, min_flow

def get_data_dict_history(n_flow, LIST_OF_DATASETS, PARENT_FOLDER, saved_data_file, type='train', flow_history_flag=False ,max_flow=[0,0], min_flow=[100,100]):
    
    # initialise list to append stacked data from each dataset
    image_lists, flow_lists, needle_mask_lists, mask_lists, flow_concat_lists = [], [], [], [], []
    all_data_dict = []
    good_pixel_count_lists = []
    n_history = 1
    count_needles = 0
    count_no_needles = 0
    # max flow and min flow in training data only and use same in test data             
    for j, dataset_name in enumerate(LIST_OF_DATASETS):
        print("dataset # " , j)
        i = 0
        image_list, flow_list, needle_mask_list, mask_list, flow_concat_list = [], [], [], [], []
        good_pixel_count_list = []
        flow_x_list, flow_y_list = [], []
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
        # mask_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
        
        # as flow for 1st frame is zero | first frame outisde while 
        img = get_image(i, dataset_name, PARENT_FOLDER)
        mask, mask_resized = get_mask(i, dataset_name, PARENT_FOLDER) #Image.open(mask_path)    
        # ipdb.set_trace()
        if mask is not None:
            mask_needle_arr_resized = find_needle_mask(mask)
            if mask_needle_arr_resized is None: 
                needle_present = 0 
                mask_needle_arr_resized = np.zeros((256,256), dtype=np.uint8)
                count_no_needles += 1
            else:
                needle_present = 1
                count_needles += 1

            cartesian = np.zeros_like(img)

            # # initial flow for frame 1 
            # flow_ = torch.zeros_like(img)[:,:, 0:1]
            flow_ = np.zeros_like(img[:,:, 0:1])#[:,:, 0:1]
            # flow_ = torch.zeros_like(img)[:,:, :]
            #? may need to change concatenate below to stack 
            flow_new = flow_ #np.concatenate([flow_]*n_history, axis=-1) #add dim here and then wrap around flow history: n_history x 3 x H x W
            flow_x = flow_new #torch.cat([flow_]*n_flow, dim = -1)
            flow_y = flow_new #torch.cat([flow_]*n_flow, dim = -1)
            flow_list.append(flow_)
            flow_x_list.append(flow_x)
            flow_y_list.append(flow_y)
            
            if flow_history_flag:
                all_data_dict.append({'images':img[:,:,0:1],
                                      'images_prev':torch.zeros_like(img[:,:,0:1]),
                                    'flows':torch.from_numpy(flow_),
                                    'needle_masks':mask_needle_arr_resized,
                                    'masks':mask_resized,
                                    'flow_concats':torch.from_numpy(flow_new),
                                    'flow_x':torch.from_numpy(flow_x),
                                    'flow_y':torch.from_numpy(flow_y),
                                     'needle_label': torch.tensor([needle_present])
                                    })  

            else:
                all_data_dict.append({'images':img[:,:,0:1],
                                    'images_prev':torch.zeros_like(img[:,:,0:1]),
                                    'flows':flow_, 
                                    'needle_masks':np.expand_dims(mask_needle_arr_resized, axis=-1),
                                    'masks':mask_resized,
                                    'flow_concats':flow_new,
                                    'flow_x':flow_x,
                                    'flow_y':flow_y,
                                    'needle_label': torch.tensor([needle_present])
                                    })
                # ipdb.set_trace()
                # all_data_dict.append({'images':img,
                #                     'images_prev':torch.zeros_like(img),
                #                     'flows': flow_, #np.concatenate([flow_]*3, axis = -1), 
                #                     'needle_masks':np.stack([mask_needle_arr_resized]*3, axis=-1),
                #                     'masks':mask_resized,
                #                     'flow_concats':flow_new,
                #                     'flow_x':flow_x,
                #                     'flow_y':flow_y
                #                     })                                    


            # finding number of good pixels
            # _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True) 
            # if count_[1] != 0:
            #     good_pixel_count_list.append(count_[0]/count_[1])

        img_H = img.shape[0]
        img_W = img.shape[1]
        while os.path.exists(image_path):            
            # print("i = " , i)
            i = i + 1            
            img_next = get_image(i, dataset_name, PARENT_FOLDER) #Image.open(image_path)        
            if img_next is None:
                print("done")
                break
            mask, mask_resized = get_mask(i, dataset_name, PARENT_FOLDER) #Image.open(mask_path)            
            # if mask is none then don't append images
            if mask is not None:                
                mask_needle_arr_resized = find_needle_mask(mask)
                # if needle mask is none don't append | make mask all zeros
                if mask_needle_arr_resized is None: 
                    mask_needle_arr_resized = np.zeros((256,256), dtype=np.uint8)
                    needle_present = 0
                    count_no_needles += 1
                    # ipdb.set_trace()
                else:
                    needle_present = 1
                    count_needles += 1
                    # ipdb.set_trace()
                    # append image, mask, mask_needle                    
                    image_list.append(img_next[:,:,0:1])   
                    needle_mask_list.append(mask_needle_arr_resized)
                    mask_list.append(mask)                                              
                    if type == 'train': 
                        _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True)                 
                        if count_[1] != 0:
                            good_pixel_count_list.append(count_[0]/count_[1])                        
                    
                    #compute optical flow and stack in groups of n
                    prvs = np.array(img)[:,:,0]
                    next = np.array(img_next)[:,:,0]

                    # computing flow b/w [T,T-1] & [T,T-2] image as well
                    if flow_history_flag:                        
                        flow, flowX, flowY = find_flow_history(image_list, img_H, img_W, n_history, abs_flag = False)
                        flow_list.append(flow)

                        all_data_dict.append({'images':img_next[:,:,0:1],
                                             'images_prev':img[:,:,0:1],
                                            'flows':torch.from_numpy(flow[:,:,-1:]),
                                            'needle_masks':mask_needle_arr_resized,
                                            'masks':mask_resized,
                                            'flow_concats':torch.from_numpy(flow),
                                            'flow_x':torch.from_numpy(flowX),
                                            'flow_y':torch.from_numpy(flowY),
                                             'needle_label': torch.tensor([needle_present])
                                            })  

                    else:
                        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)                        
                        mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])]) #extra 
                        flow = np.abs(flow) 
                        cartesian[..., [0,2]] = cv2.normalize(flow,None, 0, 255, cv2.NORM_MINMAX)
                        cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
                        flow_list.append(cartesian_bgr[:,:, 0:1]) #torch.from_numpy(cartesian_bgr)[:,:, 0:1]

                        # # to get a colored flow
                        # # hsv[..., [0,2]] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
                        # # cartesian_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        
                        flow_x_list.append(flow[..., 0:1]) #torch.from_numpy(flow[..., 0:1])
                        flow_y_list.append(flow[..., 1:2]) # torch.from_numpy(flow[..., 1:2])
                        # concatenate m flow frames along channel dim
                        tmp = concatenate_m_flow_frames_np(flow_list, n_flow, img_H, img_W)
                        ## INPUTTING NOISE                     
                        # tmp = torch.randint(0, 255, (img_W, img_H, 1))
                        tmpX = concatenate_m_flow_frames_np(flow_x_list, n_flow, img_H, img_W)
                        tmpY = concatenate_m_flow_frames_np(flow_y_list, n_flow, img_H, img_W)                                        
                    
                        all_data_dict.append({'images':img_next[:,:,0:1],
                                             'images_prev':img[:,:,0:1],
                                            'flows':cartesian_bgr[:,:,0:1], 
                                            'needle_masks':np.expand_dims(mask_needle_arr_resized, axis=-1),
                                            'masks':mask_resized,
                                            'flow_concats':tmp,
                                            'flow_x':tmpX,
                                            'flow_y':tmpY,
                                            'needle_label': torch.tensor([needle_present])
                                            })   
                        # ipdb.set_trace()
                        # all_data_dict.append({'images':img_next,
                        #                     'images_prev':img,
                        #                     'flows': cartesian_bgr, 
                        #                     'needle_masks':np.stack([mask_needle_arr_resized]*3, axis=-1),
                        #                     'masks':mask_resized,
                        #                     'flow_concats':np.concatenate([tmp]*3, axis=-1),
                        #                     'flow_x':np.concatenate([tmpX]*3, axis=-1),
                        #                     'flow_y':np.concatenate([tmpY]*3, axis=-1)
                        #                     })                                    
                        

                    ## find max and min flow in x  and y directions for train data only
                    if type == 'train':
                        max_flow, min_flow = update_max_min_flows(flow, max_flow, min_flow)
                # update previous image
                img = img_next

    # ipdb.set_trace()
    ## update flow in all entire data, normalize by global max and min 
    print("number of needle labels = " , count_needles, "  number of no needle labels = " , count_no_needles)
    print("max_flow = " , max_flow)
    print("min_flow = " , min_flow)
    cartesian = np.zeros_like(img)
    max_flow_ = np.max(max_flow)

    for k, data in enumerate(all_data_dict):
        # NORMALIZING
        if not flow_history_flag:
            updated_flow_x = (data['flow_x'] - min_flow[0])/max_flow_ #(max_flow[0] - min_flow[0])
            updated_flow_y = (data['flow_y'] - min_flow[1])/max_flow_ #(max_flow[1] - min_flow[1])     
            # print("flow x shape : " , updated_flow_x.shape)   
            # print("flow y shape : " , updated_flow_y.shape) 
            range_ = [k-n_flow+1,k+1] if k > n_flow else [0,k+1]  
            flow_bgr = []
            for j in range(n_flow):
                updated_flow = np.concatenate([updated_flow_x[:,:,j:j+1], updated_flow_y[:,:,j:j+1]], axis=-1)
                cartesian[..., [0,2]] = cv2.normalize(updated_flow,None, 0, 255, cv2.NORM_MINMAX)
                cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
                flow_bgr.append(cartesian_bgr[:,:,0:1])
                # flow_bgr.append(cartesian_bgr)
            
            flow_bgr = np.concatenate(flow_bgr, axis = -1)    
            
            data['flow_x'] = torch.from_numpy(updated_flow_x)
            data['flow_y'] = torch.from_numpy(updated_flow_y)
            data['flows'] = torch.from_numpy(cartesian_bgr[:,:,0:1]) #cartesian_bgr[:,:,0:1]
            data['flow_concats'] = torch.from_numpy(flow_bgr)

            # print("SHAPES : ")
            # print(data['flow_x'].shape) 
            # print(data['flow_y'].shape) 
            # print(data['flows'].shape) 
            # print(data['flow_concats'].shape) 

        out_transform = transform_all_in_same_way([data['images'].permute(2,0,1),data['images_prev'].permute(2,0,1),data['flows'].permute(2,0,1),
                                torch.from_numpy(data['needle_masks']).permute(2,0,1),torch.from_numpy(data['masks']).permute(2,0,1), 
                                data['flow_concats'].permute(2,0,1), data['flow_x'].permute(2,0,1), data['flow_y'].permute(2,0,1)], type, n_flow)
        # print("out_trans shape" , out_transform.shape)
        if flow_history_flag:
            n_flow = n_history
        # as every input has 3 channels 
        key_list = ['images', 'images_prev','flows','needle_masks','masks','flow_concats','flow_x','flow_y']
        ind_list = [[0,1],[1,2],[2,3],[3,4],[4,7],[7,7+n_flow],[7+n_flow,7+2*n_flow],[7+2*n_flow,7+3*n_flow]]
        for key,ind in zip(key_list, ind_list):
            data[key] = out_transform[ind[0]:ind[1]]            

        # ipdb.set_trace()
        # data['images'] = out_transform[0:1,:,:]
        # data['images_prev'] = out_transform[1:2,:,:]
        # data['flows'] = out_transform[2:3,:,:]
        # data['needle_masks'] = out_transform[3:4,:,:]
        # data['masks'] = out_transform[4:7,:,:]
        # data['flow_concats'] = out_transform[7:7+n_flow,:,:]
        # data['flow_x'] = out_transform[7+n_flow:7+2*n_flow,:,:]
        # data['flow_y'] = out_transform[7+2*n_flow:7+3*n_flow,:,:]

        # all_data_dict.append({'images':out_transform[0:1,:,:],  #1 dim
        #                     'flows':out_transform[1:2,:,:],     #1 dim 
        #                     'needle_masks':out_transform[2:3,:,:],  #1 dim
        #                     'masks':out_transform[3:6,:,:],    # 3 dim 
        #                     'flow_concats':out_transform[6:6+n_flow,:,:],
        #                     'flow_x':out_transform[6+n_flow:6+2*n_flow,:,:],
        #                     'flow_y':out_transform[6+2*n_flow:6+3*n_flow,:,:]})  #note flow_concats size depends on n_flow                        

        # ipdb.set_trace()
    data_dir = os.path.join('saved_data', saved_data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(all_data_dict, os.path.join(data_dir, type+'.pt'))
    ipdb.set_trace()

    if type == 'train':
        return all_data_dict, max_flow, min_flow 
    else:
        return all_data_dict, None, None



def get_data_dict(n_flow, LIST_OF_DATASETS, PARENT_FOLDER, saved_data_file, type='train', flow_history_flag=False ,max_flow=[0,0], min_flow=[100,100]):
    
    # initialise list to append stacked data from each dataset
    image_lists, flow_lists, needle_mask_lists, mask_lists, flow_concat_lists = [], [], [], [], []
    all_data_dict = []
    good_pixel_count_lists = []
    # max flow and min flow in training data only and use same in test data             
    for j, dataset_name in enumerate(LIST_OF_DATASETS):
        print("dataset # " , j)
        i = 0
        image_list, flow_list, needle_mask_list, mask_list, flow_concat_list = [], [], [], [], []
        good_pixel_count_list = []
        flow_x_list, flow_y_list = [], []
        image_path = os.path.join(PARENT_FOLDER, dataset_name, IMAGE_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.PNG')
        # mask_path = os.path.join(PARENT_FOLDER, LIST_OF_DATASETS[0], MASK_LOC, 'frame_' + '000{}'.format(i).zfill(6) + '.png')
        
        # as flow for 1st frame is zero | first frame outisde while 
        img = get_image(i, dataset_name, PARENT_FOLDER)
        mask, mask_resized = get_mask(i, dataset_name, PARENT_FOLDER) #Image.open(mask_path)    
        # ipdb.set_trace()
        if mask is not None:
            mask_needle_arr_resized = find_needle_mask(mask)
            cartesian = np.zeros_like(img)

            # # initial flow for frame 1 
            # flow_ = torch.zeros_like(img)[:,:, 0:1]
            flow_ = np.zeros_like(img)[:,:, 0:1]
            # flow_ = torch.zeros_like(img)[:,:, :]
            flow_new = np.concatenate([flow_]*n_flow, axis=-1) #torch.cat([flow_]*n_flow, dim = -1)
            flow_x = flow_new #torch.cat([flow_]*n_flow, dim = -1)
            flow_y = flow_new #torch.cat([flow_]*n_flow, dim = -1)
            flow_list.append(flow_)
            flow_x_list.append(flow_x)
            flow_y_list.append(flow_y)
            
            all_data_dict.append({'images':img[:,:,0:1],
                                  'images_prev':np.zeros_like(img[:,:,0:1]),
                                'flows':flow_, 
                                'needle_masks':mask_needle_arr_resized,
                                'masks':mask_resized,
                                'flow_concats':flow_new,
                                'flow_x':flow_x,
                                'flow_y':flow_y
                                })

            # finding number of good pixels
            _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True) 
            if count_[1] != 0:
                good_pixel_count_list.append(count_[0]/count_[1])

        img_H = img.shape[0]
        img_W = img.shape[1]
        while os.path.exists(image_path):            
            # print("i = " , i)
            i = i + 1            
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
                    # append image, mask, mask_needle                    
                    image_list.append(img_next[:,:,0:1])   
                    needle_mask_list.append(mask_needle_arr_resized)
                    mask_list.append(mask)                                               
                    _ , count_ = np.unique(mask_needle_arr_resized, return_counts=True) 
                    if count_[1] != 0:
                        good_pixel_count_list.append(count_[0]/count_[1])                        
                    
                    #compute optical flow and stack in groups of n
                    prvs = np.array(img)[:,:,0]
                    next = np.array(img_next)[:,:,0]


                    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)                        
                    mean_flow = np.array([np.mean(flow[..., 0]), np.mean(flow[..., 1])]) #extra 
                    flow = np.abs(flow) 
                    cartesian[..., [0,2]] = cv2.normalize(flow,None, 0, 255, cv2.NORM_MINMAX)
                    cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
                    flow_list.append(cartesian_bgr[:,:, 0:1]) #torch.from_numpy(cartesian_bgr)[:,:, 0:1]

                    # # to get a colored flow
                    # # hsv[..., [0,2]] = cv2.normalize(flow-mean_flow,None, 0, 255, cv2.NORM_MINMAX)
                    # # cartesian_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    
                    flow_x_list.append(flow[..., 0:1]) #torch.from_numpy(flow[..., 0:1])
                    flow_y_list.append(flow[..., 1:2]) # torch.from_numpy(flow[..., 1:2])
                    # concatenate m flow frames along channel dim
                    tmp = concatenate_m_flow_frames_np(flow_list, n_flow, img_H, img_W)
                    ## INPUTTING NOISE                     
                    # tmp = torch.randint(0, 255, (img_W, img_H, 1))
                    tmpX = concatenate_m_flow_frames_np(flow_x_list, n_flow, img_H, img_W)
                    tmpY = concatenate_m_flow_frames_np(flow_y_list, n_flow, img_H, img_W)                                        
                
                    all_data_dict.append({'images':img_next[:,:,0:1],
                                         'images_prev':img[:,:,0:1],
                                        'flows':cartesian_bgr[:,:,0:1], 
                                        'needle_masks':mask_needle_arr_resized,
                                        'masks':mask_resized,
                                        'flow_concats':tmp,
                                        'flow_x':tmpX,
                                        'flow_y':tmpY
                                        })   

                    ## find max and min flow in x  and y directions for train data only
                    if type == 'train':
                        max_flow, min_flow = update_max_min_flows(flow, max_flow, min_flow)
                # update previous image
                img = img_next

    # ipdb.set_trace()
    ## update flow in all entire data, normalize by global max and min 
    print("max_flow = " , max_flow)
    print("min_flow = " , min_flow)
    cartesian = np.zeros_like(img)
    max_flow_ = np.max(max_flow)

    for k, data in enumerate(all_data_dict):
        # NORMALIZING
        updated_flow_x = (data['flow_x'] - min_flow[0])/max_flow_ #(max_flow[0] - min_flow[0])
        updated_flow_y = (data['flow_y'] - min_flow[1])/max_flow_ #(max_flow[1] - min_flow[1])     
        # print("flow x shape : " , updated_flow_x.shape)   
        # print("flow y shape : " , updated_flow_y.shape) 
        range_ = [k-n_flow+1,k+1] if k > n_flow else [0,k+1]  
        flow_bgr = []
        for j in range(n_flow):
            updated_flow = np.concatenate([updated_flow_x[:,:,j:j+1], updated_flow_y[:,:,j:j+1]], axis=-1)
            cartesian[..., [0,2]] = cv2.normalize(updated_flow,None, 0, 255, cv2.NORM_MINMAX)
            cartesian_bgr = cv2.cvtColor(cartesian, cv2.COLOR_HSV2BGR)
            flow_bgr.append(cartesian_bgr[:,:,0:1])
        
        flow_bgr = np.concatenate(flow_bgr, axis = -1)    
        
        data['flow_x'] = torch.from_numpy(updated_flow_x)
        data['flow_y'] = torch.from_numpy(updated_flow_y)
        data['flows'] = torch.from_numpy(cartesian_bgr[:,:,0:1]) #cartesian_bgr[:,:,0:1]
        data['flow_concats'] = torch.from_numpy(flow_bgr)

        out_transform = transform_all_in_same_way([data['images'].permute(2,0,1), data['images_prev'].permute(2,0,1),data['flows'].permute(2,0,1),
                                torch.from_numpy(data['needle_masks']).unsqueeze(-1).permute(2,0,1),torch.from_numpy(data['masks']).permute(2,0,1), 
                                data['flow_concats'].permute(2,0,1), data['flow_x'].permute(2,0,1), data['flow_y'].permute(2,0,1)], type)
        data['images'] = out_transform[0:1,:,:]
        data['immages_prev'] = out_transform[1:2,:,:]
        data['flows'] = out_transform[2:3,:,:]
        data['needle_masks'] = out_transform[3:4,:,:]
        data['masks'] = out_transform[4:7,:,:]
        data['flow_concats'] = out_transform[7:7+n_flow,:,:]
        data['flow_x'] = out_transform[7+n_flow:7+2*n_flow,:,:]
        data['flow_y'] = out_transform[7+2*n_flow:7+3*n_flow,:,:]

        # all_data_dict.append({'images':out_transform[0:1,:,:],  #1 dim
        #                     'flows':out_transform[1:2,:,:],     #1 dim 
        #                     'needle_masks':out_transform[2:3,:,:],  #1 dim
        #                     'masks':out_transform[3:6,:,:],    # 3 dim 
        #                     'flow_concats':out_transform[6:6+n_flow,:,:],
        #                     'flow_x':out_transform[6+n_flow:6+2*n_flow,:,:],
        #                     'flow_y':out_transform[6+2*n_flow:6+3*n_flow,:,:]})  #note flow_concats size depends on n_flow                        

        # ipdb.set_trace()
    data_dir = os.path.join('saved_data', saved_data_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(all_data_dict, os.path.join(data_dir, type+'.pt'))
    # ipdb.set_trace()

    if type == 'train':
        return all_data_dict, max_flow, min_flow 
    else:
        return all_data_dict, None, None

def compute_flow(imgs):
    # prvs = np.array(imgs[0])[:,:,0]
    prvs = imgs[0][:,:,0]
    next = imgs[1][:,:,0]

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)        

    return flow 

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