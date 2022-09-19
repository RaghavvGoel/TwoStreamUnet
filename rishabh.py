import torch
import numpy as np
import math
import torch.nn.functional as F 


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=float(1e-7), kalman_flag= False, eval = False, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)


    #intersection = torch.sum(gt * pr)
    #union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    iou = torch.tensor(0.).to('cuda')
    intersection_sum = torch.tensor(0.).to('cuda')
    tp_by_all_positive = torch.tensor(0.).to('cuda')
    batch = pr.shape[0]
    num_channels = pr.shape[-3] # this will be consistent irrespective of image/video    
    if kalman_flag:
        for i in range(num_channels):
            gt_sum = gt.sum(dim=[-2, -1])
            pr_sum = pr.sum(dim=[-2, -1])
            inds = gt_sum.nonzero()
            if len(inds) > 0:
                gt = gt[inds[:, 0], inds[:, 1],: , :, :]
                pr = pr[inds[:, 0], inds[:, 1],: , :, :]
                intersection = torch.sum(gt * pr, dim=[-2, -1])
                union = torch.sum(gt, dim=[-2, -1]) + torch.sum(pr, dim=[-2, -1]) - intersection                
                iou_channel = torch.mean((intersection) / (union + eps))
                tp_by_all_positive += torch.mean(intersection / torch.sum(gt, dim=[-2, -1]))
                iou += iou_channel
                intersection_sum += torch.mean(intersection)

            elif eval:
                # print("GT has no needle and PR also has no needle")
                return None, None, None
    else:    
        for i in range(num_channels):
            gt_sum = gt.sum(dim=[-2, -1])
            inds = gt_sum.nonzero()
            if len(inds) > 0:
                gt = gt[inds[:, 0],: , :, :]
                pr = pr[inds[:, 0],: , :, :]
                intersection = torch.sum(gt * pr, dim=[-2, -1])
                union = torch.sum(gt, dim=[-2, -1]) + torch.sum(pr, dim=[-2, -1]) - intersection            
                iou_channel = torch.mean((intersection) / (union + eps))
                iou += iou_channel
                intersection_sum += torch.mean(intersection)
        # iou = iou / batch # need to divide by batches 
    
    return iou/num_channels, tp_by_all_positive, intersection_sum
        # num_channels = pr.shape[2]
        # for i in range(num_channels):
        #     import ipdb; ipdb.set_trace()
        #     intersection_channel = torch.sum(pr[:, :, i]*gt[:,:,i], dim=[-2,-1])
        #     union_channel = torch.sum(gt[: , :, i], dim=[-2, -1]) + torch.sum(pr[:,:,i], dim=[-2, -1]) - intersection_channel + eps
        #     iou_channel = (intersection_channel)/ union_channel
        #     iou_channel = torch.mean(iou_channel)
        #     iou += iou_channel


jaccard = iou


def dice_score(pr, gt, eps=float(1e-7), kalman_flag= False, eval=False, threshold=None, ignore_channels=None):
    """Calculate Dice Score
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: dice_score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)


    #intersection = torch.sum(gt * pr)
    #union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    dsc = torch.tensor(0.).to('cuda')
    intersection_sum = torch.tensor(0.).to('cuda')
    tp_by_all_positive = torch.tensor(0.).to('cuda')
    batch = pr.shape[0]
    num_channels = pr.shape[-3] # this will be consistent irrespective of image/video    
    if kalman_flag:
        for i in range(num_channels):
            gt_sum = gt.sum(dim=[-2, -1])
            pr_sum = pr.sum(dim=[-2,-1])
            inds = gt_sum.nonzero()
            if len(inds) > 0:
                gt = gt[inds[:, 0], inds[:, 1],: , :, :]
                pr = pr[inds[:, 0], inds[:, 1],: , :, :]
                intersection = torch.sum(gt * pr, dim=[-2, -1])
                union = torch.sum(gt, dim=[-2, -1]) + torch.sum(pr, dim=[-2, -1])              
                dice_channel = torch.mean(2*(intersection) / (union + eps))
                tp_by_all_positive += torch.mean(intersection / torch.sum(gt, dim=[-2, -1]))
                dsc += dice_channel
                intersection_sum += torch.mean(intersection)
            elif eval: # and len(pr_sum.nonzero()) == 0:
                # print("no prediction made, don't count")
                return None 

    else:    
        for i in range(num_channels):
            gt_sum = gt.sum(dim=[-2, -1])
            pr_sum = pr.sum(dim=[-2,-1])
            inds = gt_sum.nonzero()
            if len(inds) > 0:
                gt = gt[inds[:, 0],: , :, :]
                pr = pr[inds[:, 0],: , :, :]
                intersection = torch.sum(gt * pr, dim=[-2, -1])
                union = torch.sum(gt, dim=[-2, -1]) + torch.sum(pr, dim=[-2, -1])             
                dice_channel = torch.mean(2*(intersection) / (union + eps))
                dsc += dice_channel
                intersection_sum += torch.mean(intersection)
            elif eval: #and len(pr_sum.nonzero()) == 0:
                # print("no prediction made, don't count")
                return None
        # dsc = dsc / batch # need to divide by batches 
    
    return dsc/num_channels


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def f_score_weighted(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, weights = []):
    """
    Modification of the F-score function for incorporating channel weights
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        weights (list): weights for each channel
    Returns:
        float: Weighted F score
    """

    # global device

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)



    num_channels = pr.shape[1]

    if weights==[]:
        weights = [1.0]*num_channels
    else:
        assert(len(weights)==num_channels)
    
    
    # Previous implementation (Dec 2021, Jan 2021)

    # for i in range(num_channels):
    #     pr[:,i,:,:] = weights[i]*pr[:,i,:,:]
    #     gt[:,i,:,:] = weights[i]*gt[:,i,:,:]

    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    # fn = torch.sum(gt) - tp
    


    # Jan 2022 implemetation - (weights not squared at tp)
    

    
    weight_matrix = torch.ones(pr.shape).to(pr.get_device())


    for i in range(num_channels):
        weight_matrix[:,i,:,:] = weights[i]*weight_matrix[:,i,:,:]

    # ipdb.set_trace()

    tp = torch.sum(gt * pr * weight_matrix)
    fp = torch.sum(pr * weight_matrix) - tp
    fn = torch.sum(gt * weight_matrix) - tp




    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

def log_dice_coeff(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, weights = []):
    """
    Modification of the F-score function for incorporating channel weights
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        weights (list): weights for each channel
    Returns:
        float: Weighted F score
    """

    # global device

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)



    num_channels = pr.shape[1]

    if weights==[]:
        weights = [1.0]*num_channels
    else:
        assert(len(weights)==num_channels)
    
    
    # Previous implementation (Dec 2021, Jan 2021)

    # for i in range(num_channels):
    #     pr[:,i,:,:] = weights[i]*pr[:,i,:,:]
    #     gt[:,i,:,:] = weights[i]*gt[:,i,:,:]

    # tp = torch.sum(gt * pr)
    # fp = torch.sum(pr) - tp
    # fn = torch.sum(gt) - tp
    


    # Jan 2022 implemetation - (weights not squared at tp)
    

    
    weight_matrix = torch.ones(pr.shape).to(pr.get_device())


    for i in range(num_channels):
        weight_matrix[:,i,:,:] = weights[i]*weight_matrix[:,i,:,:]

    # ipdb.set_trace()

    tp = torch.sum(gt * pr * weight_matrix)
    fp = torch.sum(pr * weight_matrix) - tp
    fn = torch.sum(gt * weight_matrix) - tp




    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    k=torch.pow((score),0.1)

    return k

def entropy_(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, weights = []):
    """
    Modification of the F-score function for incorporating channel weights
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        weights (list): weights for each channel
    Returns:
        float: Weighted F score
    """

    # global device

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)



    num_channels = pr.shape[1]

    if weights==[]:
        weights = [1.0]*num_channels
    else:
        assert(len(weights)==num_channels)

    weight_matrix = torch.ones(pr.shape).to(pr.get_device())

    for i in range(num_channels):
        weight_matrix[:,i,:,:] = weights[i]*weight_matrix[:,i,:,:]

    # ipdb.set_trace()

    # tp = torch.sum(gt * pr * weight_matrix)
    # fp = torch.sum(pr * weight_matrix) - tp
    # fn = torch.sum(gt * weight_matrix) - tp

    pr_log = -torch.log(pr)
    score=torch.mean(pr_log*gt*weight_matrix)
    # score=torch.pow(-gt_sum*torch.log(pr_sum+eps),1)
    return (score)

def focal_(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None, weights = []):
    """
    Modification of the F-score function for incorporating channel weights
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
        weights (list): weights for each channel
    Returns:
        float: Weighted F score
    """

    # global device

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)



    num_channels = pr.shape[1]

    if weights==[]:
        weights = [1.0]*num_channels
    else:
        assert(len(weights)==num_channels)

    weight_matrix = torch.ones(pr.shape).to(pr.get_device())

    for i in range(num_channels):
        weight_matrix[:,i,:,:] = weights[i]*weight_matrix[:,i,:,:]

    # ipdb.set_trace()

    # tp = torch.sum(gt * pr * weight_matrix)
    # fp = torch.sum(pr * weight_matrix) - tp
    # fn = torch.sum(gt * weight_matrix) - tp
    gamma=1.1
    pr_log = -torch.log(pr)
    score=torch.mean((torch.pow((1-pr),gamma))*gt*pr_log)
    # score=torch.pow(-gt_sum*torch.log(pr_sum+eps),1)
    return (score)


def SSIM(img1, img2,weights ,val_range=2,sigma=1.5, window_size=15, window=None, size_average=True, full=False):
    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    _1d_window=(gauss/gauss.sum()).unsqueeze(1)

    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    # L=val_range
    pad=window_size//2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    window = torch.Tensor(_2d_window.expand(channels, 1, window_size, window_size).contiguous())
    window=window.to(img1.get_device())

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq =mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
    for i in range(5):

        ssim_score[:,i,:,:]=ssim_score[:,i,:,:]*weights[i]
    return(ssim_score.mean())

def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score

def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score

def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


def precision_recall(pr, gt, eps=float(1e-7), eval = False, threshold = None, ignore_channels=None, kalman_flag=False):
    '''
    The better written variant of precision_recall function
    @brief Calculates recall and precision pixelwise 
    @param gt: ground truth
    @param pr: predicted mask

    @return: recall = tp/(tp+fn), precision = tp/(tp+fp)
    '''

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    precision = torch.tensor(0.).to('cuda')
    recall = torch.tensor(0.).to('cuda')

    # Remove all the zero ground truth entries
    if kalman_flag:
        gt_sum = gt.sum(dim=[-2, -1])
        pr_sum = pr.sum(dim=[-2,-1])
        inds = gt_sum.nonzero()
        if len(inds) > 0:
            gt = gt[inds[:, 0], inds[:, 1],: , :, :]
            pr = pr[inds[:, 0], inds[:, 1],: , :, :]  
        elif eval: #and len(pr_sum.nonzero()) == 0:
            return None, None      

    else:
        gt_sum = gt.sum(dim=[-2, -1])
        pr_sum = pr.sum(dim=[-2,-1])
        inds = gt_sum.nonzero()
        if len(inds) > 0:
            gt = gt[inds[:, 0],: , :, :]
            pr = pr[inds[:, 0],: , :, :]
        elif eval: #and len(pr_sum.nonzero()) == 0:
            return None, None

    
    tp = torch.sum(gt * pr, dim=[-2, -1])
    fp = torch.sum(pr, dim=[-2, -1]) - tp
    fn = torch.sum(gt, dim=[-2, -1]) - tp
    precision = torch.mean((tp ) / (tp + fp + eps))
    recall = torch.mean((tp ) / (tp + fn + eps))

    return precision, recall