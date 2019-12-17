
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
import random
from torch.autograd import Variable
from util import *

def handle_data(data_list, mask_list):
    batch_size, row, col = data_list.shape[0 : 3]
    data = torch.FloatTensor(batch_size, 4, row, col)
    for i in range(3):
        data[:, i, :, :] = torch.from_numpy(data_list[:, :, :, i]) / 255.

    tmp_mask = torch.from_numpy(mask_list)
    data[:, 3, :, :] = tmp_mask[:, :, :, 0]

    return Variable(data).cuda()

def handle_mask(mask_list):
    mask_size, row, col = mask_list.shape[0 : 3]
    mask = torch.FloatTensor(mask_size, 1, row, col)
    mask0 = torch.zeros((mask_size, 1, row, col))
    for i, temp_mask in enumerate(mask_list):
        mask[i, 0] = torch.from_numpy(temp_mask[:, :, 0])

    return Variable(mask).cuda(), Variable(mask0).cuda()

def handle_label(label_list, need_vgg = True):
    label_size, row, col = label_list.shape[0 : 3]
    label = torch.FloatTensor(label_size, 3, row, col)
    label_2 = torch.FloatTensor(label_size, 3, row // 2, col // 2)
    label_4 = torch.FloatTensor(label_size, 3, row // 4, col // 4)
    label_vgg = torch.FloatTensor(label_size, 256, row // 4, col // 4)
    for i, temp_label in enumerate(label_list):
        temp_label = temp_label / 255.
        temp_label_2 = cv2.resize(temp_label, (row // 2, col // 2))
        temp_label_4 = cv2.resize(temp_label, (row // 4, col // 4))
        for j in range(3):
            label[i, j] = torch.from_numpy(temp_label[:, :, j])
            label_2[i, j] = torch.from_numpy(temp_label_2[:, :, j])
            label_4[i, j] = torch.from_numpy(temp_label_4[:, :, j])
    if need_vgg:
        label = Variable(label).cuda()
        label_vgg = vgg(label)
        return Variable(label_4).cuda(), Variable(label_2).cuda(), label, label_vgg
    return label

def load(data_list, crop = True, path = '../raindrop_data/'):
    batch_size = data_list.size(0)
    row, col = 480, 720
    data = np.zeros((batch_size, row, col, 3), dtype = np.float32)
    mask = np.zeros((batch_size, row, col, 1), dtype = np.float32)
    label = np.zeros((batch_size, row, col, 3), dtype = np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))
    for i, no in enumerate(data_list):
        data_path = path + 'rain/%d_rain.jpg'%(no)
        mask_path = path + 'mask/%d.npy'%(no) 
        label_path = path + 'clean/%d_clean.jpg'%(no)

        temp_data = cv2.imread(data_path)
        temp_data = temp_data[0 : row, 0 : col]
        temp_mask = np.load(mask_path)
        temp_mask = temp_mask[0 : row, 0 : col]
        temp_mask = cv2.dilate(temp_mask, kernel)
        temp_label = cv2.imread(label_path)
        temp_label = temp_label[0 : row, 0 : col]
        data[i] = temp_data
        mask[i, :, :, 0] = temp_mask
        label[i] = temp_label

    if crop:
        size = 224
        crop_data = np.zeros((batch_size, size, size, 3), dtype = np.float32)
        crop_mask = np.zeros((batch_size, size, size, 1), dtype = np.float32)
        crop_label = np.zeros((batch_size, size, size, 3), dtype = np.float32)
        for i in range(data_list.size(0)):
            x, y = random.randint(0, row - size), random.randint(0, col - size)
            crop_data[i] = data[i, x : x + size, y : y + size]
            crop_mask[i] = mask[i, x : x + size, y : y + size]
            crop_label[i] = label[i, x : x + size, y : y + size]
        return crop_data, crop_mask, crop_label
    return data, mask, label

def creat_loader(total_size, batch_size, gen, photo_num):
    trainable(gen, False)
    x = torch.IntTensor(total_size)
    for i in range(total_size):
        num = random.randint(0, photo_num - 1)
        x[i] = num
    dataset = Data.TensorDataset(x, x)
    loader = Data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
    )
    return loader


def save(epoch_no, gen, photo_num, weight_path, data_path, gt_path, save_path, postfix, opt):
    gen.eval()
    torch.save(gen.state_dict(), weight_path + '/epoch=%d_weights_' %(epoch_no) + postfix+'.pkl')

    predict_list = os.listdir(data_path)
    lis = os.listdir(data_path)
    predict_num = 94

    psnr = 0
    ssim = 0

    for i in range(700, 794):
        img = cv2.imread(data_path + '%d.jpg'%(i))
        a_row = np.floor(img.shape[0]/4)*4
        a_col = np.floor(img.shape[1]/4)*4

        img = img[0:int(a_row), 0:int(a_col)]
        img_o = img
        img = (img / 255.).astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = Variable(torch.from_numpy(img)).cuda()

        if opt.att_gen==True:
            result1 = gen(img, opt)[-1]
        else:
            result1 = gen(img)[-1]

        result1 = result1.cpu().data[0].numpy()
        result1 = result1.transpose((1, 2, 0))
        result1 = result1*255.

        result = np.concatenate((img_o, result1), axis=1)
        gt = cv2.imread(gt_path + '%d.jpg'%(i))
        gt = gt[0:int(a_row), 0:int(a_col)]

        result1 = np.array(result1, dtype ='uint8')
        psnr += skimage.measure.compare_psnr(result1, gt)
        ssim += skimage.measure.compare_ssim(result1, gt, multichannel = True)

        cv2.imwrite((save_path + 'epoch_%d_num_%d_'%(epoch_no, i)+postfix+'.png'), result)

    psnr = psnr/94.
    ssim = ssim/94.

    print('PSNR = %.4f SSIM = %.4f'%(psnr, ssim))
