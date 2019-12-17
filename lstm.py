import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import cv2
import random
import time
import os
import skimage
from skimage import measure

ITERATION_GAN = 40
EPOCH_GEN_PRE = 400
EPOCH_DIS_PRE = 6
EPOCH_GEN = 1
EPOCH_DIS = 1            
BATCH_SIZE = 3
LOCAL_SIZE = 128
CROP_TIME = 2
photo_num = 700
LR = 0.00005
ITERATION = 4
rate = 0.6      

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            #nn.Conv2d(64, 1, 3, 1, 1),
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2),
            #nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            #nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation = 2),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation = 4),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation = 8),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation = 16),
            #nn.BatchNorm2d(256)
            nn.ReLU()
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            #nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            #nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            #nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            #nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            #nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            #nn.Sigmoid()
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).cuda() / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda() 
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        mask_list = []
        cell_list = []
        for i in range(ITERATION):
            x = torch.cat((input, mask), 1)
            x = self.det_conv0(x)
            resx = x
            x = F.relu(self.det_conv1(x) + resx)
            resx = x
            x = F.relu(self.det_conv2(x) + resx)
            resx = x
            x = F.relu(self.det_conv3(x) + resx)
            resx = x
            x = F.relu(self.det_conv4(x) + resx)
            resx = x
            x = F.relu(self.det_conv5(x) + resx)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            
            cell_list.append(c)
            c = f * c + i * g
            h = o * F.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return mask_list, frame1, frame2, cell_list, x

class AttDiscriminator(nn.Module):
    def __init__(self):
        super(AttDiscriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(128, 1, 5, 1, 2)
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 5, 4, 1),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 4, 1),
            nn.ReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        return mask, self.fc(x)

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable

def vgg_init():
    vgg_model = torchvision.models.vgg16(pretrained = False).cuda()
    vgg_model.load_state_dict(torch.load('/home/qianrui/.torch/models/models/vgg16-397923af.pth'))
    trainable(vgg_model, False)
    return vgg_model

class vgg(nn.Module):
    def __init__(self, vgg_model):
        super(vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

def load(data_list, crop = True, path = '/home/qianrui/raindrop/'):
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

def handle_data(data_list, mask_list):
    batch_size, row, col = data_list.shape[0 : 3]
    data = torch.FloatTensor(batch_size, 4, row, col)
    for i in range(3):
        data[:, i, :, :] = torch.from_numpy(data_list[:, :, :, i]) / 255.
    data[:, 3, :, :] = torch.from_numpy(mask_list)
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


def train_gen(no, EPOCH, loader, train_with_dis = False):
    global gen, gen_optimizer
    loss = nn.MSELoss()   
    for e in range(1, EPOCH + 1):
        strat_time = time.time()
        epoch_loss = 0
        ma_loss = 0
        ms_loss = 0
        p_loss = 0

        gen.train()
        print("(Generator)EPOCH %d"%(e))
        for step, (batch_data, batch_label) in enumerate(loader): 
            batch_data, batch_mask, batch_label = load(batch_data)
            batch_data = handle_data(batch_data, batch_mask)
            real_mask = batch_data[:, -1]
            batch_data = batch_data[:, : 3]
            mask_list, data_frame1, data_frame2, data_frame3 = gen(batch_data)
            data_frame4 = vgg(data_frame3)
            label_frame1, label_frame2, label_frame3, label_frame4 = handle_label(batch_label)

            multi_scale_loss = 0.6*loss(data_frame1, label_frame1) + 0.8 * loss(data_frame2, label_frame2) + loss(data_frame3, label_frame3)
            perceptual_loss = loss(data_frame4[0], label_frame4[0]) + 0.6*loss(data_frame4[1], label_frame4[1]) + 0.4*loss(data_frame4[2], label_frame4[2]) + 0.2*loss(data_frame4[3], label_frame4[3])

            mask_loss = 0
            for i in range(ITERATION):
                mask_loss += rate**(ITERATION - i)*loss(mask_list[i], real_mask)

            total_loss = multi_scale_loss + perceptual_loss + mask_loss
            if train_with_dis:
                trainable(dis, False)
                result = dis(data_frame3)[-1]
                gan_loss = torch.log(1 - result + 0.001).mean()
                total_loss += 0.01 * gan_loss
            epoch_loss += total_loss.data[0]
            ma_loss += mask_loss.data[0]
            ms_loss += multi_scale_loss.data[0]
            p_loss += perceptual_loss.data[0]

            gen_optimizer.zero_grad()  
            total_loss.backward()          
            gen_optimizer.step() 
        print('total loss is %.5f'%(epoch_loss))
        print('scale_loss = %.5f perceptual_loss = %.5f mask_loss = %.5f'%(ms_loss, p_loss, ma_loss))
        end_time = time.time()
        print("using time %.0fs"%(end_time - strat_time))
        if e % 1 == 0:
            save(e)

def train_dis(EPOCH, loader):
    global dis, dis_optimizer
    #LR, interval, threshold = 0.0001, 6, 0.0004
    bi_loss = nn.BCELoss()
    trainable(gen, False)
    trainable(dis, True)
    #dis_optimizer = torch.optim.RMSprop(dis.parameters(), lr=LR)
    #loss_list = []
    for e in range(1, EPOCH + 1):
        strat_time = time.time()
        epoch_loss = 0
        data_size = 0
        print("(Discriminator)EPOCH %d"%(e))
        for step, (batch_data, batch_label) in enumerate(loader): 
            #center = load_center(batch_data).cuda()
            data_size += batch_data.size(0)
            batch_data, batch_mask, batch_label = load(batch_data)
            batch_fake = handle_data(batch_data, batch_mask)[:, : 3]
            batch_real = Variable(handle_label(batch_label, False)).cuda()
            batch_fake = gen(batch_fake)[-1]
            mask_real, result_real = dis(batch_real)
            mask_fake, result_fake = dis(batch_fake)
            gt_mask_fake, gt_mask_real = handle_mask(batch_mask)
            #total_loss = ((1. - result_real) * (1. - result_real)).sum() + (result_fake * result_fake).sum()
            total_loss = -torch.log(result_real + 0.001).mean() - torch.log(1 - result_fake + 0.001).mean()
            epoch_loss += (1. - result_real + result_fake).sum().data[0]
            total_loss += 0.05*(loss(mask_fake, gt_mask_fake) + loss(mask_real, gt_mask_real))
            #epoch_loss += total_loss.data[0]
            dis_optimizer.zero_grad()
            total_loss.backward()
            dis_optimizer.step()
        print("Error rate is %.5f"%(epoch_loss / data_size / 2))
        #loss_list.append(epoch_loss)
        """if e > 10 * interval and e % interval == interval - 1:
            if min(loss_list[: e - interval + 1]) - min(loss_list[e - interval + 1 : e + 1]) < threshold * min(loss_list[: e - interval + 1]):
                LR *= 0.9
                dis_optimizer = torch.optim.SGD(dis.parameters(), lr=LR)"""
        end_time = time.time()
        print("using time %.0fs"%(end_time - strat_time))

def creat_loader(total_size, batch_size):
    global gen, photo_num
    trainable(gen, False)
    x = torch.IntTensor(total_size)
    for i in range(total_size):
        num = random.randint(0, photo_num - 1)
        x[i] = num
    dataset = Data.TensorDataset(data_tensor = x, target_tensor = x)
    loader = Data.DataLoader(
        dataset = dataset,      
        batch_size = BATCH_SIZE,      
        shuffle = True,               
    )
    return loader

ignore_list = [710, 729, 736, 747, 755, 762, 777, 706, 713, 723, 726, 732, 743, 763, 782]

def save(epoch_no):
    global gen, photo_num
    gen.eval()
    #torch.save(gen.state_dict(), '/mnt/hdd/qianrui/derain/weights/epoch=%d_weights.pkl'%(epoch_no))
    
    data_path = '/home/qianrui/raindrop/input/'
    predict_list = os.listdir(data_path)
    lis = os.listdir(data_path)
    save_path = '/home/qianrui/raindrop/result/'
    gt_path = '/home/qianrui/raindrop/gt/'
    predict_num = 94

    psnr = 0
    ssim = 0
    p_y = 0
    s_y = 0

    for i in range(700, 794):
        if i in ignore_list:
            pass
        else:  
            img = cv2.imread(data_path + '%d.jpg'%(i))
            a_row = (img.shape[0]/4)*4
            a_col = (img.shape[1]/4)*4
            img = img[0:a_row, 0:a_col]
            img_o = img
            img = (img / 255.).astype(np.float32)
            img = img.transpose((2, 0, 1))
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            img = Variable(torch.from_numpy(img)).cuda()

            result1 = gen(img)[3]
            result1 = result1.cpu().data[0].numpy()
            result1 = result1.transpose((1, 2, 0))
            result1 = result1*255.

            result = np.concatenate((img_o, result1), axis=1)
            gt = cv2.imread(gt_path + '%d.jpg'%(i))
            gt = gt[0:a_row, 0:a_col]

            result1 = np.array(result1, dtype ='uint8')
            psnr1 = skimage.measure.compare_psnr(result1, gt)
            psnr += psnr1
            ssim1 = skimage.measure.compare_ssim(result1, gt, multichannel = True)
            ssim += ssim1
            result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2YCR_CB)[:,:,0]
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCR_CB)[:,:,0]
            p_y += skimage.measure.compare_psnr(result1, gt)
            s_y += skimage.measure.compare_ssim(result1, gt, multichannel = False)
            print('for image %d, psnr = %.4f, ssim = %.4f'%(i, psnr1, ssim1))

            cv2.imwrite((save_path + 'epoch_%d_num_%d.jpg'%(epoch_no, i)), result)

    psnr = psnr/(94. - len(ignore_list))
    ssim = ssim/(94. - len(ignore_list))
    p_y = p_y/(94. - len(ignore_list))
    s_y = s_y/(94. - len(ignore_list))

    print('PNSR = %.4f SSIM = %.4f'%(psnr, ssim))
    print('In y-channel PNSR = %.4f SSIM = %.4f'%(p_y, s_y))


def save2(epoch_no):
    global gen, photo_num
    gen.eval()
    #torch.save(gen.state_dict(), '/mnt/hdd/qianrui/derain/weights/epoch=%d_weights.pkl'%(epoch_no))
    
    data_path = '/home/qianrui/raindrop/valid_data/'
    predict_list= os.listdir(data_path)
    save_path = '/home/qianrui/raindrop/result/'
    gt_path = '/home/qianrui/raindrop/valid_gt/'
    predict_num = 30

    ignore_list = [3, 9, 16, 17, 19, 26, 20]

    psnr = 0
    ssim = 0
    p_y = 0
    s_y = 0

    for i in range(30):
        if i in ignore_list:
            pass
        else: 
            name = predict_list[i].split('.')[0]
            img = cv2.imread(data_path + predict_list[i])
            a_row = (img.shape[0]/4)*4
            a_col = (img.shape[1]/4)*4
            img = img[0:a_row, 0:a_col]
            img_o = img
            img = (img / 255.).astype(np.float32)
            img = img.transpose((2, 0, 1))
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            img = Variable(torch.from_numpy(img)).cuda()

            result1 = gen(img)[3]
            result1 = result1.cpu().data[0].numpy()
            result1 = result1.transpose((1, 2, 0))
            result1 = result1*255.

            result = np.concatenate((img_o, result1), axis=1)
            gt = cv2.imread(gt_path + name + '_GT.JPG')
            gt = gt[0:a_row, 0:a_col]

            result1 = np.array(result1, dtype ='uint8')
            psnr1 = skimage.measure.compare_psnr(result1, gt)
            psnr += psnr1
            ssim1 = skimage.measure.compare_ssim(result1, gt, multichannel = True)
            ssim += ssim1
            result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2YCR_CB)[:,:,0]
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCR_CB)[:,:,0]
            p_y += skimage.measure.compare_psnr(result1, gt)
            s_y += skimage.measure.compare_ssim(result1, gt, multichannel = False)
            print('for image %d, psnr = %.4f, ssim = %.4f'%(i, psnr1, ssim1))

            cv2.imwrite((save_path + 'epoch_%d_num_%d.jpg'%(epoch_no, i+100)), result)

    psnr = psnr/(30 - len(ignore_list))
    ssim = ssim/(30 - len(ignore_list))
    p_y = p_y/(30 - len(ignore_list))
    s_y = s_y/(30 - len(ignore_list))

    print('PNSR = %.4f SSIM = %.4f'%(psnr, ssim))
    print('In y-channel PNSR = %.4f SSIM = %.4f'%(p_y, s_y))

def save3():
    global gen, photo_num
    gen.eval()
    #torch.save(gen.state_dict(), '/mnt/hdd/qianrui/derain/weights/epoch=%d_weights.pkl'%(epoch_no))
    
    data_path = '/home/qianrui/raindrop/input/'
    predict_list= os.listdir(data_path)
    save_path = '/home/qianrui/raindrop/lstm/cell_state/'
    predict_num = len(predict_list)

    for i in range(707, 794):
        print (i)
        img = cv2.imread(data_path + '%d.jpg'%(i))
        a_row = (img.shape[0]/4)*4
        a_col = (img.shape[1]/4)*4
        img = img[0:a_row, 0:a_col]
        img_o = img
        img = (img / 255.).astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = Variable(torch.from_numpy(img)).cuda()

        cell_list = gen(img)[-2]

        '''
        for j in range(4):
            #print (cell_list[j][0])
            a = cell_list[j][0].cpu().data[0].numpy()
            b = cell_list[j][1].cpu().data[0].numpy()
            b = a-b
            c = cell_list[j][2].cpu().data[0].numpy()
            a = a.transpose((1, 2, 0))
            b = b.transpose((1, 2, 0))
            c = c.transpose((1, 2, 0))
            d = np.concatenate((a, b, c), axis = 1)
            e = d[:,:,0]
            maxi = np.max(e)
            e = e*(255./maxi)
            e = np.array(e, dtype='uint8')
            e = cv2.applyColorMap(e, cv2.COLORMAP_JET)
            for k in range(1, 32):
               f = d[:,:,k]
               maxi = np.max(f)
               f = f*(255./maxi) 
               f = np.array(f, dtype='uint8')
               f = cv2.applyColorMap(f, cv2.COLORMAP_JET)
               e = np.concatenate((e, f), axis=0)
            cv2.imwrite((save_path + 'img_%d_it_%d.png'%(i, j)), e)
            if j == 0:
                a0 = a
            a0 = np.concatenate((a0, a), axis = 1)

        e = a0[:,:,0]
        maxi = np.max(e)
        e = e*(255./maxi)
        e = np.array(e, dtype='uint8')
        e = cv2.applyColorMap(e, cv2.COLORMAP_JET)
        for k in range(1, 32):
           f = a0[:,:,k]
           maxi = np.max(f)
           f = f*(255./maxi) 
           f = np.array(f, dtype='uint8')
           f = cv2.applyColorMap(f, cv2.COLORMAP_JET)
           e = np.concatenate((e, f), axis=0)
        cv2.imwrite((save_path + 'cell_%d.png'%(i)), e)
        '''
        a = cell_list[1].cpu().data[0].numpy()
        b = cell_list[2].cpu().data[0].numpy()
        c = cell_list[3].cpu().data[0].numpy()
        a = a.transpose((1, 2, 0))
        b = b.transpose((1, 2, 0))
        c = c.transpose((1, 2, 0))
        d = np.concatenate((a, b, c), axis = 0)
        e = d[:, :, 0]
        maxi = np.max(e)
        e = e*(255./maxi)
        e = np.array(e, dtype='uint8')
        e = cv2.applyColorMap(e, cv2.COLORMAP_JET)
        for k in range(0, 32):
           f = d[:,:,k]
           maxi = np.max(f)
           f = f*(255./maxi) 
           f = np.array(f, dtype='uint8')
           f = cv2.applyColorMap(f, cv2.COLORMAP_JET)
           cv2.imwrite((save_path + 'cell_%d_ch_%d.jpg'%(i, k)), f)

def predict():
    global gen
    input_path = "/home/qianrui/raindrop/input/"
    output_path = "/home/qianrui/raindrop/output/"
    file_list = os.listdir(input_path)
    file_list = sorted(file_list)
    file_num = len(file_list)
    for i in range(file_num):
        img = cv2.imread(input_path + file_list[i])
        print ('processing image: %s'%(file_list[i]))
        print ('row = %d, col = %d'%(img.shape[0], img.shape[1]))
        #align to four
        a_row = (img.shape[0]/4)*4
        a_col = (img.shape[1]/4)*4
        img = img[0:a_row, 0:a_col]
        print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
        #align to four
        img = (img / 255.).astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = Variable(torch.from_numpy(img)).cuda()
        result1 = gen(img)[-1]
        result1 = result1.cpu().data[0].numpy()
        result1 = result1.transpose((1, 2, 0))
        name = file_list[i].split(".")[0]
        cv2.imwrite(output_path + name + '.png', result1 * 255.0)



if __name__ == '__main__': 
    #vgg_model = vgg_init()
    #vgg = vgg(vgg_model)
    #vgg.eval()
    gen, dis = Generator().cuda(), Discriminator().cuda()
    gen_optimizer = torch.optim.RMSprop(gen.parameters(), lr=LR)
    dis_optimizer = torch.optim.RMSprop(dis.parameters(), lr=LR)

    list = []
    for i in range(CROP_TIME):
        list.extend(range(photo_num))
    tensor = torch.IntTensor(list)
    dataset = Data.TensorDataset(data_tensor = tensor, target_tensor = tensor)
    loader = Data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True)
    #train_gen(-1, EPOCH_GEN_PRE, loader)
    gen.load_state_dict(torch.load('/mnt/hdd/qianrui/derain/weights/pretrain.pkl'))
    save3()
'''
train_dis(EPOCH_DIS_PRE, creat_loader(photo_num * CROP_TIME // 3, BATCH_SIZE // 2))
for i in range(1, ITERATION_GAN + 1):
    print("ITERATION %d"%(i))
    #strat_time = time.time()
    train_dis(EPOCH_DIS, creat_loader(photo_num * CROP_TIME // 6, BATCH_SIZE // 2))
    train_gen(i, EPOCH_GEN, loader, True)
    save(i)
    #end_time = time.time()
    """if i % 400 == 399:
        print("Iteration %d : using time %.2fs"%(i, end_time - strat_time))
        save(i)"""
save()
'''
