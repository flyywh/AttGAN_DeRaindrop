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
import argparse
import os

ITERATION_GAN = 0
EPOCH_PRE = 40
EPOCH_GEN = 2
EPOCH_DIS = 2            
BATCH_SIZE = 6
LOCAL_SIZE = 64
CROP_TIME = 10
photo_num = 700
LR = 0.0001
ITERATION = 4
rate = 0.5      

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

    def forward(self, input_pic, input_mask = 0):
        batch_size, row, col = input_pic.size(0), input_pic.size(2), input_pic.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).cuda() / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda() 
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        mask_list = []
        for i in range(ITERATION):
            x = torch.cat((input_pic, mask), 1)
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
            c = f * c + i * g
            h = o * F.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)
        
        if isinstance(input_mask, int):
            x = torch.cat((input_pic, mask), 1)
        else:
            x = torch.cat((input_pic, input_mask), 1)
            print("concat with input mask")
        
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
        return mask_list, frame1, frame2, res1, x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
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
            nn.Linear(1024, 1)
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

def YCR_CB2BGR(Y, CR, CB):
    photo_num = Y.data.size(0)
    row, col = Y.data.size(2), Y.data.size(3)
    result = Variable(torch.FloatTensor(photo_num, 3, row, col)).cuda()
    result[:, 0] = 1.164 * (Y - 16 / 255.0) + 2.017 * (CB - 128 / 255.0)
    result[:, 1] = 1.164 * (Y - 16 / 255.0) - 0.392 * (CB - 128 / 255.0) - 0.813 * (CR - 128 / 255.0)
    result[:, 2] = 1.164 * (Y - 16 / 255.0) + 1.596 * (CR - 128 / 255.0)
    return result 

def trainable(net, trainable):
	for para in net.parameters():
		para.requires_grad = trainable

def vgg_init():
    vgg_model = torchvision.models.vgg16(pretrained = False).cuda()
    vgg_model.load_state_dict(torch.load('/home/qianrui/.torch/models/models/vgg16-397923af.pth'))
    trainable(vgg_model, False)
    return vgg_model

def vgg(img):
    global vgg_model
    x = img
    for idx, layer in enumerate(vgg_model.modules()):
        if idx >= 2 and idx <= 17:
            x = layer(x)
    return x

def load(data_list, crop = True, path = '/home/qianrui/aligned_data/'):
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

"""def load_data(data_list):
    photo_num = data_list.size(0)
    size = 224  
    data = torch.FloatTensor(photo_num, 2, size, size)
    CR = torch.FloatTensor(photo_num, 1, size, size)
    CB = torch.FloatTensor(photo_num, 1, size, size)
    for i, no in enumerate(data_list):
        data_path = '/home/qianrui/data/rain_mid/%d.jpg'%(no) 
        mask_path = '/home/qianrui/data/mask_mid/%d.jpg'%(no) 
        #data_path = './rain_mid/%d.jpg'%(no) 
        #mask_path = './mask_mid/%d.jpg'%(no) 
        temp_data = cv2.imread(data_path)
        temp_data = cv2.cvtColor(temp_data, cv2.COLOR_BGR2YCR_CB) / 255.
        temp_mask = cv2.imread(mask_path)
        data[i, 0, :, :] = torch.from_numpy(temp_mask[:, :, 0])
        data[i, 1, :, :] = torch.from_numpy(temp_data[:, :, 0])
        CR[i, :, :] = torch.from_numpy(temp_data[:, :, 1])
        CB[i, :, :] = torch.from_numpy(temp_data[:, :, 2])
    return Variable(data).cuda(), Variable(CR).cuda(), Variable(CB).cuda()"""

def handle_data(data_list, mask_list):
    batch_size, row, col = data_list.shape[0 : 3]
    data = torch.FloatTensor(batch_size, 4, row, col)
    #CR = torch.FloatTensor(data_size, 1, row, col)
    #CB = torch.FloatTensor(data_size, 1, row, col)
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
    trainable(gen, True)
    loss = nn.MSELoss()   
    for e in range(1, EPOCH + 1):
        strat_time = time.time()
        epoch_loss = 0
        print("(Generator)EPOCH %d"%(e))
        for step, (batch_data, batch_label) in enumerate(loader): 
            batch_data, batch_mask, batch_label = load(batch_data)
            batch_data = handle_data(batch_data, batch_mask)
            real_mask = batch_data[:, -1]
            batch_data = batch_data[:, : 3]
            mask_list, data_frame1, data_frame2, data_frame3 = gen(batch_data)
            data_frame4 = vgg(data_frame3)
            label_frame1, label_frame2, label_frame3, label_frame4 = handle_label(batch_label)
            total_loss = loss(data_frame1, label_frame1) + 0.8 * loss(data_frame2, label_frame2) 
            total_loss += 1.2 * loss(data_frame3, label_frame3) + 1.2 * loss(data_frame4, label_frame4)
            for mask in mask_list:              
                total_loss = total_loss * rate + loss(mask, real_mask)
            if train_with_dis:
                trainable(dis, False)
                _, result = dis(data_frame3)
                gan_loss = 0.5 * ((1. - result) * (1. - result)).sum()
                total_loss += gan_loss
            epoch_loss += total_loss.data[0]
            gen_optimizer.zero_grad()  
            total_loss.backward()          
            gen_optimizer.step() 
        print("total loss is %.5f"%(epoch_loss))
        end_time = time.time()
        print("using time %.0fs"%(end_time - strat_time))
        if e % 5 == 0:
            save(e)

def train_dis(EPOCH, loader):
    global dis, dis_optimizer
    #LR, interval, threshold = 0.0001, 6, 0.0004
    loss = nn.MSELoss()
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
            batch_fake = handle_data(batch_data, batch_mask)
            batch_real = Variable(handle_label(batch_label, False)).cuda()
            _, __, batch_fake = gen(batch_fake)
            mask_real, result_real = dis(batch_real)
            mask_fake, result_fake = dis(batch_fake)
            gt_mask_fake, gt_mask_real = handle_mask(batch_mask)
            total_loss = ((1. - result_real) * (1. - result_real)).sum() + (result_fake * result_fake).sum()
            epoch_loss += (1. - result_real + result_fake).sum().data[0]
            total_loss += loss(mask_fake, gt_mask_fake) + loss(mask_real, gt_mask_real)
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
    global photo_num
    #gen.eval()
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

def save(epoch_no = -1):
    global gen, dis, photo_num
    trainable(gen, False)
    row, col = 480, 720
    path = "./result4/"
    torch.save(gen.state_dict(), path + "gen.pkl") 
    torch.save(dis.state_dict(), path + "dis.pkl") 
    comp_img = torch.FloatTensor(3, row, col)
    epoch_str = ""
    if epoch_no > -1:
        epoch_str = "iter" + str(epoch_no) + '_'
    #watch_list = [2, 8, 40, 62, 64, 79, 88, 118]
    #watch_list = range(7)
    for i in range(700, 794):
        img = torch.FloatTensor(row, 4 * col, 3)
        batch_data, batch_mask, batch_label = load(torch.IntTensor([i]), False, '/home/qianrui/predict_data/')
        temp_data = handle_data(batch_data, batch_mask)
        label = handle_label(batch_label, False)[0]
        result = gen(temp_data[:, : 3])[-1]
        result = result.cpu().data[0]
        mask = temp_data.cpu().data[0, 3]
        comp_img = mask * result + (1 - mask) * label
        for j in range(3):
            img[:, 0 : col, j] = label[j, :, :]
            img[:, col : 2 * col, j] = torch.from_numpy(batch_data[0, :, :, j] / 255.)
            img[:, 2 * col : 3 * col, j] = result[j, :, :]
            img[:, 3 * col : 4 * col, j] = comp_img[j, :, :]
        img = img.numpy()
        cv2.imwrite(path + epoch_str + str(i) + ".jpg", img * 255.0)

def predict():
    global gen1, gen2, input_path, output_path
    trainable(gen1, False)
    trainable(gen2, False)
    """input_path = '/home/qianrui/pytorch/gan/input/'
    output_path = '/home/qianrui/pytorch/gan/output/'"""
    #file_list = os.popen("ls " + input_path).read().split("\n")
    #print(file_list)
    file_list = os.listdir(input_path)
    file_list = sorted(file_list)
    file_num = len(file_list)
    for i in range(file_num):
        #img = cv2.imread(input_path + file_list[i])
        img = cv2.imread(input_path + '778_rain.jpg')
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
        #print(img.shape)
        img = Variable(torch.from_numpy(img)).cuda()
        #result_list = gen(img)
        #result = result_list[-1]
        #result_mask = result_list[0][-1]

        if input_mask:
            """mask = np.load(mask_path + "mask.npy")
            mask[:, :] = 0
            mask = mask[0 : a_row, 0 : a_col]
            mask = mask.reshape(1, 1, a_row, a_col)"""
            mask = np.zeros((1, 1, a_row, a_col), dtype = 'float32')
            mask[:, :, :, :] = 255 
            mask = Variable(torch.from_numpy(mask)).cuda()
            print(mask.size())
            result = gen(img, mask)[-1]
        else:
            result1 = gen1(img)[-1]
            fea1 = gen1(img)[-2]
            result2 = gen2(img)[-1]
            fea2 = gen2(img)[-2]

        result_mask_list1 = gen1(img)[0]
        result1 = result1.cpu().data[0].numpy()
        result1 = result1.transpose((1, 2, 0))
        name = file_list[i].split(".")[0]

        fea1 = fea1.cpu().data[0].numpy()
        fea1 = fea1.transpose((1, 2, 0))
        np.save('./feature/conv1_with_a.npy', fea1)

        result_mask_list2 = gen2(img)[0]
        result2 = result2.cpu().data[0].numpy()
        result2 = result2.transpose((1, 2, 0))

        fea2 = fea2.cpu().data[0].numpy()
        fea2 = fea2.transpose((1, 2, 0))
        np.save('./feature/conv1_wo_a.npy', fea2)
        result = np.concatenate((result1, result2, abs(result1 - result2)), 1)
        c = (result1 - result2)*255.
        print ('max = %.4f min = %.4f mean = %.4f'%(np.max(c), np.min(c), np.mean(c)))

        cv2.imwrite(output_path + name + '.png', result * 255.0)
        raw_input()
        '''
        for result_mask in result_mask_list:
            print ('fuckfuck')
            result_mask = result_mask.cpu().data[0].numpy()
            result_mask = result_mask.transpose((1, 2, 0))
            result_mask = np.array(result_mask * 255.0, dtype='uint8')
            result_mask = cv2.applyColorMap(result_mask, cv2.COLORMAP_JET)
            #result_mask = np.concatenate((result_mask, result_mask, result_mask), axis=2)
            img = np.concatenate((img, result_mask), 1)
        cv2.imwrite(output_path + name + '_mask.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        '''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type = str, default = './input/')
    parser.add_argument("--output_path", type = str, default = './output/')
    parser.add_argument("--input_mask", type = bool, default = False)
    parser.add_argument("--mask_path", type = str, default = './mask/')
    args = parser.parse_args()
    return args


args = get_args()
input_path = args.input_path
output_path = args.output_path
input_mask = args.input_mask
mask_path = args.mask_path

vgg_model = vgg_init()
gen1 = Generator().cuda()
gen2 = Generator().cuda()
gen1.load_state_dict(torch.load('/mnt/hdd/qianrui/derain/weights/cmp/w.pkl'))
gen2.load_state_dict(torch.load('/mnt/hdd/qianrui/derain/weights/cmp/wo.pkl'))
predict()
