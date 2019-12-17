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

from model import * 
from data import *
from config import *


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

def train_gen(no, EPOCH, loader, gen, gen_optimizer, opt, train_with_dis = False):
    loss = nn.MSELoss()
    criterion = nn.BCELoss()

    for e in range(1, EPOCH + 1):
        opt.gen_epoch = opt.gen_epoch+1
        strat_time = time.time()
        epoch_loss = 0
        ma_loss = 0
        ms_loss = 0
        p_loss = 0

        gen.train()
        print("Generator EPOCH %d"%(e))
        for step, (batch_data, batch_label) in enumerate(loader):
            batch_data, batch_mask, batch_label = load(batch_data)
            batch_data = handle_data(batch_data, batch_mask)

            cur_size = batch_data.size(0)

            batch_data = batch_data[:, : 3]

            if opt.att_gen==True:
                mask_list, data_frame1, data_frame2, data_frame3 = gen(batch_data, opt)
            else:
                data_frame1, data_frame2, data_frame3 = gen(batch_data)

            data_frame4 = vgg(data_frame3)
            label_frame1, label_frame2, label_frame3, label_frame4 = handle_label(batch_label)

            multi_scale_loss = 0.6*loss(data_frame1, label_frame1) + 0.8 * loss(data_frame2, label_frame2) + loss(data_frame3, label_frame3)
            perceptual_loss = loss(data_frame4[0], label_frame4[0]) + 0.6*loss(data_frame4[1], label_frame4[1]) + 0.4*loss(data_frame4[2], label_frame4[2]) + 0.2*loss(data_frame4[3], label_frame4[3])

            mask_loss = 0
            if opt.att_gen==True:
                real_mask = batch_data[:, -1]
                for i in range(opt.ITERATION):
                    mask_loss += opt.rate**(opt.ITERATION - i)*loss(mask_list[i][:,0,:,:], real_mask)

            total_loss = multi_scale_loss + perceptual_loss + mask_loss

            if train_with_dis:
                trainable(dis, False)
                result = dis(data_frame3)[-1]
                result.requires_grad=True

                label_fake = Variable(torch.ones((cur_size, 1))).cuda()
                gan_loss = criterion(result, label_fake)

                total_loss += 0.01 * gan_loss

            epoch_loss += total_loss
            ma_loss += mask_loss
            ms_loss += multi_scale_loss
            p_loss += perceptual_loss

            gen_optimizer.zero_grad()
            total_loss.backward()
            gen_optimizer.step()

        print('total loss is %.5f'%(epoch_loss))
        print('scale_loss = %.5f perceptual_loss = %.5f mask_loss = %.5f'%(ms_loss, p_loss, ma_loss))

        end_time = time.time()
        print("using time %.0fs"%(end_time - strat_time))

        save(opt.gen_epoch, gen, opt.photo_num, opt.model_save_path, opt.eval_input_path, opt.eval_gt_path, opt.eval_save_path, opt.postfix+'_gen', opt)

    return gen, gen_optimizer, opt

def train_dis(EPOCH, loader, dis, dis_optimizer, opt):
    loss = nn.MSELoss()
    criterion = nn.BCELoss()
    trainable(gen, False)
    trainable(dis, True)

    for e in range(1, EPOCH + 1):
        opt.dis_epoch = opt.dis_epoch+1

        strat_time = time.time()
        class_loss_real = 0
        mse_loss_real = 0
        class_loss_fake = 0
        mse_loss_fake = 0
        acc = 0
        data_size = 0
        print("Discriminator EPOCH %d"%(e))

        for step, (batch_data, batch_label) in enumerate(loader):
            data_size += batch_data.size(0)
            cur_size = batch_data.size(0)
            batch_data, batch_mask, batch_label = load(batch_data)
            batch_fake = handle_data(batch_data, batch_mask)[:, : 3]
            batch_real = Variable(handle_label(batch_label, False)).cuda()

            if opt.att_gen==True:
                batch_fake = gen(batch_fake, opt)[-1]
            else:
                batch_fake = gen(batch_fake)[-1]

            result_real = dis(batch_real)
            result_fake = dis(batch_fake)

            dis_optimizer.zero_grad()
            label_real = torch.ones(cur_size, 1).cuda()
            result_real = dis(batch_real)

            real_loss = criterion(result_real, label_real)
            class_loss_real += real_loss

            real_loss.backward()

            label_fake = Variable(torch.zeros((cur_size, 1))).cuda()
            result_fake = dis(batch_fake)

            fake_loss = criterion(result_fake, label_fake)
            class_loss_fake += fake_loss
            fake_loss.backward()

            dis_optimizer.step()

            result_real = np.round(result_real.cpu().data.numpy().squeeze())
            result_fake = np.round(result_fake.cpu().data.numpy().squeeze())
            acc += np.sum(result_real)
            acc += np.sum(1 - result_fake)

        print("Acc rate is %.5f"%(acc / (data_size*2)))
        print('real bce loss is %.4f, mse loss is %.4f'%(class_loss_real, mse_loss_real))
        print('fake bce loss is %.4f, mse loss is %.4f'%(class_loss_fake, mse_loss_fake))

        """if e > 10 * interval and e % interval == interval - 1:
            if min(loss_list[: e - interval + 1]) - min(loss_list[e - interval + 1 : e + 1]) < threshold * min(loss_list[: e - interval + 1]):
                LR *= 0.9
                dis_optimizer = torch.optim.SGD(dis.parameters(), lr=LR)"""
        end_time = time.time()
        print("using time %.0fs"%(end_time - strat_time))
    return dis, dis_optimizer, opt
        
def train_ad(gen, gen_optimizer, dis, dis_optimizer, opt):
    loss = nn.MSELoss()
    criterion = nn.BCELoss()
    for e in range(1, opt.ITERATION_GAN + 1):
        opt.gen_epoch = opt.gen_epoch + 1
        opt.dis_epoch = opt.dis_epoch + 1
        strat_time = time.time()
        print("Adversarial EPOCH %d"%(e))
        it = 0
        acc = 0
        for step, (batch_data, batch_label) in enumerate(loader):
            it += 1
            trainable(gen, False)
            trainable(dis, True)
            class_loss_real = 0
            mse_loss_real = 0
            class_loss_fake = 0
            mse_loss_fake = 0
            data_size = 0
            data_size += batch_data.size(0)
            cur_size = batch_data.size(0)
            batch_data, batch_mask, batch_label = load(batch_data)
            batch_data = handle_data(batch_data, batch_mask)

            batch_fake = batch_data[:, : 3]
            batch_real = Variable(handle_label(batch_label, False)).cuda()

            if opt.att_gen==True:
                batch_fake = gen(batch_fake, opt)[-1]
            else:
                batch_fake = gen(batch_fake)[-1]

            dis_optimizer.zero_grad()

            label_real = Variable(torch.ones((cur_size, 1))).cuda()
            result_real = dis(batch_real)

            real_loss = criterion(result_real, label_real)
            class_loss_real = real_loss
            real_loss.backward()

            label_fake = Variable(torch.zeros((cur_size, 1))).cuda()

            result_fake = dis(batch_fake)

            fake_loss = criterion(result_fake, label_fake)
            class_loss_fake = fake_loss
            fake_loss.backward()

            dis_optimizer.step()

            result_acc1 = result_real.cpu().data.numpy().squeeze()
            result_real = np.round(result_acc1)

            result_acc2 = result_fake.cpu().data.numpy().squeeze()
            result_fake = np.round(result_acc2)

            acc += np.sum(result_real)
            acc += np.sum(1 - result_fake)

            '''
            print('Iteration %d:'%(it))
            print('(Discriminator) acc rate is %.5f'%(acc / (data_size*2)))
            print('result real =')
            print(result_acc1, result_real)
            print('result fake =')
            print(result_acc2, result_fake)
            print('real bce loss is %.4f, mse loss is %.4f'%(class_loss_real, mse_loss_real))
            print('fake bce loss is %.4f, mse loss is %.4f'%(class_loss_fake, mse_loss_fake))
            '''

            trainable(gen, True)
            trainable(dis, False)
            epoch_loss = 0
            ma_loss =  0
            ms_loss = 0
            p_loss = 0

            if opt.att_gen==True:
                mask_list, data_frame1, data_frame2, data_frame3 = gen(batch_data[:, : 3], opt)
                real_mask = batch_data[:, -1]
            else:
                data_frame1, data_frame2, data_frame3 = gen(batch_data[:, :3])



            data_frame4 = vgg(data_frame3)
            label_frame1, label_frame2, label_frame3, label_frame4 = handle_label(batch_label)

            multi_scale_loss = 0.6*loss(data_frame1, label_frame1) + 0.8 * loss(data_frame2, label_frame2) + loss(data_frame3, label_frame3)
            perceptual_loss = loss(data_frame4[0], label_frame4[0]) + 0.6*loss(data_frame4[1], label_frame4[1]) + 0.4*loss(data_frame4[2], label_frame4[2]) + 0.2*loss(data_frame4[3], label_frame4[3])

            mask_loss = 0
            if opt.att_gen==True:
                for i in range(opt.ITERATION):
                    mask_loss += opt.rate**(opt.ITERATION - i)*loss(mask_list[i][:,0,:,:], real_mask)

            total_loss = multi_scale_loss + perceptual_loss + mask_loss

            #result = dis(data_frame3)[-1]
            result = dis(data_frame3)
            #result.requires_grad=True
            label_fake = Variable(torch.ones((cur_size, 1))).cuda()
            gan_loss = criterion(result, label_fake)

            total_loss += 0.01 * gan_loss
            epoch_loss += total_loss
            ma_loss += mask_loss
            ms_loss += multi_scale_loss
            p_loss += perceptual_loss

            gen_optimizer.zero_grad()
            total_loss.backward()
            gen_optimizer.step()
            #print('(Generator)')

        print('(Discriminator) Accuracy = %.4f'%(acc*1./(opt.photo_num*2)))
        save(opt.gen_epoch, gen, opt.photo_num, opt.model_save_path, opt.eval_input_path, opt.eval_gt_path, opt.eval_save_path, opt.postfix+'_dis', opt)
        end_time = time.time()
        print("using time %.0fs"%(end_time - strat_time))

    return gen, gen_optimizer, dis, dis_optimizer, opt


def predict(gen, input_path, output_path):
    #input_path = "/home/whyang/pytorch_project/raindrop_data/rain/"
    #output_path = "/home/whyang/pytorch_project/raindrop/result_wa/"
    file_list = os.popen("ls " + input_path).read().split("\n")

    for file_name in file_list:
        if file_name == "":
            continue
        print(input_path + file_name)
        img = cv2.imread(input_path + file_name)
        img = (img / 255.).astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

        img = Variable(torch.from_numpy(img)).cuda()
        result = gen(img)[-1]
        result = result.cpu().numpy()
        result = result.transpose((1, 2, 0))
        cv2.imwrite(output_path + file_name, result * 255.0)


if __name__ == '__main__': 
    
    vgg_model = vgg_init(opt.vgg_loc)
    vgg = vgg(vgg_model)
    vgg.eval()

    ensure_exists(opt.model_save_path)
    ensure_exists(opt.eval_save_path)

    if opt.att_gen==True:
        gen = AttGenerator().cuda()
    else:
        gen = Generator().cuda()

    if opt.att_dis==True:
        dis = AttDiscriminator().cuda()
    else:
        dis = Discriminator().cuda()

    gen_optimizer = torch.optim.RMSprop(gen.parameters(), lr=opt.LR_g)
    dis_optimizer = torch.optim.RMSprop(dis.parameters(), lr=opt.LR_d)

    list = []
    for i in range(opt.CROP_TIME):
        list.extend(range(opt.photo_num))

    tensor = torch.IntTensor(list)
    dataset = Data.TensorDataset(tensor, tensor)

    loader = Data.DataLoader(dataset = dataset, batch_size = opt.BATCH_SIZE, shuffle = True)

    opt.gen_epoch = 0
    opt.dis_epoch = 0

    gen, gen_optimizer, opt = train_gen(-1, opt.EPOCH_GEN_PRE, loader, gen, gen_optimizer, opt)
    dis, dis_optimizer, opt = train_dis(opt.EPOCH_DIS_PRE, creat_loader(opt.photo_num * opt.CROP_TIME // 3, opt.BATCH_SIZE // 2, gen, opt.photo_num), dis, dis_optimizer, opt)
    gen, gen_optimizer, dis, dis_optimizer, opt = train_ad(gen, gen_optimizer, dis, dis_optimizer, opt)

