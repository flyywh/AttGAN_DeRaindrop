
class item:
    def __init__(self):
        self.name = '' 

opt = item()

opt.ITERATION_GAN = 400
#opt.EPOCH_GEN_PRE = 30
opt.EPOCH_GEN_PRE = 2
#opt.EPOCH_DIS_PRE = 20
opt.EPOCH_DIS_PRE = 2

opt.EPOCH_GEN = 1
opt.EPOCH_DIS = 1
opt.BATCH_SIZE = 2
opt.LOCAL_SIZE = 128
opt.CROP_TIME = 2
opt.photo_num = 700
opt.LR_g = 0.00002
opt.LR_d = 0.00001
opt.ITERATION = 4
opt.rate = 0.6

opt.model_save_path = './weights_gen_att_dis_att/'
opt.eval_input_path = '../raindrop_data/input/'
opt.eval_gt_path = '../raindrop_data/gt/'
opt.eval_save_path = './results_gen_att_dis_att/'
opt.vgg_loc = './models/vgg16-397923af.pth'
opt.postfix = 'gen_att_dis_att'

opt.att_gen = True
opt.att_dis = True
