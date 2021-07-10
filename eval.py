import torch
import time
import argparse
from model_backwarp import GridNet
from dataset import vimeo_dataset
from torch.utils.data import DataLoader
import os
from utils_test import predict, to_psnr,to_ssim_skimage
from test_dataloader import vimeo_test_dataset
from tensorboardX import SummaryWriter

# --- Parse hyper-parameters  train --- #
parser = argparse.ArgumentParser(description='context-aware')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=16, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=50, type=int)
parser.add_argument('--train_dataset', type=str, default='/mnt/HDD/vimeo_3_frame/compressed_viemo_3_frames/')
parser.add_argument('--train_dataset_imgs', type=str, default='/mnt/HDD/vimeo_3_frame/vimeo_triplet/sequences/')
parser.add_argument('--out_dir', type=str, default='./output_result')
parser.add_argument('--load_model', type=str, default=None)
# --- Parse hyper-parameters  test --- #
parser.add_argument('--test_data_dir', type=str, default='/mnt/HDD/vimeo_3_frame/compressed_viemo_3_frames/')
parser.add_argument('--test_input', type=str, default='/mnt/HDD/vimeo_3_frame/vimeo_triplet/sequences/')
parser.add_argument('--test_gt', type=str, default='/mnt/HDD/vimeo_3_frame/vimeo_triplet/sequences/')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=10,  type=int)

args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch= args.train_epoch
train_data_dir= args.train_dataset
# --- test --- #
test_input = args.test_input
test_gt = args.test_gt
predict_result= args.predict_result
test_data_dir= args.test_data_dir
test_batch_size=args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
result_dir = args.out_dir + '/picture'
ckpt_dir = args.out_dir + '/checkpoint'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
logfile = open(args.out_dir + '/log.txt', 'w')
logfile.write('batch_size: ' + str(train_batch_size) + '\n')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --- Define the network --- #
net = GridNet(134, 3)

# --- Build optimizer --- #
optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, betas=(0.9, 0.999))

# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

# --- Load training data --- #
# --- Load testing data --- #
test_dataset = vimeo_test_dataset(test_data_dir, test_input)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

criterion=torch.nn.L1Loss()

# --- Multi-GPU --- #
gpu_nums=torch.cuda.device_count()
net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load('./epoch9.pkl'))
writer = SummaryWriter()
# --- Strat training --- #
iteration = 0


with torch.no_grad():
    psnr_list = []
    ssim_list = []
    for batch_idx, (frame1, frame2, frame3, mask_flow) in enumerate(test_loader):
        frame1 = frame1.to(device)
        frame3 = frame3.to(device)
        gt = frame2.to(device)
        # print(frame1)
        mask_flow = mask_flow.to(device)
        
        frame_out = net(frame1, frame3,1)
        Final_prediction = frame_out*(1-mask_flow) + frame1*mask_flow
        psnr_list.extend(to_psnr(Final_prediction, gt))
        ssim_list.extend(to_ssim_skimage(Final_prediction, gt))

        frame_debug = torch.cat((frame1, Final_prediction, gt, frame3), dim =0)
        #print(frame_debug.size())
        writer.add_images('my_image_batch', frame_debug, iteration)
        iteration+=1

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    print(avr_psnr,avr_ssim)#'psnr:{}, ssim:{}'.format(avr_psnr, avr_ssim))
        # print(frame_out)
    frame_debug = torch.cat((frame1, Final_prediction, gt, frame3), dim =0)
    #print(frame_debug.size())
    writer.add_images('my_image_batch', frame_debug, epoch)
    writer.add_scalars('testing', {'testing psnr':avr_psnr,
        'testing ssim': avr_ssim,
                            }, epoch)

logfile.close()
writer.close()


