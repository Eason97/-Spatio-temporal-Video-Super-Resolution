import torch
import time
import argparse
from multiscale_backwarp_model import VISRNet
from dataset import vimeo_dataset
from torch.utils.data import DataLoader
import os
from torchvision.models import vgg16
import torch.nn.functional as F
from utils_test import to_psnr,to_ssim_skimage
from test_dataloader import vimeo_test_dataset
from perceptual import LossNetwork
from tensorboardX import SummaryWriter


# --- Parse hyper-parameters  train --- #
parser = argparse.ArgumentParser(description='context-aware')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=12, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=300, type=int)
parser.add_argument('--train_data_mask', type=str, default='/ssd2/minghan/context_aware_mask/mask/')
parser.add_argument('--train_dataset_imgs', type=str, default='/ssd2/minghan/vimeo_triplet/sequences/')
parser.add_argument('--out_dir', type=str, default='./output_result')
parser.add_argument('--load_model', type=str, default=None)
# --- Parse hyper-parameters  test --- #
parser.add_argument('--test_data_mask', type=str, default='/ssd2/minghan/context_aware_mask/mask/')
parser.add_argument('--test_dataset_imgs', type=str, default='/ssd2/minghan/vimeo_triplet/sequences/')
parser.add_argument('--test_gt', type=str, default='/ssd2/minghan/vimeo_triplet/sequences/')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=80,  type=int)

args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch= args.train_epoch
train_dataset_imgs=args.train_dataset_imgs
train_data_mask= args.train_data_mask
train_model=args.load_model

# --- test --- #
test_input = args.test_dataset_imgs
test_mask = args.test_data_mask
predict_result= args.predict_result
test_batch_size=args.test_batch_size

# --- output picture and logfile --- #
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
net = VISRNet()


# --- Build optimizer --- #
optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80,100], gamma=0.5)
# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data --- #
dataset = vimeo_dataset(train_dataset_imgs, train_data_mask)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
# --- Load testing data --- #
test_dataset = vimeo_test_dataset(test_input, test_mask)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

max_step = train_loader.__len__()
criterion=torch.nn.L1Loss()

# --- Multi-GPU --- #
gpu_nums=torch.cuda.device_count()
net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=device_ids)
writer = SummaryWriter()

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load the network weight --- #
try:
    net.load_state_dict(torch.load('./VISR.pkl'))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# --- Strat training --- #
iteration = 0
for epoch in range(train_epoch):
    start_time = time.time()
    scheduler_G.step()
    net.train()
    print(epoch)
    for batch_idx, (frame0_low,frame0_high,frame1_low,frame1_high,frame2_low,frame2_high) in enumerate(train_loader):
        iteration +=1
        frame0_low= frame0_low.to(device)
        frame0_high= frame0_high.to(device)
        frame1_low = frame1_low.to(device)
        frame1_high= frame1_high.to(device)
        frame2_low = frame2_low.to(device)
        frame2_high = frame2_high.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()
        # --- Forward + Backward + Optimize --- #
        interpolated_frame,SR_frame1,SR_middle_frame,SR_frame3 = net(frame0_low,frame2_low)

        l1_VI=F.smooth_l1_loss(interpolated_frame, frame1_low)
        #super resolution loss
        smooth_loss_l1_middle = F.smooth_l1_loss(SR_middle_frame, frame1_high)
        smooth_loss_l1_1 = F.smooth_l1_loss(SR_frame1, frame0_high)
        smooth_loss_l1_3 = F.smooth_l1_loss(SR_frame3, frame2_high)


        perceptual_loss = loss_network(SR_middle_frame, frame1_high)
        total_loss =smooth_loss_l1_1+smooth_loss_l1_3+smooth_loss_l1_middle+0.04*perceptual_loss+l1_VI
        total_loss.backward()
        optimizer.step()
        if iteration%100==0:
            frame_debug1 = torch.cat((SR_middle_frame, frame1_high), dim =0)
            frame_debug2 = torch.cat((interpolated_frame, frame1_low), dim =0)
            writer.add_scalars('training', {'training total loss': total_loss.item()
                                            }, iteration)
            writer.add_scalars('super_resolution', {'frame1': smooth_loss_l1_1.item(),
                                                    'frame2': smooth_loss_l1_middle.item(),
                                                    'frame3': smooth_loss_l1_3.item(),
                                                'perceptual':0.04*perceptual_loss.item(),
                                                }, iteration)
            writer.add_scalars('video_interpolation', {'rgb': l1_VI.item()}, iteration)

    if epoch % 1 == 0:
        print('we are testing on epoch: ' + str(epoch))
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            for batch_idx, (frame0_left,frame0_right, frame2_left,frame2_right,frame1) in enumerate(test_loader):
                frame0_l = frame0_left.to(device)
                frame0_r= frame0_right.to(device)

                frame2_l = frame2_left.to(device)
                frame2_r =frame2_right.to(device)

                gt = frame1.to(device)
                frame_out_drop1,frame_out_l = net(frame0_l, frame2_l) #[2, 3, 256, 256]
                frame_out_drop2,frame_out_r = net(frame0_r, frame2_r) #[2, 3, 256, 256]

                #torch.Size([2, 3, 256, 448])
                frame_out = (torch.cat([frame_out_l[:, :, :, 0:224].permute(0, 3, 1, 2),
                                        frame_out_r[:, :, :, 32:256].permute(0, 3, 1, 2)],1)).permute(0, 2, 3, 1)
                psnr_list.extend(to_psnr(frame_out, gt))
                ssim_list.extend(to_ssim_skimage(frame_out, gt))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            print('psnr:',avr_psnr,'ssim:',avr_ssim)
            frame_debug = torch.cat((frame_out, gt), dim =0)
            #print(frame_debug.size())
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr':avr_psnr,
                'testing ssim': avr_ssim,
                                    }, epoch)
            torch.save(net.state_dict(), 'epoch'+str(epoch) + '.pkl')
logfile.close()
writer.close()


