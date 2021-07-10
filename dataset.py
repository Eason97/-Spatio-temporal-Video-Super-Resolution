import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
import random
import torchvision

#data augmentation for image rotate
def augment(frame0, frame1,frame2,mask):
    # 0 represent rotate,1 represent vertical,2 represent horizontal,3 represent remain orginal
    augmentation_method=random.choice([0,1,2,3,4,5])
    rotate_degree = random.choice([0, 90, 180, 270])
    '''Rotate'''
    if augmentation_method==0:
        frame0 = transforms.functional.rotate(frame0, rotate_degree)
        frame1 = transforms.functional.rotate(frame1, rotate_degree)
        frame2 = transforms.functional.rotate(frame2, rotate_degree)
        mask = transforms.functional.rotate(mask, rotate_degree)
        return frame0,frame1,frame2,mask
    '''Vertical'''
    if augmentation_method==1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        frame0 = vertical_flip(frame0)
        frame1 = vertical_flip(frame1)
        frame2 = vertical_flip(frame2)
        mask = vertical_flip(mask)
        return frame0, frame1, frame2, mask
    '''Horizontal'''
    if augmentation_method==2:
        horizontal_flip= torchvision.transforms.RandomHorizontalFlip(p=1)
        frame0 = horizontal_flip(frame0)
        frame1 = horizontal_flip(frame1)
        frame2 = horizontal_flip(frame2)
        mask = horizontal_flip(mask)
        return frame0, frame1, frame2, mask
    '''no change'''
    if augmentation_method==3 or augmentation_method==4 or augmentation_method==5:
        return frame0, frame1, frame2, mask


class vimeo_dataset(Dataset):
    def __init__(self, viemo_dir, mask_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train=[]
        for line in open('/ssd2/minghan/vimeo_triplet/'+'tri_trainlist.txt'):
            line = line.strip('\n')
            if line!='':
                self.list_train.append(line)
        self.root=viemo_dir
        #/home/fum16/context_aware_mask/mask
        self.mask_dir = mask_dir
        self.file_len = len(self.list_train)

    def __getitem__(self, index, is_train = True):

        frame0 = Image.open(self.root + self.list_train[index] + '/' + "im1.png")
        frame1 = Image.open(self.root + self.list_train[index] + '/' + "im2.png")
        frame2 = Image.open(self.root + self.list_train[index] + '/' + "im3.png")
        mask_image = Image.open(self.mask_dir+ self.list_train[index] + '.png')

        #crop a patch
        i,j,h,w = transforms.RandomCrop.get_params(frame0, output_size = (256, 256))
        frame0_ = TF.crop(frame0, i, j, h, w)
        frame1_ = TF.crop(frame1, i, j, h, w)
        frame2_ = TF.crop(frame2, i, j, h, w)
        mask_=TF.crop(mask_image, i, j, h, w)


        #data argumentation
        frame0_arg, frame1_arg,frame2_arg,mask_arg = augment(frame0_, frame1_,frame2_,mask_)

        #BICUBIC down-sampling
        frame0_arg_down = frame0_arg.resize((int(frame0_arg.size[0]//4), int(frame0_arg.size[1]//4)), Image.BICUBIC)
        frame2_arg_down = frame2_arg.resize((int(frame2_arg.size[0]//4), int(frame2_arg.size[1]//4)), Image.BICUBIC)
        gt_arg_down= frame1_arg.resize((int(frame1_arg.size[0]//4), int(frame1_arg.size[1]//4)), Image.BICUBIC)

        #transform
        frame0_high = self.transform(frame0_arg)#torch.Size([3, 64, 64])
        frame0_low = self.transform(frame0_arg_down)

        frame1_high = self.transform(frame1_arg)#torch.Size([3, 256, 256])
        frame1_low = self.transform(gt_arg_down)

        frame2_high = self.transform(frame2_arg)#torch.Size([3, 64, 64])
        frame2_low = self.transform(frame2_arg_down)

        return frame0_low,frame0_high,frame1_low,frame1_high,frame2_low,frame2_high
    def __len__(self):
        return self.file_len