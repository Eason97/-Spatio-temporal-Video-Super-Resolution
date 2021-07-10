from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

class vimeo_test_dataset(Dataset):
    def __init__(self, viemo_dir, mask_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train=[]
        for line in open('/ssd2/minghan/vimeo_triplet/'+'tri_testlist.txt'):
            line = line.strip('\n')
            if line!='':
                self.list_train.append(line)
        self.root=viemo_dir
        #/home/fum16/context_aware_mask/mask
        self.mask_dir = mask_dir
        self.file_len = len(self.list_train)

    def __getitem__(self, index, is_train=True):
        # if is_train:
        #frame2 = self.transform(frame2)
        # frame2_up=frame2[:,0:256,:]
        # frame2_down=frame2[:,192:448,:]

        frame1 = Image.open(self.root + self.list_train[index] + '/' + "im2.png")
        frame1 = self.transform(frame1)

        frame0 = Image.open(self.root + self.list_train[index] + '/' + "im1.png")
        frame0 = frame0.resize((int(frame0.size[0]//4), int(frame0.size[1]//4)), Image.BICUBIC)
        frame0 = self.transform(frame0)#[3, 64, 112]
        frame0_left=frame0[:,:,0:64]
        frame0_right=frame0[:,:,48:112]



        frame2 = Image.open(self.root + self.list_train[index] + '/' + "im3.png")
        frame2 = frame2.resize((int(frame2.size[0]//4), int(frame2.size[1]//4)), Image.BICUBIC)
        frame2 = self.transform(frame2)#[3, 64, 112]
        frame2_left=frame2[:,:,0:64]
        frame2_right=frame2[:,:,48:112]


        return frame0_left,frame0_right, frame2_left,frame2_right,frame1

    def __len__(self):
        return self.file_len





