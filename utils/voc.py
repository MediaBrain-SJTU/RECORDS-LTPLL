import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
try:
    from .randaugment import RandomAugment
except:
    from randaugment import RandomAugment
import pandas as pd
import os
from PIL import Image

All_labels = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
id2label = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}
class VOC(Dataset):
    def __init__(self,csv_dir,image_dir,transform=None,size=(128,128),split="train",con=False):
        self.transform = transform
        self.size = size
        data = pd.read_csv(csv_dir)
        self.images = data['name'].to_list()
        self.labels = data['label'].to_list()
        self.split = split
        if self.split == "train":
            self.label_sets = data['label set'].to_list()
        self.image_dir = image_dir
        self.con = con
        if self.split == "train":
            self.partialY = torch.zeros(len(self.images),len(All_labels))
            for i in range(len(self.images)):
                label_set = self.label_sets[i]
                label_set = label_set.split(" ")
                label_set = [All_labels[label] for label in label_set]
                self.partialY[i][label_set] = 1
            

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir,image_name))
        height,width = image.size
        # print(height,width)
        if height>width:
            padding = (height-width)//2
            image = transforms.Pad([0,padding,0,height-width-padding])(image)
            
        elif height<width:
            padding = (width-height)//2
            image = transforms.Pad([padding,0,width-height-padding,0])(image)
        # print(image.size)
        image = transforms.Resize(self.size)(image)
        image_i = image
        # print(image)
        if self.transform:
            if isinstance(self.transform, list):
                transform = self.transform[0]
                image = transform(image)

            else:
                image = self.transform(image)
            if self.con:
                transform = self.transform[1]
                # print(transform)
                image_s = transform(image_i)
        if self.split == "train":
            Partial_Y = self.partialY[idx]
        label = self.labels[idx]
        label = All_labels[label]
        Y = label
        if self.split == "train":
            if self.con:
                return image,image_s,Partial_Y,Y,idx
            else:
                return image,Partial_Y,Y,idx
        elif self.split == "test":
            return image,Y

def load_voc(batch_size,con=True,test=False):
    train_image_dir = './data/VOC2017/PLL/train_var/images'
    train_csv_dir = './data/VOC2017/PLL/train_var/images_PLL.csv'
    test_image_dir = './data/VOC2017/PLL/test/images'
    test_csv_dir = './data/VOC2017/PLL/test/images_uniform.csv'
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.2554, 0.2243, 0.2070), (0.2414, 0.2207, 0.2104))])
    test_dataset = VOC(test_csv_dir,test_image_dir,transform=test_transform,split="test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    if test:
        return test_loader
    
    weak_transform = transforms.Compose(
            [
            transforms.RandomResizedCrop(size=(128,128), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.2554, 0.2243, 0.2070), (0.2414, 0.2207, 0.2104))])
    strong_transform = transforms.Compose(
            [
            transforms.RandomResizedCrop(size=(128,128), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.2554, 0.2243, 0.2070), (0.2414, 0.2207, 0.2104))])
    
    if con:
        train_dataset = VOC(train_csv_dir,train_image_dir,transform=[weak_transform,strong_transform],split="train",con=True)
    else:
        train_dataset = VOC(train_csv_dir,train_image_dir,transform=weak_transform,split="train",con=False)
    n = len(train_dataset)
    partialY = torch.zeros(n,len(All_labels))
    label_names = train_dataset.labels
    cls_num_list_true_label = [0] * len(All_labels)
    for label_name in label_names:
        label_id = All_labels[label_name]
        cls_num_list_true_label[label_id] += 1
    partialY = train_dataset.partialY
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    return train_loader,partialY,train_sampler,test_loader, cls_num_list_true_label

if __name__=="__main__":
    import torch.distributed as dist
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:10112",
                                world_size=1, rank=0)
    train_loader,partialY,train_sampler,test_loader,cls_num_list_true_label = load_voc(batch_size=128,con=True,test=False)
    print(cls_num_list_true_label)
    print(partialY.shape)
    print(partialY.sum(1).mean())
    for i, (images_w,images_s, labels, true_labels, index) in enumerate(train_loader):
        print(images_w.shape)
        print(images_s.shape)
        print(labels[2])
        print(true_labels[2])
        print(index.shape)
        break
    for i, (images, labels) in enumerate(test_loader):
        print(images.shape)
        print(labels[1])
        break

    


