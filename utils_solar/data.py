import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torchvision.datasets as dsets
from torch.utils.data import Dataset

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, reverse=False,shuffle=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse,shuffle=shuffle)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse,shuffle=False):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num =  img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))                    
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        if shuffle:
            random.shuffle(img_num_per_cls)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


def load_cifar100_imbalance(partial_rate, batch_size, hierarchical, imb_type='exp', imb_factor=0.01,con=True,test=False,shuffle=False):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    # temp_train = dsets.CIFAR100(root='./data', train=True, download=True)
    temp_train = IMBALANCECIFAR100(root='./data', train=True, download=True,imb_type=imb_type, imb_factor=imb_factor,shuffle=shuffle)
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
    # get original data and labels

    test_dataset = dsets.CIFAR100(root='./data', train=False, transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
    #     sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4)
    num_classes = len(np.unique(temp_train.targets))
    assert num_classes == 100
    cls_num_list_true_label = [0] * num_classes
    for label in temp_train.targets:
        cls_num_list_true_label[label] += 1
    if test:
        return test_loader,cls_num_list_true_label
    
    if hierarchical:
        # partialY = generate_hierarchical_cv_candidate_labels('cifar100',labels, partial_rate)
        partialY=generate_hierarchical_and_uniform_cv_candidate_labels('cifar100',labels,partial_rate=partial_rate)
        # for fine-grained classification
    else:
        partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    print('Average candidate num: ', partialY.sum(1).mean())
    train_label_cnt = torch.unique(labels, sorted=True, return_counts=True)[-1]
    partial_matrix_dataset = CIFAR100_Augmentention(data, partialY.float(), labels.float(),con=con)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    # partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=(train_sampler is None), 
    #     num_workers=4,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     drop_last=True)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    init_label_dist = torch.ones(num_classes)/num_classes
    est_dataset = CIFAR100_Augmentention(data, partialY.float(), labels.float(),con=con,transform=test_transform)
    est_loader = torch.utils.data.DataLoader(dataset=est_dataset, 
        batch_size=batch_size * 4, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True)
    
    return partial_matrix_train_loader,partialY,test_loader,est_loader,init_label_dist,train_label_cnt
    # return train_loader, train_givenY, test_loader,est_loader, cls_num_list_true_label

def load_cifar10_imbalance(partial_rate, batch_size, hierarchical=False, imb_type='exp', imb_factor=0.01,con=True,test=False,shuffle=False):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    hierarchical = False
    # temp_train = dsets.CIFAR10(root='./data', train=True, download=True)
    temp_train = IMBALANCECIFAR10(root='./data', train=True, download=True,imb_type=imb_type, imb_factor=imb_factor,shuffle=shuffle)
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
    # get original data and labels

    test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
    #     sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4)
    num_classes = len(np.unique(temp_train.targets))
    assert num_classes == 10
    cls_num_list_true_label = [0] * num_classes
    for label in temp_train.targets:
        cls_num_list_true_label[label] += 1
    if test:
        return test_loader,cls_num_list_true_label
    
    # if hierarchical:
    #     # partialY = generate_hierarchical_cv_candidate_labels('cifar100',labels, partial_rate)
    #     partialY=generate_hierarchical_and_uniform_cv_candidate_labels('cifar100',labels,partial_rate=partial_rate)
    #     # for fine-grained classification
    # else:
    #     partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    
    # generate partial labels
    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    print('Average candidate num: ', partialY.sum(1).mean())

    train_label_cnt = torch.unique(labels, sorted=True, return_counts=True)[-1]
    partial_matrix_dataset = CIFAR10_Augmentention(data, partialY.float(), labels.float(),con=con)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    # partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
    #     batch_size=batch_size, 
    #     shuffle=(train_sampler is None), 
    #     num_workers=4,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     drop_last=True)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    init_label_dist = torch.ones(num_classes)/num_classes
    est_dataset = CIFAR10_Augmentention(data, partialY.float(), labels.float(),con=con,transform=test_transform)
    est_loader = torch.utils.data.DataLoader(dataset=est_dataset, 
        batch_size=batch_size * 4, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True)

    
    return partial_matrix_train_loader,partialY,test_loader,est_loader,init_label_dist,train_label_cnt


class CIFAR10_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels,con=True,transform=None):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.transform = transform
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.con = con
        if self.transform is not None:
            self.con = False

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        if self.con:
            each_image_w = self.weak_transform(self.images[index])
            each_image_s = self.strong_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            
            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            if self.transform is not None:
                each_image_w = self.transform(self.images[index])
            else:
                each_image_w = self.weak_transform(self.images[index])
            each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            return each_image_w, each_label, each_true_label, index



class CIFAR100_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels,con=True,transform=None):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.transform = transform
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.con = con
        if self.transform is not None:
            self.con = False

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        if self.con:
            each_image_w = self.weak_transform(self.images[index])
            each_image_s = self.strong_transform(self.images[index])
            if self.given_label_matrix is None:
                each_label = -1
            else:
                each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            return each_image_w, each_image_s, each_label, each_true_label, index
        else:
            if self.transform is not None:
                each_image_w = self.transform(self.images[index])
            else:
                each_image_w = self.weak_transform(self.images[index])
            if self.given_label_matrix is None:
                each_label = -1
            else:
                each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            return each_image_w, each_label, each_true_label, index


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def generate_hierarchical_and_uniform_cv_candidate_labels(dataname, train_labels, partial_rate=0.1,root = "data",ratio_hi=8):
    assert dataname == 'cifar100'

    meta_root = os.path.join(root,'cifar-100-python/meta')
    # meta = unpickle('data/cifar-100-python/meta')
    meta = unpickle(meta_root)

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1*ratio_hi
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    # transition_matrix *= 5
    transition_matrix[transition_matrix==0] = p_1
    
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0
    print("Finish Generating Candidate Label Sets!\n")
    return partialY

import os
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res


import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)



def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

    
def augment_list():  
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

    
class RandomAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        self.augment_list = augment_list()

        
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val) 
        cutout_val = random.random() * 0.5 
        img = Cutout(img, cutout_val) #for fixmatch
        return img

    