import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
from .utils_algo import generate_uniform_cv_candidate_labels,generate_hierarchical_and_uniform_cv_candidate_labels
from .imbalance_cifar import IMBALANCECIFAR100

def load_cifar100_imbalance(partial_rate, batch_size, hierarchical, imb_type='exp', imb_factor=0.01,con=True,test=False,shuffle=False):
    """Load PLL version of CIFAR-100-LT dataset

    Args:
        partial_rate: Ambiguity q in PLL
        batch_size: batch size
        hierarchical (bool): Whether to return CIFAR-100-LT-NU (Labels in the same superclass of GT have a higher probability to be selected into the candidate label set).
        imb_type (str, optional): Type of imbalance. Defaults to 'exp'.
        imb_factor (float, optional): Imbalance ratio: min_num / max_num. Defaults to 0.01.
        con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        test (bool, optional): Whether to return test loader. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the classes when generating the LT dataset. Defaults to False.

    """
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    temp_train = IMBALANCECIFAR100(root='./data', train=True, download=True,imb_type=imb_type, imb_factor=imb_factor,shuffle=shuffle)
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()

    test_dataset = dsets.CIFAR100(root='./data', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    num_classes = len(np.unique(temp_train.targets))
    assert num_classes == 100
    cls_num_list_true_label = [0] * num_classes
    for label in temp_train.targets:
        cls_num_list_true_label[label] += 1
    if test:
        return test_loader,cls_num_list_true_label
    
    if hierarchical:
        partialY=generate_hierarchical_and_uniform_cv_candidate_labels('cifar100',labels,partial_rate=partial_rate)
        # NU version dataset
    else:
        partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')
    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = CIFAR100_Augmentention(data, partialY.float(), labels.float(),con=con)
    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    return partial_matrix_train_loader,partialY,train_sampler,test_loader,cls_num_list_true_label



    
class CIFAR100_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels,con=True):
        """
        Args:
            images: images
            given_label_matrix: PLL candidate labels
            true_labels: GT labels
            con (bool, optional): Whether to use both weak and strong augmentation. Defaults to True.
        """
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
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
            each_image_w = self.weak_transform(self.images[index])
            if self.given_label_matrix is None:
                each_label = -1
            else:
                each_label = self.given_label_matrix[index]
            each_true_label = self.true_labels[index]
            return each_image_w, each_label, each_true_label, index




