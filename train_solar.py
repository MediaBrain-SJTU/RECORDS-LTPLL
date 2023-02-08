import argparse
import math
import os
import random
import shutil
import time
import torch
import torch.nn 
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from utils_solar.resnet import *
from utils_solar.general import *
from utils_solar.data import *
from datetime import datetime

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of SoLar')
parser.add_argument('--dataset', default='cifar10_im', type=str,
                    help='dataset name (cifar10)')
parser.add_argument('--exp_dir', default='experiment/CIFAR-10', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--data_dir', default='data/pre-processed-data', type=str,
                    help='experiment directory for loading pre-generated data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 used in SoLar)')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, 
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_class', default=10, type=int,
                    help='number of class')
parser.add_argument('--queue_length', default=64, type=int, 
                    help='the queue size is queue_length*batch_size')
parser.add_argument('--lamd', default=3, type=float,
                    help='parameter for sinkhorn algorithm')
parser.add_argument('--eta', default=0.9, type=float,
                    help='final weight of re-normalize loss')
parser.add_argument('--tau', default=0.99, type=float,
                    help='high-confidence selection threshold')
parser.add_argument('--rho_range', default='0.2,0.8', type=str,
                    help='ratio of clean labels (rho)')
parser.add_argument('--gamma', default='0.1,0.01', type=str,
                    help='distribution refinery param')
parser.add_argument('--warmup_epoch', default=50, type=int, 
                    help = 'warm-up training for unreliable examples')
parser.add_argument('--est_epochs', default=20, type=int, 
                    help = 'epochs for pre-estimating the class prior')
parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='ambiguity level (phi)')
parser.add_argument('--hierarchical', action='store_true', 
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--imb_type', default='exp', choices=['exp', 'step'],
                    help='imbalance data type')
parser.add_argument('--imb_ratio', default=50, type=float, 
                    help='imbalance ratio for long-tailed dataset generation')
parser.add_argument('--save_ckpt', action='store_true', 
                    help='whether save the model')

class Trainer():
    def __init__(self, args):
        self.args = args

        model_path = '{ds}_{pr}_ql{ql}_rho{rho}_gm{gm}_t{t}_ep{we}_{ee}_imb_{it}{imf}_sd_{seed}'.format(
                                                ds=args.dataset,
                                                pr=args.partial_rate,
                                                ep=args.epochs,
                                                ql=args.queue_length,
                                                rho=args.rho_range,
                                                it=args.imb_type,
                                                imf=args.imb_factor,
                                                seed=args.seed,
                                                gm=args.gamma,
                                                t=args.tau,
                                                we=args.warmup_epoch,
                                                ee=args.est_epochs)
        args.exp_dir = os.path.join(args.exp_dir, model_path)
        args.exp_dir = os.path.join(
        args.exp_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            cudnn.deterministic = True
        if args.hierarchical:
            class_shuffle = True
        else:
            class_shuffle = False
        if args.dataset == 'cifar10_im':
            # train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt\
                # = load_cifar10_imbalance(args=args)
            train_loader, train_givenY, test_loader,est_loader, init_label_dist, train_label_cnt = load_cifar10_imbalance(
            partial_rate=args.partial_rate, batch_size=args.batch_size, hierarchical=args.hierarchical, imb_factor=args.imb_factor, con=True,shuffle=class_shuffle)
            many_shot_num = 3
            low_shot_num = 3
        elif args.dataset == 'cifar100_im':
            train_loader, train_givenY, test_loader,est_loader, init_label_dist, train_label_cnt = load_cifar100_imbalance(
            partial_rate=args.partial_rate, batch_size=args.batch_size, hierarchical=args.hierarchical, imb_factor=args.imb_factor, con=True,shuffle=class_shuffle)
            many_shot_num = 33
            low_shot_num = 33
        # elif args.dataset == 'cub200':
        #     input_size = 224
        #     train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt = load_cub200(
        #                                                                     data_dir=args.data_dir,
        #                                                                     input_size=input_size, 
        #                                                                     partial_rate=args.partial_rate, 
        #                                                                     batch_size=args.batch_size, 
        #                                                                     imb_factor=args.imb_factor)
        #     many_shot_num = 66
        #     low_shot_num = 66
        # elif args.dataset == 'sun397':
        #     input_size = 224
        #     train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt = load_sun397(
        #                                                                     data_dir=args.data_dir,
        #                                                                     input_size=input_size, 
        #                                                                     partial_rate=args.partial_rate, 
        #                                                                     batch_size=args.batch_size)
        #     many_shot_num = 132
        #     low_shot_num = 132
        else:
            raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
        # this train loader is the partial label training loader

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.est_loader = est_loader
        self.init_label_dist = init_label_dist
        self.train_givenY = train_givenY
        # set loss functions (with pseudo-targets maintained)
        self.acc_shot = AccurracyShot(train_label_cnt, args.num_class, many_shot_num, low_shot_num)

    def train(self, emp_dist=None, is_est_dist=False, total_epochs=0, gamma=0):
        # create model
        print("=> creating model '{}'".format(args.arch))
        if args.dataset in ['cub200', 'sun397']:
            print('Loading Pretrained Model')
            # model = resnet18(num_class=args.num_class, pretrained=True)
            model = ResNet_s(name='resnet18', num_class=args.num_class, pretrained=True)
        else:
            # model = resnet18(num_class=args.num_class)
            model = ResNet_s(name='resnet18', num_class=args.num_class, pretrained=False)
        model = model.cuda(args.gpu)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # set optimizer
        loss_fn = partial_loss(self.train_givenY)
        # Reinitialize loss function with uniform targets
        queue = None
        if args.queue_length > 0 and queue is None:
            queue = torch.zeros(args.queue_length, args.num_class).cuda()
        # initialize queue for Sinkhorn iteration

        best_acc = 0
        
        if is_est_dist:
            tip = '------------- Stage: Pre-Estimation --------------'
        else:
            tip = '------------- Stage: Final Training --------------'
            total_epochs = args.epochs

        print(tip)
        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write(tip + '\n')

        if emp_dist is None:
            # if emp_dist is not given, use initialized dist
            emp_dist = self.init_label_dist.unsqueeze(dim=1)

        for epoch in range(total_epochs):
            is_best = False

            adjust_learning_rate(args, optimizer, epoch)
            self.train_loop(model, loss_fn, queue, emp_dist, optimizer, epoch)

            emp_dist_train = self.estimate_empirical_distribution(model, self.est_loader, num_class=args.num_class)
            # estimating empirical class prior by counting prediction
            emp_dist = emp_dist_train * gamma + emp_dist * (1 - gamma)
            # moving-average updating class prior

            acc_test, acc_many, acc_med, acc_few = self.test(model, self.test_loader)
            
            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write('Epoch {}: Acc {:.2f}, Best Acc {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(epoch
                    , acc_test, best_acc, acc_many, acc_med, acc_few, optimizer.param_groups[0]['lr']))
            
            if acc_test > best_acc:
                best_acc = acc_test
                is_best = True

            if not is_est_dist and args.save_ckpt:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
                best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))
            # save checkpoints
            
        return emp_dist

    def train_loop(self, model, loss_fn, queue, emp_dist, optimizer, epoch):
        args = self.args
        train_loader = self.train_loader

        batch_time = AverageMeter('Time', ':1.2f')
        data_time = AverageMeter('Data', ':1.2f')
        acc_cls = AverageMeter('Acc@Cls', ':2.2f')
        acc_sink = AverageMeter('Acc@Sink', ':2.2f')
        loss_cls_log = AverageMeter('Loss@RC', ':2.2f')
        loss_sink_log = AverageMeter('Loss@Sink', ':2.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, acc_cls, acc_sink, loss_cls_log, loss_sink_log],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        eta = args.eta * linear_rampup(epoch, args.warmup_epoch)
        rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup(epoch, args.warmup_epoch)
        # calculate weighting parameters
        
        end = time.time()
        
        for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
            # X_w, Y, index = images_w.cuda(), labels.cuda(), index.cuda()
            Y_true = true_labels.long().detach().cuda()
            # for showing training accuracy and will not be used when training

            logits_w = model(X_w)
            logits_s = model(X_s)
            bs = args.batch_size

            prediction = F.softmax(logits_w.detach(), dim=1)
            sinkhorn_cost = prediction * Y
            # calculate sinkhorn cost (M matrix in our paper)
            conf_rn = sinkhorn_cost / sinkhorn_cost.sum(dim=1).repeat(prediction.size(1), 1).transpose(0, 1)
            # re-normalized prediction for unreliable examples

            # time to use queue, output now represent queue+output
            prediction_queue = sinkhorn_cost.detach()
            if queue is not None:
                if not torch.all(queue[-1, :] == 0):
                    prediction_queue = torch.cat((queue, prediction_queue))
                # fill the queue
                queue[bs:] = queue[:-bs].clone().detach()
                queue[:bs] = prediction_queue[-bs:].clone().detach()
            pseudo_label_soft, flag = sinkhorn(prediction_queue, args.lamd, r_in=emp_dist)
            pseudo_label = pseudo_label_soft[-bs:]
            pseudo_label_idx = pseudo_label.max(dim=1)[1]

            _, rn_loss_vec = loss_fn(logits_w, index)
            _, pseudo_loss_vec = loss_fn(logits_w, None, targets=pseudo_label)

            idx_chosen_sm = []
            sel_flags = torch.zeros(X_w.shape[0]).cuda().detach()
            # initialize selection flags
            for j in range(args.num_class):
                indices = np.where(pseudo_label_idx.cpu().numpy()==j)[0]
                # torch.where will cause device error
                if len(indices) == 0:
                    continue
                    # if no sample is assigned this label (by argmax), skip
                bs_j = bs * emp_dist[j]
                pseudo_loss_vec_j = pseudo_loss_vec[indices]
                sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
                partition_j = max(min(int(math.ceil(bs_j*rho)), len(indices)), 1)
                # at least one example
                idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])

            idx_chosen_sm = np.concatenate(idx_chosen_sm)
            sel_flags[idx_chosen_sm] = 1
            # filtering clean sinkhorn labels
            high_conf_cond = (pseudo_label * prediction).sum(dim=1) > args.tau
            sel_flags[high_conf_cond] = 1
            idx_chosen = torch.where(sel_flags == 1)[0]
            idx_unchosen = torch.where(sel_flags == 0)[0]

            if epoch < 1 or idx_chosen.shape[0] == 0:
                # first epoch, using uniform labels for training
                # else, if no samples are chosen, run rn 
                loss = rn_loss_vec.mean()
            else:
                if idx_unchosen.shape[0] > 0:
                    loss_unreliable = rn_loss_vec[idx_unchosen].mean()
                else:
                    loss_unreliable = 0
                loss_sin = pseudo_loss_vec[idx_chosen].mean()
                loss_cons, _ = loss_fn(logits_s[idx_chosen], None, targets=pseudo_label[idx_chosen])
                # consistency regularization
                
                l = np.random.beta(4, 4)
                l = max(l, 1-l)
                X_w_c = X_w[idx_chosen]
                pseudo_label_c = pseudo_label[idx_chosen]
                idx = torch.randperm(X_w_c.size(0))
                X_w_c_rand = X_w_c[idx]
                pseudo_label_c_rand = pseudo_label_c[idx]
                X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand        
                pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
                logits_mix = model(X_w_c_mix)
                loss_mix, _  = loss_fn(logits_mix, None, targets=pseudo_label_c_mix)
                # mixup training

                loss = (loss_sin + loss_mix + loss_cons) * eta + loss_unreliable * (1 - eta)
            # loss = rn_loss_vec.mean()
            
            loss_sink_log.update(pseudo_loss_vec.mean().item())
            loss_cls_log.update(rn_loss_vec.mean().item())

            # log accuracy
            acc = accuracy(logits_w, Y_true)[0]
            acc_cls.update(acc[0])
            acc = accuracy(pseudo_label, Y_true)[0]
            acc_sink.update(acc[0])
    
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time

            loss_fn.confidence_update(conf_rn, index)
            # update confidences for re-normalization loss (for unreliable examples)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

    def test(self, model, test_loader):
        with torch.no_grad():
            print('==> Evaluation...')       
            model.eval()    
            pred_list = []
            true_list = []
            for _, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                outputs = model(images)
                pred = F.softmax(outputs, dim=1)
                pred_list.append(pred.cpu())
                true_list.append(labels)
            
            pred_list = torch.cat(pred_list, dim=0)
            true_list = torch.cat(true_list, dim=0)
            
            acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
            acc_many, acc_med, acc_few = self.acc_shot.get_shot_acc(pred_list.max(dim=1)[1], true_list)
            print('==> Test Accuracy is %.2f%% (%.2f%%), [%.2f%%, %.2f%%, %.2f%%]'%(acc1, acc5, acc_many, acc_med, acc_few))
        return float(acc1), float(acc_many), float(acc_med), float(acc_few)

    def estimate_empirical_distribution(self, model, est_loader, num_class):
        with torch.no_grad():
            print('==> Estimating empirical label distribution ...')       
            model.eval()
            est_pred_list = []
            for _, (images, labels, _,_) in enumerate(est_loader):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)   
                pred = torch.softmax(outputs, dim=1) * labels
                est_pred_list.append(pred.cpu())
            
        est_pred_idx = torch.cat(est_pred_list, dim=0).max(dim=1)[1]
        est_pred = F.one_hot(est_pred_idx, num_class).detach()
        emp_dist = est_pred.sum(0)
        emp_dist = emp_dist / float(emp_dist.sum())

        return emp_dist.unsqueeze(1)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'sun397':
        hints = '[Warning]: Under the offline empirical distribution mode.\n\
                Can be slow for the SUN397 dataset; recommend running train_online.py.\n\
                Still run offline mode? Type yes (y) or no (n):'
        while True:
            answer = input(hints)
            if answer in ['yes', 'y']:
                break
            elif answer in ['no', 'n']:
                quit()
            else:
                hints = 'Wrong input; Type yes (y) or no (n):'
    
    [args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(',')]
    [args.gamma1, args.gamma2] = [float(item) for item in args.gamma.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.queue_length *= args.batch_size
    print(args)
    torch.cuda.set_device(args.gpu)
    args.imb_factor = 1. / args.imb_ratio
    # set imb_factor as 1/imb_ratio
    trainer = Trainer(args)
    emp_dist = trainer.train(is_est_dist=True, total_epochs=args.est_epochs, gamma=args.gamma1)
    trainer.train(emp_dist=emp_dist, gamma=args.gamma2)