import argparse
import shutil
import datetime
import time
import random
import utils
import preprocessing

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import profiler
from torch.nn import init





parser = argparse.ArgumentParser(description='CliqueNet ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to the imagenet dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default = 100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default= 256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default = 0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default = 50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='./cliquenet_64&36_80_120_100&5_6_6_6.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: cliquenet_64&36_80_120_100&5_6_6_6)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_prec1 = 0



def main():
    
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = TI_II(input_channels = 64, list_channels = [36, 80, 120, 100], list_layer_num = [5, 6, 6, 6])
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

   # # optionally resume from a checkpoint
   #  if args.resume:
   #      if os.path.isfile(args.resume):
   #          print("=> loading checkpoint '{}'".format(args.resume))
   #          checkpoint = torch.load(args.resume)
   #          args.start_epoch = checkpoint['epoch']
   #          best_prec1 = checkpoint['best_prec1']
   #          model.load_state_dict(checkpoint['state_dict'])
   #          optimizer.load_state_dict(checkpoint['optimizer'])
   #          print("=> loaded checkpoint '{}' (epoch {})"
   #                .format(args.resume, checkpoint['epoch']))
   #      else:
   #          print("=> no checkpoint found at '{}'".format(args.resume))

   #  cudnn.benchmark = True


   #  # Data loading code
   #  traindir = os.path.join(args.data, 'train')
   #  valdir = os.path.join(args.data, 'val')
   #  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
   #                                   std=[0.229, 0.224, 0.225])

   #  train_loader = torch.utils.data.DataLoader(
   #      datasets.ImageFolder(traindir, transforms.Compose([
   #          transforms.RandomSizedCrop(224),
   #          transforms.RandomHorizontalFlip(),
   #          transforms.ToTensor(),
   #          ColorJitter(
   #          brightness=0.4,
   #          contrast=0.4,
   #          saturation=0.4,
   #          ),
   #          normalize,
   #      ])),
   #      batch_size=args.batch_size, shuffle=True,
   #      num_workers=args.workers, pin_memory=True)

   #  val_loader = torch.utils.data.DataLoader(
   #      datasets.ImageFolder(valdir, transforms.Compose([
   #          transforms.Scale(256),
   #          transforms.CenterCrop(224),
   #          transforms.ToTensor(),
   #          normalize,
   #      ])),
   #      batch_size=args.batch_size, shuffle=False,
   #      num_workers=args.workers, pin_memory=True)

   #  if args.evaluate:
   #      validate(val_loader, model, criterion)
   #      return
    get_number_of_param(model)
    # init.normal(model.parameters(),0.01)
    # pdb.set_trace()
    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch)

    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch)

    #     # evaluate on validation set
    #     prec1 = validate(val_loader, model, criterion)

    #     # remember best prec@1 and save checkpoint
    #     is_best = prec1 > best_prec1
    #     best_prec1 = max(prec1, best_prec1)
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': model.state_dict(),
    #         'best_prec1': best_prec1,
    #         'optimizer' : optimizer.state_dict(),
    #     }, is_best)

def get_number_of_param(model):
    """get the number of param for every element"""
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for dis in param_size:
            count_of_one_param *= dis
        print(param.size(), count_of_one_param)
        count += count_of_one_param
    print('total number of the model is %d'%count)



def train(train_loader, model, criterion, optimizer, epoch):
    """train model"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # last_datetime = datetime.datetime.now()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        # with profiler.profile() as prof_in:
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        # pdb.set_trace()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
      # print(prof_in)
                  
        if i % 10 == 0:
            now_datetime = datetime.datetime.now()
            print(now_datetime)
            # print("delta time:", now_datetime - last_datetime)
            # last_datetime = now_datetime
    print time.ctime()


def validate(val_loader, model, criterion):
    """validate model"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # last_datetime = datetime.datetime.now()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:       
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            print time.ctime()
        
        if i % 10 == 0:
            now_datetime = datetime.datetime.now()
            print(now_datetime)
            # print("delta time:", now_datetime - last_datetime)
            # last_datetime = now_datetime

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    print time.ctime()
    return top1.avg


def save_checkpoint(state, is_best, filename='./cliquenet_64&36_80_120_100&5_6_6_6.pth.tar'):
    """Save the trained model"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './best_cliquenet_64&36_80_120_100&5_6_6_6.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial Learning rate decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('current learning rate is: %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
