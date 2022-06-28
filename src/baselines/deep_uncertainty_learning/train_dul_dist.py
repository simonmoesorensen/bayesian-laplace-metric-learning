import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm 

from tensorboardX import SummaryWriter, writer
import os
import time
import numpy as np
from PIL import Image
import random

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from config import Backbone_Dict, dul_args_func
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax, Softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, separate_irse_bn_paras, \
                       warm_up_lr, schedule_lr, get_time, AverageMeter, accuracy, add_gaussian_noise

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    # https://pytorch.org/docs/master/distributed.html#backends
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

class DUL_Trainer_dist():
    def __init__(self, rank, world_size, dul_args):
        self.dul_args = dul_args
        self.device = f"cuda:{rank}"
        self.rank = rank
        self.world_size = world_size

    def _report_configurations(self):
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        print(f"[RANK: {self.rank}] "'Experiment time: ', get_time())
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        print(f"[RANK: {self.rank}] "'Overall Configurations:')
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        for k in self.dul_args.__dict__:
            print(f"[RANK: {self.rank}] "" '{}' : '{}' ".format(k, str(self.dul_args.__dict__[k])))
        os.makedirs(self.dul_args.model_save_folder, exist_ok=True)
        os.makedirs(self.dul_args.log_tensorboard, exist_ok=True)
        writer = SummaryWriter(self.dul_args.log_tensorboard)
        return writer


    def _data_loader(self):
        time_start = time.perf_counter()
        
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        print(f"[RANK: {self.rank}] "'Loading Data...')
        if self.dul_args.center_crop:
            train_transform = transforms.Compose([ 
            transforms.Resize([int(128 * self.dul_args.input_size[0] / 112), int(128 * self.dul_args.input_size[0] / 112)]),
            transforms.RandomCrop([self.dul_args.input_size[0], self.dul_args.input_size[1]]),
            transforms.RandomHorizontalFlip(),
            add_gaussian_noise(p=self.dul_args.image_noise),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.dul_args.rgb_mean,
                                 std = self.dul_args.rgb_std),
            #transforms.RandomErasing(scale=(0.02,0.25))
        ])
        else:
            train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
            transforms.Resize([112, 112]), # smaller side resized
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.dul_args.rgb_mean,
                                 std = self.dul_args.rgb_std),
            transforms.RandomErasing(scale=(0.02,0.25))
        ])

        dataset_train = datasets.ImageFolder(self.dul_args.data_dir / 'ms1m/imgs', train_transform)

        # ----- create a weighted random sampler to process imbalanced data
        # weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
        # weights = torch.DoubleTensor(weights)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        
        sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                  num_replicas=self.world_size,
                                                                  rank=self.rank,
                                                                  seed=self.dul_args.random_seed)
        
        # In distributed training the batch size should be divided by the number of processes.                                                                  
        batch_size = self.dul_args.batch_size // self.world_size
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, sampler=sampler,
            pin_memory=self.dul_args.pin_memory, num_workers=self.dul_args.num_workers,
            shuffle=False,
            drop_last=self.dul_args.drop_last,
        )

        # num_class = train_loader.dataset.identity.max().item()
        num_class = len(train_loader.dataset.classes)

        print(f"[RANK: {self.rank}] "'Data loaded in {:.2f} seconds'.format(time.perf_counter() - time_start))
        
        print(f"[RANK: {self.rank}] ""Number of Training Classes: '{}' ".format(num_class))
        print(f"[RANK: {self.rank}] ""Number of Training Samples: '{}' ".format(len(train_loader)))
        return train_loader, num_class


    def _model_loader(self, num_class, data_size):
        # ----- backbone generate
        BACKBONE = Backbone_Dict[self.dul_args.backbone_name]
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        print(f"[RANK: {self.rank}] ""Backbone Generated: '{}' ".format(self.dul_args.backbone_name))

        # ----- head generate
        Head_Dict = {
            'ArcFace': ArcFace(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = [self.rank], s=self.dul_args.arcface_scale),
            'CosFace': CosFace(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = [self.rank]),
            'SphereFace': SphereFace(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = [self.rank]),
            'Am_softmax': Am_softmax(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = [self.rank]),
            'Softmax': Softmax(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = [self.rank])
        }
        HEAD = Head_Dict[self.dul_args.head_name]
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        print(f"[RANK: {self.rank}] ""Head Generated: '{}' ".format(self.dul_args.head_name))

        # ----- loss generate
        Loss_Dict = {
            'Focal': FocalLoss(),
            'Softmax': nn.CrossEntropyLoss(),
        }
        LOSS = Loss_Dict[self.dul_args.loss_name]
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        print(f"[RANK: {self.rank}] ""Loss Generated: '{}' ".format(self.dul_args.loss_name))
        # ----- separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)

        # ----- optimizer generate
        Optimizer_Dict = {
            'SGD': optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': self.dul_args.weight_decay}, 
                            {'params': backbone_paras_only_bn}], lr=self.dul_args.lr, momentum=self.dul_args.momentum),
            'Adam': optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': self.dul_args.weight_decay}, 
                            {'params': backbone_paras_only_bn}], lr=self.dul_args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        }
        OPTIMIZER = Optimizer_Dict[self.dul_args.optimizer]
        
        # Use Triangular Learning Rate Policy
        epoch = data_size // self.dul_args.batch_size // self.world_size
        SCHEDULER = optim.lr_scheduler.CyclicLR(OPTIMIZER, 
                                   base_lr=self.dul_args.base_lr, 
                                   max_lr=self.dul_args.max_lr,
                                   step_size_up=epoch * 2, # Recommended by https://ieeexplore.ieee.org/document/7926641
                                   mode='triangular')
        
        print(f"[RANK: {self.rank}] " + ("=" * 60))
        print(f"[RANK: {self.rank}] ""Optimizer Generated: '{}' ".format(self.dul_args.optimizer))
        print(f"[RANK: {self.rank}]", OPTIMIZER)

        # ----- optional resume
        if self.dul_args.resume_backbone or self.dul_args.resume_head:
            print(f"[RANK: {self.rank}] " + ("=" * 60))
            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            dist.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: self.device}
            
            if os.path.isfile(self.dul_args.resume_backbone):
                print(f"[RANK: {self.rank}] ""Loading Backbone Checkpoint '{}'".format(self.dul_args.resume_backbone))
                BACKBONE.load_state_dict(
                    torch.load(self.dul_args.resume_backbone, map_location=map_location)
                )
            if os.path.isfile(self.dul_args.resume_head):
                print(f"[RANK: {self.rank}] ""Loading Head Checkpoint '{}'".format(self.dul_args.resume_head))
                HEAD.load_state_dict(
                    torch.load(self.dul_args.resume_head, map_location=map_location)
                )
            
            print(f"[RANK: {self.rank}] ""Resuming from checkpoints")
        else:
            print(f"[RANK: {self.rank}] ""No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".\
                format(self.dul_args.resume_backbone, self.dul_args.resume_head))

        # ----- Distributed and multi gpu
        BACKBONE = BACKBONE.to(self.rank)
        HEAD = HEAD.to(self.rank)
        
        BACKBONE = DDP(BACKBONE, device_ids=[self.rank], output_device=self.rank)
        HEAD = DDP(HEAD, device_ids=[self.rank], output_device=self.rank)
        LOSS = LOSS.cuda()


        return BACKBONE, HEAD, LOSS, OPTIMIZER, SCHEDULER



    def _dul_runner(self):
        writer = self._report_configurations()

        train_loader, num_class = self._data_loader()
       
        BACKBONE, HEAD, LOSS, OPTIMIZER, SCHEDULER = self._model_loader(num_class=num_class, 
                                                             data_size=len(train_loader))

        DISP_FREQ = len(train_loader) // 100 # frequency to display training loss & acc

        # NUM_EPOCH_WARM_UP = self.dul_args.warm_up_epoch
        # NUM_BATCH_WARM_UP = int(len(train_loader) * NUM_EPOCH_WARM_UP)
        batch = 0  # batch index

        print(f"[RANK: {self.rank}] " + ('=' * 60))
        print(f"[RANK: {self.rank}] ""Display Freqency: '{}' ".format(DISP_FREQ))
        # print(f"[RANK: {self.rank}] ""Number of Epoch for Warm Up: '{}' ".format(NUM_EPOCH_WARM_UP))
        # print(f"[RANK: {self.rank}] ""Number of Batch for Warm Up: '{}' ".format(NUM_BATCH_WARM_UP))
        print(f"[RANK: {self.rank}] "'Start Training: ')

        for epoch in range(self.dul_args.num_epoch):
            train_loader.sampler.set_epoch(epoch)
            
            BACKBONE.train()  # set to training mode
            HEAD.train()
            BACKBONE.training = True

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            losses_KL = AverageMeter()

            for inputs, labels in tqdm(train_loader, desc=f"[Rank: {self.rank}]"):
                OPTIMIZER.zero_grad()
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long()
                loss = 0

                mu_dul, std_dul = BACKBONE(inputs) # namely, mean and std

                epsilon = torch.randn_like(std_dul)
                features = mu_dul + epsilon * std_dul
                variance_dul = std_dul**2

                loss_kl = ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5).sum(dim=-1).mean()
                losses_KL.update(loss_kl.item(), inputs.size(0))
                loss += self.dul_args.kl_scale * loss_kl

                outputs = HEAD(features, labels)

                loss_head = LOSS(outputs, labels)

                loss += loss_head

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
                losses.update(loss_head.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                loss.backward()
                OPTIMIZER.step()
                SCHEDULER.step()

                # dispaly training loss & acc every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print(f"[RANK: {self.rank}] " + ("=" * 60), flush=True)
                    print(f"[RANK: {self.rank}] "'Epoch {}/{} Batch (Step) {}/{}\t'
                          'Time {}\t'
                          'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Training Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t'
                          'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch + 1, self.dul_args.num_epoch, batch + 1, len(train_loader) * self.dul_args.num_epoch, time.asctime(time.localtime(time.time())), loss = losses, loss_KL=losses_KL,  top1 = top1, top5 = top5), flush=True)

                batch += 1 # batch index
                
                
            # training statistics per epoch (buffer for visualization)
            epoch_loss = losses.avg
            epoch_acc = top1.avg
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
            print(f"[RANK: {self.rank}] " + ("=" * 60), flush=True)
            print(f"[RANK: {self.rank}] "'Epoch: {}/{}\t'
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, self.dul_args.num_epoch, loss = losses, top1 = top1, top5 = top5), flush=True)

            # ----- save model
            if (epoch == 0 or epoch==4 or epoch==7 or epoch==12 or epoch>17) and self.rank == 0:
                print(f"[RANK: {self.rank}] " + ("=" * 60), flush=True)
                print(f"[RANK: {self.rank}] "'Saving NO.EPOCH {} trained model'.format(epoch+1), flush=True)
                
                backbone_path = os.path.join(
                    self.dul_args.model_save_folder, 
                    "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(self.dul_args.backbone_name, epoch + 1, batch, get_time()))
                head_path = os.path.join(
                    self.dul_args.model_save_folder, 
                    "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(self.dul_args.head_name, epoch + 1, batch, get_time()))

                if self.dul_args.multi_gpu:
                    torch.save(BACKBONE.state_dict(), backbone_path)
                    torch.save(HEAD.state_dict(), head_path)
                else:
                    torch.save(BACKBONE.state_dict(), backbone_path)
                    torch.save(HEAD.state_dict(), head_path)
                    
        print(f"[RANK: {self.rank}] " + ('=' * 60), flush=True)
        print(f"[RANK: {self.rank}] "'Training process finished!', flush=True)
        print(f"[RANK: {self.rank}] " + ('=' * 60), flush=True)

   
    
def run(rank, world_size, dul_args):
    setup(rank, world_size)
    dul_train = DUL_Trainer_dist(rank, world_size, dul_args)
    dul_train._dul_runner()
    cleanup()
    
if __name__ == '__main__':
    dul_args = dul_args_func()
    gpus = [int(item) for item in dul_args.gpu_id]
    world_size = len(gpus)
    print(f'World size: {world_size}')
    mp.spawn(run,
             args=(world_size, dul_args),
             nprocs=world_size,
             join=True)
    
