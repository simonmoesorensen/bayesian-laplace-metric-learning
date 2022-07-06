import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from config import Backbone_Dict, parse_args
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax, Softmax
from loss.focal import FocalLoss
from util.utils import (
    l2_norm,
    make_weights_for_balanced_classes,
    separate_batchnorm_params,
    warm_up_lr,
    schedule_lr,
    get_time,
    AverageMeter,
    accuracy,
    add_gaussian_noise,
)

from tensorboardX import SummaryWriter, writer
import os
import time


class DUL_Trainer:
    def __init__(self, dul_args):
        self.dul_args = dul_args
        self.dul_args.gpu_id = [int(item) for item in self.dul_args.gpu_id]

    def _report_configurations(self):
        print("=" * 60)
        print("Experiment time: ", get_time())
        print("=" * 60)
        print("Overall Configurations:")
        print("=" * 60)
        for k in self.dul_args.__dict__:
            print(" '{}' : '{}' ".format(k, str(self.dul_args.__dict__[k])))
        os.makedirs(self.dul_args.model_save_folder, exist_ok=True)
        os.makedirs(self.dul_args.log_tensorboard, exist_ok=True)
        writer = SummaryWriter(self.dul_args.log_tensorboard)
        return writer

    def _data_loader(self):
        time_start = time.perf_counter()

        print("=" * 60)
        print("Loading Data...")
        if self.dul_args.dataset == "MNIST":
            dataset_train = datasets.MNIST(
                root=self.dul_args.data_dir,
                train=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(), 
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                ),
                download=True,
            )

            num_class = 10

        else:
            raise NotImplementedError(
                f"Dataset {self.dul_args.dataset} is not implemented"
            )

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.dul_args.batch_size,
            pin_memory=self.dul_args.pin_memory,
            num_workers=self.dul_args.num_workers,
            drop_last=self.dul_args.drop_last,
        )

        print("Data loaded in {:.2f} seconds".format(time.perf_counter() - time_start))
        print("Number of Training Classes: '{}' ".format(num_class))

        return train_loader, num_class

    def _model_loader(self, num_class, data_size):
        # ----- backbone generate
        BACKBONE = Backbone_Dict[self.dul_args.dataset]
        print("=" * 60)
        print("Backbone Generated: '{}' ".format(self.dul_args.dataset))

        # ----- head generate
        Head_Dict = {
            "ArcFace": ArcFace(
                in_features=self.dul_args.embedding_size,
                out_features=num_class,
                device_id=self.dul_args.gpu_id,
                s=self.dul_args.arcface_scale,
                m=self.dul_args.arcface_margin,
            ),
            "CosFace": CosFace(
                in_features=self.dul_args.embedding_size,
                out_features=num_class,
                device_id=self.dul_args.gpu_id,
            ),
            "SphereFace": SphereFace(
                in_features=self.dul_args.embedding_size,
                out_features=num_class,
                device_id=self.dul_args.gpu_id,
            ),
            "Am_softmax": Am_softmax(
                in_features=self.dul_args.embedding_size,
                out_features=num_class,
                device_id=self.dul_args.gpu_id,
            ),
            "Softmax": Softmax(
                in_features=self.dul_args.embedding_size,
                out_features=num_class,
                device_id=self.dul_args.gpu_id,
            ),
        }
        HEAD = Head_Dict[self.dul_args.head_name]
        print("=" * 60)
        print("Head Generated: '{}' ".format(self.dul_args.head_name))

        # ----- loss generate
        Loss_Dict = {
            "Focal": FocalLoss(),
            "Softmax": nn.CrossEntropyLoss(),
        }
        LOSS = Loss_Dict[self.dul_args.loss_name]
        print("=" * 60)
        print("Loss Generated: '{}' ".format(self.dul_args.loss_name))
        # ----- separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_batchnorm_params(BACKBONE)
        _, head_paras_wo_bn = separate_batchnorm_params(HEAD)

        # ----- optimizer generate
        Optimizer_Dict = {
            "SGD": optim.SGD(
                [
                    {
                        "params": backbone_paras_wo_bn + head_paras_wo_bn,
                        "weight_decay": self.dul_args.weight_decay,
                    },
                    {"params": backbone_paras_only_bn},
                ],
                lr=self.dul_args.lr,
                momentum=self.dul_args.momentum,
            ),
            "Adam": optim.Adam(
                [
                    {
                        "params": backbone_paras_wo_bn + head_paras_wo_bn,
                        "weight_decay": self.dul_args.weight_decay,
                    },
                    {"params": backbone_paras_only_bn},
                ],
                lr=self.dul_args.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
            ),
        }
        OPTIMIZER = Optimizer_Dict[self.dul_args.optimizer]

        # Use Triangular Learning Rate Policy
        epoch = data_size // self.dul_args.batch_size
        SCHEDULER = optim.lr_scheduler.CyclicLR(
            OPTIMIZER,
            base_lr=self.dul_args.base_lr,
            max_lr=self.dul_args.max_lr,
            step_size_up=epoch
            * 2,  # Recommended by https://ieeexplore.ieee.org/document/7926641
            mode="triangular",
        )

        print("=" * 60)
        print("Optimizer Generated: '{}' ".format(self.dul_args.optimizer))
        print(OPTIMIZER)

        # ----- optional resume
        if self.dul_args.resume_backbone or self.dul_args.resume_head:
            print("=" * 60)
            if os.path.isfile(self.dul_args.resume_backbone):
                print(
                    "Loading Backbone Checkpoint '{}'".format(
                        self.dul_args.resume_backbone
                    )
                )
                BACKBONE.load_state_dict(torch.load(self.dul_args.resume_backbone))
            if os.path.isfile(self.dul_args.resume_head):
                print("Loading Head Checkpoint '{}'".format(self.dul_args.resume_head))
                HEAD.load_state_dict(torch.load(self.dul_args.resume_head))

            print("Resuming from checkpoints")
        else:
            print(
                "No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(
                    self.dul_args.resume_backbone, self.dul_args.resume_head
                )
            )

        # ----- multi-gpu or single-gpu
        if self.dul_args.multi_gpu:
            BACKBONE = nn.DataParallel(BACKBONE, device_ids=self.dul_args.gpu_id).cuda()
            HEAD = HEAD.cuda()
            LOSS = LOSS.cuda()
        else:
            BACKBONE = BACKBONE.cuda()
            HEAD = HEAD.cuda()
            LOSS = LOSS.cuda()

        return BACKBONE, HEAD, LOSS, OPTIMIZER, SCHEDULER

    def _dul_runner(self):
        writer = self._report_configurations()

        train_loader, num_class = self._data_loader()

        BACKBONE, HEAD, LOSS, OPTIMIZER, SCHEDULER = self._model_loader(
            num_class=num_class, data_size=len(train_loader)
        )

        DISP_FREQ = len(train_loader) // 100  # frequency to display training loss & acc
        NUM_EPOCH_WARM_UP = self.dul_args.warm_up_epoch
        NUM_BATCH_WARM_UP = int(len(train_loader) * NUM_EPOCH_WARM_UP)
        batch = 0  # batch index

        print("=" * 60)
        print("Display Freqency: '{}' ".format(DISP_FREQ))
        print(
            f"Warm Up Epoch: '{NUM_EPOCH_WARM_UP}', Warm Up Batch: '{NUM_BATCH_WARM_UP}'"
        )
        print("Start Training: ")

        for epoch in range(self.dul_args.num_epoch):
            if epoch < self.dul_args.resume_epoch:
                continue

            BACKBONE.train()  # set to training mode
            HEAD.train()
            BACKBONE.training = True

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            losses_KL = AverageMeter()

            for inputs, labels in tqdm(train_loader):
                OPTIMIZER.zero_grad()
                should_warmup = (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                    batch + 1 <= NUM_BATCH_WARM_UP
                )

                if should_warmup:  # adjust LR for each training batch during warm up
                    warm_up_lr(
                        batch + 1, NUM_BATCH_WARM_UP, self.dul_args.lr, OPTIMIZER
                    )

                inputs = inputs.cuda()
                labels = labels.cuda().long()
                loss = 0

                mu_dul, std_dul = BACKBONE(inputs)  # namely, mean and std

                epsilon = torch.randn_like(std_dul)
                features = mu_dul + epsilon * std_dul
                variance_dul = std_dul**2

                loss_kl = (
                    ((variance_dul + mu_dul**2 - torch.log(variance_dul) - 1) * 0.5)
                    .sum(dim=-1)
                    .mean()
                )
                losses_KL.update(loss_kl.item(), inputs.size(0))
                loss += self.dul_args.kl_scale * loss_kl

                features = l2_norm(features)
                outputs = HEAD(features, labels)

                loss_head = LOSS(outputs, labels)

                loss += loss_head

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                losses.update(loss_head.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                loss.backward()
                OPTIMIZER.step()

                # Adjust LR using Triangular policy when it's not in warmup phase
                if not (should_warmup):
                    SCHEDULER.step()

                # dispaly training loss & acc every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60, flush=True)
                    print(
                        "Epoch {}/{} Batch (Step) {}/{}\t"
                        "Time {}\t"
                        "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "Training Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t"
                        "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                            epoch + 1,
                            self.dul_args.num_epoch,
                            batch + 1,
                            len(train_loader) * self.dul_args.num_epoch,
                            time.asctime(time.localtime(time.time())),
                            loss=losses,
                            loss_KL=losses_KL,
                            top1=top1,
                            top5=top5,
                        ),
                        flush=True,
                    )

                batch += 1  # batch index
            # training statistics per epoch (buffer for visualization)
            epoch_loss = losses.avg
            epoch_acc = top1.avg
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
            print("=" * 60, flush=True)
            print(
                "Epoch: {}/{}\t"
                "Training Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch + 1,
                    self.dul_args.num_epoch,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                ),
                flush=True,
            )

            # ----- save model
            if (
                epoch == self.dul_args.num_epoch - 1
            ) or epoch % self.dul_args.save_freq == 0:
                print("=" * 60, flush=True)
                print("Saving NO.EPOCH {} trained model".format(epoch + 1), flush=True)
                if self.dul_args.multi_gpu:
                    torch.save(
                        BACKBONE.module.state_dict(),
                        os.path.join(
                            self.dul_args.model_save_folder,
                            "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                self.dul_args.dataset, epoch + 1, batch, get_time()
                            ),
                        ),
                    )
                    torch.save(
                        HEAD.state_dict(),
                        os.path.join(
                            self.dul_args.model_save_folder,
                            "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                self.dul_args.head_name, epoch + 1, batch, get_time()
                            ),
                        ),
                    )
                else:
                    torch.save(
                        BACKBONE.state_dict(),
                        os.path.join(
                            self.dul_args.model_save_folder,
                            "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                self.dul_args.dataset, epoch + 1, batch, get_time()
                            ),
                        ),
                    )
                    torch.save(
                        HEAD.state_dict(),
                        os.path.join(
                            self.dul_args.model_save_folder,
                            "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                                self.dul_args.head_name, epoch + 1, batch, get_time()
                            ),
                        ),
                    )

        print("=" * 60, flush=True)
        print("Training process finished!", flush=True)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    dul_train = DUL_Trainer(parse_args())
    dul_train._dul_runner()
