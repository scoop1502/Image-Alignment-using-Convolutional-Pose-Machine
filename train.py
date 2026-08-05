import torch
import argparse
import torchvision
import utils
import os
import datetime
from dataset import KPDataset
from model import KeypointModel


def main(args):
    utils.init_distributed_mode(args)

    if args.rank == 0:
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        logger_path = f"{args.ckpt_dir}/log.txt"
    else:
        logger_path = None

    logger = utils.TextLogger(logger_path)



    model = KeypointModel()
    model.cuda()


    train_dataset = KPDataset(root=args.train_data, split="train")
    val_dataset  = KPDataset(root=args.val_data, split="test")


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    train_sampler =  torch.utils.data.DistributedSampler(train_dataset, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        num_workers=args.num_workers,
        pin_memory = True,
        drop_last=True,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = args.batch_size,
        num_workers=args.num_workers,
        pin_memory = True,
    )


    start_epoch = 0

    optim = torch.optim.AdamW(
        model.parameters(),
        lr = args.lr,
        weight_decay = args.wd
    )

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size

    lr_scheduler = utils.cosine_scheduler(
                args.lr, args.min_lr, args.max_epoch, num_training_steps_per_epoch,
                warmup_epochs=args.warmup_epochs
    )
    lr_scheduler = None

    if args.wd_end is not None:
        wd_scheduler = utils.cosine_scheduler(
                    args.wd,
                    args.wd_end,
                    args.max_epoch, num_training_stesp_per_epoch
        )
    else:
        wd_scheduler = None


    if args.resume is not None and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        optim.load_state_dict(checkpoint["optim"])
        print(f"Loaded Model From {args.resume}")
        print(f"Resume Training From Epoch {start_epoch}")

    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        global_start = (epoch - 1) * num_training_steps_per_epoch
        train(epoch, model, optim, train_loader, args.max_epoch, logger, lr_scheduler=lr_scheduler, global_start=global_start, wd_scheduler=wd_scheduler)

        if epoch % args.eval_freq == 0 or epoch == args.max_epoch:
            validate(epoch, model, val_loader, args.max_epoch, logger)


        if args.rank == 0:
            model_path = f"{args.ckpt_dir}/epoch-latest.pth"
            save_model(model_path, model, optim, epoch)

        if (epoch % args.save_freq == 0 or epoch == args.max_epoch) and args.rank == 0:
            model_path = f"{args.ckpt_dir}/epoch-{epoch:04d}.pth"
            save_model(model_path, model, optim, epoch)




def save_model(model_path, model, optim, epoch, best_score=None):
    state_dict = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "optim": optim.state_dict(),
        "best_score": best_score
    }
    torch.save(state_dict, model_path)
    print(f"Saving model at epoch {epoch} to {model_path}")

def train(epoch, model, optim, loader, max_epoch=100, logger=None, lr_scheduler=None, global_start=0, wd_scheduler=None):
    loader.sampler.set_epoch(epoch)
    model.train()
    print(f"Start Training Epoch [{epoch}/{max_epoch}]")
    logger.log(f"Start Training Epoch [{epoch}/{max_epoch}]")


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('wd', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mse', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f"Train Epoch [{epoch}/{max_epoch}]"
    loss_cls_fn = torch.nn.CrossEntropyLoss()
    loss_lmk_fn = torch.nn.L1Loss(reduction='none')
    it = 0
    total_it = len(loader)
    for (images, targets, lmks, lmk_masks) in metric_logger.log_every(loader, 1, header):
        it += 1
        global_it = global_start + it - 1

        for i, param_group in enumerate(optim.param_groups):
            if lr_scheduler is not None:
                param_group["lr"] = lr_scheduler[global_it] #* param_group["lr_scale"]
            if wd_scheduler is not None:
                param_group["weight_decay"] = wd_scheduler[global_it]


        images = images.cuda(non_blocking=True)
        lmks = lmks.cuda(non_blocking=True)
        lmk_masks = lmk_masks.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        pre_cls, pre_lmk = model(images) #, texts)

        cls_loss = loss_cls_fn(pre_cls, targets)
        #print(pre_lmk.shape, lmks.shape, lmk_masks.shape)
        lmk_loss = (loss_lmk_fn(pre_lmk, lmks) * lmk_masks).mean()
        loss = cls_loss + 100 * lmk_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        acc1, _ = utils.accuracy(pre_cls, targets, topk=(1, 2))

        #torch.cuda.synchronize()
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optim.param_groups[0]["lr"])
        metric_logger.update(wd=optim.param_groups[0]["weight_decay"])

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['mse'].update(lmk_loss.item(), n=batch_size)

        cur_lr = optim.param_groups[0]["lr"]
        cur_wd = optim.param_groups[0]["weight_decay"]
        if (it % 20) == 0:
            logger.log(f"train epoch [{epoch}/{max_epoch}]. iter [{it}/{total_it}]. lr {cur_lr:0.4f}. wd: {cur_wd:0.4f}. loss {loss.item():0.5f}. acc@1 {acc1.item():0.5f}. mse: {lmk_loss.item():0.5f}")


    #metric_logger.synchronize_between_processes()

    logger.log('Finish {header} Acc@1 {top1.global_avg:.3f} mse {mse.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(header=header, top1=metric_logger.acc1, mse=metric_logger.mse, losses=metric_logger.loss))



@torch.no_grad()
def validate(epoch, model, loader, max_epoch=100, logger=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('mse', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f"Val Epoch [{epoch}/{max_epoch}]"
    loss_cls_fn = torch.nn.CrossEntropyLoss()
    loss_lmk_fn = torch.nn.L1Loss(reduction='none')

    it = 0
    total_it = len(loader)
    for (images, targets, lmks, lmk_masks) in metric_logger.log_every(loader, 20, header):
        it += 1

        images = images.cuda(non_blocking=True)
        lmks = lmks.cuda(non_blocking=True)
        lmk_masks = lmk_masks.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)


        pre_cls, pre_lmk = model(images) #, texts)
        cls_loss = loss_cls_fn(pre_cls, targets)
        lmk_loss = (loss_lmk_fn(pre_lmk, lmks) * lmk_masks).mean()
        loss = cls_loss + 100 * lmk_loss


        acc1, _ = utils.accuracy(pre_cls, targets, topk=(1, 2))

        #torch.cuda.synchronize()
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['mse'].update(lmk_loss.item(), n=batch_size)

        if (it % 20) == 0:
            logger.log(f"val epoch [{epoch}/{max_epoch}]. iter [{it}/{total_it}]. loss {loss.item():0.5f}. acc@1 {acc1.item():0.5f}. mse: {lmk_loss.item():0.5f}")



    logger.log('Finish {header} Acc@1 {top1.global_avg:.3f} mse {mse.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(header=header, top1=metric_logger.acc1, mse=metric_logger.mse, losses=metric_logger.loss))





if __name__  == "__main__":

    parser = argparse.ArgumentParser('slurm training')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--train_data", type=str, default="data/imagenet1k/train")
    parser.add_argument("--val_data", type=str, default="data/imagenet1k/val")
    parser.add_argument("--ckpt_dir", type=str, default="ckpt/ViT-B-32-imagenet/")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                                                help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--wd_end", type=float, default=None)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)





    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                                                 help='epochs to warmup LR, if scheduler supports')


    args = parser.parse_args()

    main(args)

