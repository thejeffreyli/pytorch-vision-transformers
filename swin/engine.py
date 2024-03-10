# modifying https://github.com/WangFeng18/Swin-Transformer
"""
Train and eval functions used in main.py
"""
import math
import sys
import torch
from timm.data import Mixup
from timm.utils import accuracy
import utils

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, max_norm = 0,
                    mixup_fn = None, set_training_mode=True, reconstruct=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if (reconstruct):
            # restrict to [0,1]
            b, c, h, w = samples.size()
            samples = samples.view(samples.size(0), -1)
            samples -= samples.min(1, keepdim=True)[0]
            samples /= samples.max(1, keepdim=True)[0]
            samples = samples.view(b, c, h, w)


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # print(targets.size())
        with torch.cuda.amp.autocast():
            if (reconstruct):
                # print('YES')
                outputs, mu, sigma = model(samples)
                loss, reproduction_loss, KLD = criterion(samples, outputs, mu, sigma)
                loss /= (b*c*h*w)
                reproduction_loss /= (b*c*h*w)
                KLD /= (b*c*h*w)
            elif (not reconstruct):
                outputs = model(samples)
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Added to retrieve the accuracy
        if (not reconstruct):
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = samples.shape[0]
            # print(f'batch size train: {batch_size}')
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        elif (reconstruct):
            metric_logger.meters['reproduction_loss'].update(reproduction_loss.item())
            metric_logger.meters['KLD'].update(KLD.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, reconstruct=False):
    if (not reconstruct):
        criterion = torch.nn.CrossEntropyLoss()
    elif (reconstruct):
        criterion = utils.loss_function_reconstruct()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if (reconstruct):
            # restrict to [0,1]
            b, c, h, w = images.size()
            images = images.view(images.size(0), -1)
            images -= images.min(1, keepdim=True)[0]
            images /= images.max(1, keepdim=True)[0]
            images = images.view(b, c, h, w)
        
        # compute output
        with torch.cuda.amp.autocast():
            if (reconstruct):
                outputs, mu, sigma = model(images)
                loss, reproduction_loss, KLD = criterion(images, outputs, mu, sigma)
                loss /= (b*c*h*w)
                reproduction_loss /= (b*c*h*w)
                KLD /= (b*c*h*w)
            elif (not reconstruct):
                output = model(images)
                loss = criterion(output, target)

        if (not reconstruct):
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            # print(f'batch size test: {batch_size}')
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        elif (reconstruct):
            metric_logger.meters['reproduction_loss'].update(reproduction_loss.item())
            metric_logger.meters['KLD'].update(KLD.item())
        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
