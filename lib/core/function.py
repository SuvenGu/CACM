
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch

from core.evaluate import accuracy
import numpy as np
from utils.SupCon import SupConLoss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from core.evaluate import F1Score,Recall,Precision

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict,cls_num_list=None):
    num_classes = config.MODEL.NUM_CLASSES
    class_names = config.MODEL.CLASS_NAMES
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    cls_losses = AverageMeter()
    con_losses = AverageMeter()
    cm = torch.zeros((num_classes,num_classes))

    criterion_con = SupConLoss(temperature=config.TEMPERATURE)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, cond) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output,F,x_rec,_ = model(input,cond)
        target = target.cuda(non_blocking=True)

        loss_c = criterion(output, target) 

        loss_con = 0

        if config.LOSS_CON>0:
          f_embed =F.unsqueeze(1) 
          norms = torch.norm(f_embed, p=2, dim=2, keepdim=True)
          normalized_features = torch.div(f_embed, norms)
          loss_con = config.LOSS_CON*criterion_con(normalized_features,target.reshape(-1))

        loss = loss_c +loss_con

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        con_losses.update(loss_con, input.size(0))
        cls_losses.update(loss_c, input.size(0))


        prec1 = accuracy(output, target)
        top1.update(prec1[0].item(), input.size(0))

        output = output.argmax(dim=1)
        cm = cm+confusion_matrix(target.cpu(),output.cpu(),labels=[0,1,2,3,4])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % config.PRINT_FREQ == 0:

            recalls = Recall(cm)
            pres = Precision(cm)
            f1s = F1Score(cm)

            # valid class
            idx = torch.nonzero(cm.sum(dim=1)).squeeze()

            m_f1 = (f1s[idx]).mean()
            m_pre = (pres[idx]).mean()
            m_rec  = (recalls[idx]).mean()
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t' \
                'F1 {avg_f1:.3f}\t'\
                'Precision {avg_precision:.3f}\t'\
                  'Recall {avg_recall:.3f}\t'\
                    .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1,avg_f1= m_f1,avg_precision = m_pre, avg_recall = m_rec)
            logger.info(msg)

            msg = '====' \
            'F1 {f1s}\t' \
            'Precision {precisions}\t' \
            'Recall {recalls}\t'.format(
            f1s=' '.join([f'{class_names[i]}:{f1s[i]:.4f}' for i in idx]),
            precisions=' '.join([f'{class_names[i]}:{pres[i]:.4f}' for i in idx]),
            recalls=' '.join([f'{class_names[i]}:{recalls[i]:.4f}' for i in idx]),
        ) 
            logger.info(msg)


            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer.add_scalar('train_F1', m_f1, global_steps)

                writer.add_scalar('cls_loss', cls_losses.val, global_steps)
                writer.add_scalar('con_loss', con_losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1




def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
             writer_dict=None):
    num_classes = config.MODEL.NUM_CLASSES
    class_names = config.MODEL.CLASS_NAMES
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cm = torch.zeros((num_classes,num_classes))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, cond) in enumerate(val_loader):
            # compute output
            output,_,_,_= model(input,cond)

            target = target.cuda(non_blocking=True)

            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            prec1 = accuracy(output, target)
            top1.update(prec1[0].item(), input.size(0))

            output = output.argmax(dim=1)
            cm = cm+confusion_matrix(target.cpu(),output.cpu(),labels=[0,1,2,3,4])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        print(output ==target)
        msg = cm
        logger.info(msg)
        recalls = Recall(cm)
        pres = Precision(cm)
        f1s = F1Score(cm)

        idx = torch.nonzero(cm.sum(dim=1)).squeeze()
        print(class_names)

        m_f1 = (f1s[idx]).mean()
        m_pre = (pres[idx]).mean()
        m_rec  = (recalls[idx]).mean()
        msg = '******************' \
            'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy {top1.avg:.4f}\t' \
                'F1 {avg_f1:.4f}\t'\
                'Precision {avg_precision:.4f}\t'\
                  'Recall {avg_recall:.4f}\t'\
                    .format(
                  batch_time=batch_time, loss=losses, top1=top1,  avg_f1= m_f1,avg_precision = m_pre, avg_recall = m_rec)
        logger.info(msg)

        msg = '====' \
        'F1 {f1s}\t' \
        'Precision {precisions}\t' \
        'Recall {recalls}\t'\
         '******************'\
            .format(
            f1s=' '.join([f'{class_names[i]}:{f1s[i]:.4f}' for i in idx]),
            precisions=' '.join([f'{class_names[i]}:{pres[i]:.4f}' for i in idx]),
            recalls=' '.join([f'{class_names[i]}:{recalls[i]:.4f}' for i in idx]),
        ) 
        logger.info(msg)


        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', top1.avg, global_steps)
            writer.add_scalar('valid_F1', m_f1, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return m_f1

def predict(config, val_loader, model,cm):
    num_classes = config.MODEL.NUM_CLASSES
    class_names = config.MODEL.CLASS_NAMES
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    yt_pred_batch_list = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, cond) in enumerate(val_loader):
            # compute output
            output,_,_,_= model(input,cond)

            target = target.cuda(non_blocking=True)

            output = output.argmax(dim=1)
            cm = cm+confusion_matrix(target.cpu(),output.cpu(),labels=[0,1,2,3,4])
            yt_pred_batch_list.append(output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        y_pred = torch.cat(
            yt_pred_batch_list, dim=0
        ).cpu().numpy()  


    return y_pred,cm


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
