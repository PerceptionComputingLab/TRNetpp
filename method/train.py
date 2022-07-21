from torch.utils.data import DataLoader
import numpy as np
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

from method.model import trnet_pp
import method.eval as evalu
from config import opt
import dataset.dataset as dataset
import features as feat


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class Loss_Saver:
    def __init__(self, moving=False):
        self.loss_list, self.last_loss = [], 0.0
        self.moving = moving
        return

    def updata(self, value):

        if not self.moving:
            self.loss_list += [value]
        elif not self.loss_list:
            self.loss_list += [value]
            self.last_loss = value
        else:
            update_val = self.last_loss * 0.9 + value * 0.1
            self.loss_list += [update_val]
            self.last_loss = update_val
        return

    def loss_drawing(self, root_file):
        print(self.loss_list)
        self.loss_list.to_csv(f'{root_file}loss.csv')
        return


def train(train_num_fold=0):
    model = trnet_pp(in_channels=opt.in_channels,
                     local_proj_shape=opt.local_proj_shape, local_dim_hidden=opt.local_dim_hidden,
                     local_num_layers=opt.local_num_layers, local_num_heads=opt.local_num_heads,
                     local_head_dim=opt.local_head_dim, local_patch_shape=opt.local_patch_shape,
                     local_switch_position=opt.local_switch_position,

                     global_dim_seq=opt.global_dim_seq, global_num_heads=opt.global_num_heads,
                     global_head_dim=opt.global_head_dim, global_num_encoders=opt.global_num_encoders,

                     CLS_num_linear=opt.CLS_num_linear, CLS_num_class=opt.CLS_num_class)
    if opt.use_gpu:
        model = model.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_LR[0], gamma=opt.decay_LR[1])

    train_dataset = dataset.cubic_sequence_data(pattern='train', num_fold=train_num_fold, num_class=opt.CLS_num_class)
    train_dataLoader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    eval_dataset = dataset.cubic_sequence_data(pattern='eval', num_fold=train_num_fold, num_class=opt.CLS_num_class)
    eval_dataLoader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=False)

    time = datetime.now()
    name_exp = f'{str(time.month).zfill(2)}{str(time.day).zfill(2)}_{str(time.hour).zfill(2)}' \
               f'{str(time.minute).zfill(2)}_fold{str(train_num_fold).zfill(2)}'

    feat.create_root(f'{opt.exp_file_root}/{name_exp}/')
    root_param = f'{opt.exp_file_root}/{name_exp}/param/'
    feat.create_root(root_param)

    logger = get_logger(f'{opt.exp_file_root}/{name_exp}/exp.log')
    logger.info('start training!')
    losssaver, max_acc = Loss_Saver(), 0.0

    for epoch in range(opt.max_epoch):

        model.train()
        epoch_loss = feat.Counter()
        for batch_id, (sequence_image, sequence_label) in tqdm(enumerate(train_dataLoader),
                                                               total=int(len(train_dataset) / opt.batch_size)):
            if sequence_image.shape[0] < opt.batch_size:
                continue

            Input, target = sequence_image.requires_grad_(), sequence_label

            if opt.use_gpu:
                Input, target = Input.cuda(), target.cuda()

            pred = model(Input)
            pred, target = pred.view(-1, 2), target.view(-1)

            loss = loss_fn(pred, target)

            epoch_loss.updata(float(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losssaver.updata(epoch_loss.avg)

        model.eval()
        indexs = evalu.print_evaluate_index(model, eval_dataLoader, num_indexes=opt.train_index)

        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc:{:.1f}'.format(epoch, opt.max_epoch, epoch_loss.avg, indexs[0]))
        scheduler.step()

        if max_acc < indexs[0]:
            torch.save(model.state_dict(), f'{root_param}BEP{epoch}_ACC{round(indexs[0], 1)}.pkl')
            max_acc = indexs[0]
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{root_param}EP{epoch}_ACC{round(indexs[0], 1)}.pkl')

    losssaver.loss_drawing(f'{opt.exp_file_root}/{name_exp}/')
    logger.info('finish training!')

    return
