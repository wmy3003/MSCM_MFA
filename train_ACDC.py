import os
import logging
import argparse
import sys
import yaml
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time
from models.mscm_mfa import net
import os

import numpy as np
from tqdm import tqdm
from medpy.metric import dc, hd95
from scipy.ndimage import zoom

from utils.utils import powerset, test_single_volume
from utils.utils import DiceLoss, calculate_dice_percase, val_single_volume
from utils.dataset_ACDC import ACDCdataset, RandomGenerator


# from lib.networks import PVT_GCASCADE, MERIT_GCASCADE
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument('--skip_aggregation', default='additive',
                    help='Type of skip-aggregation: additive or concatenation')
parser.add_argument("--batch_size", default=6, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=400)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="./model_pth/ACDC")
parser.add_argument("--gpu_id", default='1')
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=2222, help='random seed')
parser.add_argument('--n_skip', type=int, default=3, help='using how many previous frames as skip-connection')
args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.cuda.empty_cache()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

args.is_pretrain = True
args.exp = 'MSCM_MFA' + str(args.img_size)
snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'MSCM_MFA')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path


if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

net = net()


train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
transforms.Compose(
    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)
db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

net = net.cuda()
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
macs, params = get_model_complexity_info(net, (1, args.img_size, args.img_size), as_strings=True,
                                         print_per_layer_stat=True, verbose=True)

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
net.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
save_interval = args.n_skip

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.80
Best_test_dcs = 0.80
logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)


def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
                i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1],
                np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class (%d) mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (
                i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        mean_jacard = np.mean(metric_list, axis=0)[2]
        mean_asd = np.mean(metric_list, axis=0)[3]
        logging.info(
            'Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (
                performance, mean_hd95, mean_jacard, mean_asd))
        logging.info("Testing Finished!")
        return performance, mean_hd95, mean_jacard, mean_asd


def val():
    logging.info("Validation ===>")
    dc_sum = 0
    metric_list = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch, val_label_batch = val_image_batch.squeeze(0).cpu().detach().numpy(), val_label_batch.squeeze(
            0).cpu().detach().numpy()
        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3)
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()

        P = net(val_image_batch)

        val_outputs = P

        val_outputs = torch.softmax(val_outputs, dim=1)
        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()

        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        else:
            val_outputs = val_outputs
        dc_sum += dc(val_outputs, val_label_batch[:])
    performance = dc_sum / len(valloader)
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))

    print('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    return performance


l = [0, 1, 2, 3]
ss = [x for x in powerset(l)]
# ss = [[0],[1],[2],[3]]
print(ss)    # logging.info("save model to {}".format(save_model_path))

for epoch in iterator:
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        P = net(image_batch)

        loss = 0.0
        lc1, lc2 = 0.3, 0.7

        loss_ce = ce_loss(P, label_batch[:].long())
        loss_dice = dice_loss(P, label_batch, softmax=True)
        loss += (lc1 * loss_ce + lc2 * loss_dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1
        if iter_num % 50 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss / len(train_dataset))
    logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
    print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))

    save_model_path = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path)

    avg_dcs = val()

    if avg_dcs >= Best_dcs:
        save_model_path = os.path.join(snapshot_path, 'best.pth')
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        Best_dcs = avg_dcs

        avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
        print("test avg_dsc: %f" % (avg_test_dcs))
        logging.info("test avg_dsc: %f" % (avg_test_dcs))
        Test_Accuracy.append(avg_test_dcs)
        if (Best_test_dcs <= avg_test_dcs):
            Best_test_dcs = avg_test_dcs
            save_model_path = os.path.join(snapshot_path, 'test_best.pth')
            torch.save(net.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            print("save model to {}".format(save_model_path))

    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("save model to {}".format(save_model_path))
        print("save model to {}".format(save_model_path))
        iterator.close()
        break
