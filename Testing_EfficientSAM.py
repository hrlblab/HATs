import argparse
import os, sys
import pandas as pd

#sys.path.append("..")
sys.path.append("/Data4/HATs/EfficientSAM_token_dynamichead_logits")
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from scipy.ndimage import morphology
from matplotlib import cm

import skimage
import maxflow

# from unet2D_Dodnet import UNet2D as UNet2D
# from unet2D_Dodnet_ms_scalecontrol import UNet2D as UNet2D_ms_scalecontrol
import os.path as osp
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSValDataSet as MOTSValDataSet

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from sklearn import metrics
from math import ceil

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
from sklearn.metrics import f1_score, confusion_matrix

start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util_a.image_pool import ImagePool

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from unet2D_Dodnet_scale_token import UNet2D as UNet2D_scale

def smooth_segmentation_with_blur(logits, lambda_term=50, sigma=10.0):
    smoothed_mask = cv2.GaussianBlur(logits, (3, 3), 0)
    smoothed_mask[smoothed_mask >= 0.5] = 1.
    smoothed_mask[smoothed_mask != 1.] = 0.

    return smoothed_mask.astype(np.uint8)


def one_hot_3D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")
    parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_train_scale_aug_patch')


    parser.add_argument("--valset_dir", type=str, default='/Data/HATs/Data/test/data_list.csv')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/EfficientSAM_dynamichead_logits_HATs')
    parser.add_argument("--reload_path", type=str, default='snapshots_2D/EfficientSAM_dynamichead_logits_HATs/MOTS_DynConv_EfficientSAM_dynamichead_logits_HATs_e%d.pth' % (100))
    parser.add_argument("--best_epoch", type=int, default=100)

    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--edge_weight", type=float, default=1.2)
    # parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def count_score_only_two(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, 0., 0.


def surfd(input1, input2, sampling=1, connectivity=1):
    # input_1 = np.atleast_1d(input1.astype(bool))
    # input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input1.ndim, connectivity)

    S = input1 - morphology.binary_erosion(input1, conn)
    Sprime = input2 - morphology.binary_erosion(input2, conn)

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))


    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return np.max(sds), np.mean(sds)

def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_HD = 0
    Val_MSD = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        # pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        # label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        pred = preds[ki,:,rmin:rmax,cmin:cmax]
        label = labels[ki,:,rmin:rmax,cmin:cmax]

        # preds1 = sum(preds,[])
        # labels1 = sum(labels,[])
        #try:
        Val_DICE += dice_score(pred, label)
        # preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1, ...].flatten().detach().cpu().numpy()
        # labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

        if preds1.sum() == 0 and labels1.sum() == 0:
            Val_HD += 0
            Val_MSD += 0

        else:
            hausdorff, meansurfaceDistance = surfd(preds1, labels1)
            Val_HD += hausdorff
            Val_MSD += meansurfaceDistance

        Val_F1 += f1_score(preds1, labels1, average='macro')

        # cnf_matrix = confusion_matrix(preds1, labels1)
        #
        # try:
        #     FP = cnf_matrix[1,0]
        #     FN = cnf_matrix[0,1]
        #     TP = cnf_matrix[1,1]
        #     TN = cnf_matrix[0,0]
        # except:
        #     FP = np.array(1)
        #     FN = np.array(1)
        #     TP = np.array(1)
        #     TN = np.array(1)
        #
        # FP = FP.astype(float)
        # FN = FN.astype(float)
        # TP = TP.astype(float)
        # TN = TN.astype(float)
        #
        # Val_TPR += TP / (TP + FN)
        # Val_PPV += TP / (TP + FP)

        # except:
        #
        #     Val_DICE += 1.
        #     Val_F1 += 1.
        #     Val_TPR += 1.
        #     Val_PPV += 1.

    return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()

def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))
    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
        cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)


def TAL_pred(preds, task_id):
    if task_id == 0:
        preds_p2 = preds[:, 0:1, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 1:
        preds_p2 = preds[:, 1:2, :, :].clone()
        preds_p1 = preds[:, 0:1, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 2:
        preds_p2 = preds[:, 2:3, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 3:
        preds_p2 = preds[:, 3:4, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    elif task_id == 4:
        preds_p2 = preds[:, 4:5, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 0:1, :, :].clone() + preds[:, 5:6, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    else:
        preds_p2 = preds[:, 5:6, :, :].clone()
        preds_p1 = preds[:, 1:2, :, :].clone() + preds[:, 2:3, :, :].clone() + preds[:, 3:4, :, :].clone() + preds[:, 4:5, :, :].clone() + preds[:, 0:1, :, :].clone()
        new_preds = torch.cat((preds_p1, preds_p2), 1)

    return new_preds


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        model = build_efficient_sam_vits(task_num = 15, scale_num = 4)

        # PrPSeg backbone
        model = UNet2D_scale(num_classes=15, num_scale = 4, weight_std=False)

        check_wo_gpu = 0

        print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

        if not check_wo_gpu:
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not check_wo_gpu:
            weights = [1., 1.]
            class_weights = torch.FloatTensor(weights).cuda()
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
            loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).to(device)

        else:
            weights = [1., 1.]
            class_weights = torch.FloatTensor(weights)
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
            loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        valloader = DataLoader(
            MOTSValDataSet(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=4,shuffle=False,num_workers=8)

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = np.inf
        batch_size = args.batch_size
        task_num = 15

        model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

        model.eval()
        # semi_pool_image = ImagePool(8 * 6)
        task0_pool_image = ImagePool(8)
        task0_pool_mask = ImagePool(8)
        task0_scale = []
        task0_layer = []
        task0_name = []

        task1_pool_image = ImagePool(8)
        task1_pool_mask = ImagePool(8)
        task1_scale = []
        task1_layer = []
        task1_name = []

        task2_pool_image = ImagePool(8)
        task2_pool_mask = ImagePool(8)
        task2_scale = []
        task2_layer = []
        task2_name = []

        task3_pool_image = ImagePool(8)
        task3_pool_mask = ImagePool(8)
        task3_scale = []
        task3_layer = []
        task3_name = []

        task4_pool_image = ImagePool(8)
        task4_pool_mask = ImagePool(8)
        task4_scale = []
        task4_layer = []
        task4_name = []

        task5_pool_image = ImagePool(8)
        task5_pool_mask = ImagePool(8)
        task5_scale = []
        task5_layer = []
        task5_name = []

        task6_pool_image = ImagePool(8)
        task6_pool_mask = ImagePool(8)
        task6_scale = []
        task6_layer = []
        task6_name = []

        task7_pool_image = ImagePool(8)
        task7_pool_mask = ImagePool(8)
        task7_scale = []
        task7_layer = []
        task7_name = []


        'extend to 15 classes'
        task8_pool_image = ImagePool(8)
        task8_pool_mask = ImagePool(8)
        task8_scale = []
        task8_layer = []
        task8_name = []

        task9_pool_image = ImagePool(8)
        task9_pool_mask = ImagePool(8)
        task9_scale = []
        task9_layer = []
        task9_name = []

        task10_pool_image = ImagePool(8)
        task10_pool_mask = ImagePool(8)
        task10_scale = []
        task10_layer = []
        task10_name = []

        task11_pool_image = ImagePool(8)
        task11_pool_mask = ImagePool(8)
        task11_scale = []
        task11_layer = []
        task11_name = []

        task12_pool_image = ImagePool(8)
        task12_pool_mask = ImagePool(8)
        task12_scale = []
        task12_layer = []
        task12_name = []

        task13_pool_image = ImagePool(8)
        task13_pool_mask = ImagePool(8)
        task13_scale = []
        task13_layer = []
        task13_name = []

        task14_pool_image = ImagePool(8)
        task14_pool_mask = ImagePool(8)
        task14_scale = []
        task14_layer = []
        task14_name = []

        val_loss = np.zeros((task_num))
        val_F1 = np.zeros((task_num))
        val_Dice = np.zeros((task_num))
        val_HD = np.zeros((task_num))
        val_MSD = np.zeros((task_num))
        cnt = np.zeros((task_num))

        single_df_0 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_1 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_2 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_3 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_4 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_5 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_6 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_7 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_8 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_9 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_10 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_11 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_12 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_13 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_14 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])

        # layer_num = [0,2,6]
        layer_num = [0,5,12]

        # for iter, batch1, batch2 in enumerate(zip(valloaderloader, semi_valloaderloader)):
        with torch.no_grad():
            for iter, batch1 in enumerate(valloader):

                'dataloader'
                imgs = batch1[0].cuda()
                lbls = batch1[1].cuda()
                wt = batch1[2].cuda().float()
                volumeName = batch1[3]
                l_ids = batch1[4].cuda()
                t_ids = batch1[5].cuda()
                s_ids = batch1[6].cuda()

                # semi_img = batch2[0]

                for ki in range(len(imgs)):
                    now_task = layer_num[l_ids[ki]] + t_ids[ki]

                    if now_task != 12:
                        continue

                    if now_task == 0:
                        task0_pool_image.add(imgs[ki].unsqueeze(0))
                        task0_pool_mask.add(lbls[ki].unsqueeze(0))
                        task0_scale.append((s_ids[ki]))
                        task0_layer.append((l_ids[ki]))
                        task0_name.append((volumeName[ki]))
                    elif now_task == 1:
                        task1_pool_image.add(imgs[ki].unsqueeze(0))
                        task1_pool_mask.add(lbls[ki].unsqueeze(0))
                        task1_scale.append((s_ids[ki]))
                        task1_layer.append((l_ids[ki]))
                        task1_name.append((volumeName[ki]))
                    elif now_task == 2:
                        task2_pool_image.add(imgs[ki].unsqueeze(0))
                        task2_pool_mask.add(lbls[ki].unsqueeze(0))
                        task2_scale.append((s_ids[ki]))
                        task2_layer.append((l_ids[ki]))
                        task2_name.append((volumeName[ki]))
                    elif now_task == 3:
                        task3_pool_image.add(imgs[ki].unsqueeze(0))
                        task3_pool_mask.add(lbls[ki].unsqueeze(0))
                        task3_scale.append((s_ids[ki]))
                        task3_layer.append((l_ids[ki]))
                        task3_name.append((volumeName[ki]))
                    elif now_task == 4:
                        task4_pool_image.add(imgs[ki].unsqueeze(0))
                        task4_pool_mask.add(lbls[ki].unsqueeze(0))
                        task4_scale.append((s_ids[ki]))
                        task4_layer.append((l_ids[ki]))
                        task4_name.append((volumeName[ki]))
                    elif now_task == 5:
                        task5_pool_image.add(imgs[ki].unsqueeze(0))
                        task5_pool_mask.add(lbls[ki].unsqueeze(0))
                        task5_scale.append((s_ids[ki]))
                        task5_layer.append((l_ids[ki]))
                        task5_name.append((volumeName[ki]))
                    elif now_task == 6:
                        task6_pool_image.add(imgs[ki].unsqueeze(0))
                        task6_pool_mask.add(lbls[ki].unsqueeze(0))
                        task6_scale.append((s_ids[ki]))
                        task6_layer.append((l_ids[ki]))
                        task6_name.append((volumeName[ki]))
                    elif now_task == 7:
                        task7_pool_image.add(imgs[ki].unsqueeze(0))
                        task7_pool_mask.add(lbls[ki].unsqueeze(0))
                        task7_scale.append((s_ids[ki]))
                        task7_layer.append((l_ids[ki]))
                        task7_name.append((volumeName[ki]))

                        'extend to 15 classes'
                    elif now_task == 8:
                        task8_pool_image.add(imgs[ki].unsqueeze(0))
                        task8_pool_mask.add(lbls[ki].unsqueeze(0))
                        task8_scale.append((s_ids[ki]))
                        task8_layer.append((l_ids[ki]))
                        task8_name.append((volumeName[ki]))


                    elif now_task == 9:
                        task9_pool_image.add(imgs[ki].unsqueeze(0))
                        task9_pool_mask.add(lbls[ki].unsqueeze(0))
                        task9_scale.append((s_ids[ki]))
                        task9_layer.append((l_ids[ki]))
                        task9_name.append((volumeName[ki]))

                    elif now_task == 10:
                        task10_pool_image.add(imgs[ki].unsqueeze(0))
                        task10_pool_mask.add(lbls[ki].unsqueeze(0))
                        task10_scale.append((s_ids[ki]))
                        task10_layer.append((l_ids[ki]))
                        task10_name.append((volumeName[ki]))

                    elif now_task == 11:
                        task11_pool_image.add(imgs[ki].unsqueeze(0))
                        task11_pool_mask.add(lbls[ki].unsqueeze(0))
                        task11_scale.append((s_ids[ki]))
                        task11_layer.append((l_ids[ki]))
                        task11_name.append((volumeName[ki]))

                    elif now_task == 12:
                        task12_pool_image.add(imgs[ki].unsqueeze(0))
                        task12_pool_mask.add(lbls[ki].unsqueeze(0))
                        task12_scale.append((s_ids[ki]))
                        task12_layer.append((l_ids[ki]))
                        task12_name.append((volumeName[ki]))

                    elif now_task == 13:
                        task13_pool_image.add(imgs[ki].unsqueeze(0))
                        task13_pool_mask.add(lbls[ki].unsqueeze(0))
                        task13_scale.append((s_ids[ki]))
                        task13_layer.append((l_ids[ki]))
                        task13_name.append((volumeName[ki]))

                    elif now_task == 14:
                        task14_pool_image.add(imgs[ki].unsqueeze(0))
                        task14_pool_mask.add(lbls[ki].unsqueeze(0))
                        task14_scale.append((s_ids[ki]))
                        task14_layer.append((l_ids[ki]))
                        task14_name.append((volumeName[ki]))

                output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/', '/Data4/HATs/Testing_'),
                                             str(args.best_epoch))

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                optimizer.zero_grad()

                'medulla'
                if task0_pool_image.num_imgs >= batch_size:
                    images = task0_pool_image.query(batch_size)
                    labels = task0_pool_mask.query(batch_size)
                    now_task = torch.tensor(0)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task0_scale.pop(0)
                        layers[bi] = task0_layer.pop(0)
                        filename.append(task0_name.pop(0))

                    preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                    preds[:, :, :512, :512] = model(images[:, :, :512, :512], torch.ones(batch_size).cuda() * now_task,
                                                       scales)
                    preds[:, :, :512, 512:] = model(images[:, :, :512, 512:], torch.ones(batch_size).cuda() * now_task,
                                                       scales)
                    preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:], torch.ones(batch_size).cuda() * now_task,
                                                       scales)
                    preds[:, :, 512:, :512] = model(images[:, :, 512:, :512], torch.ones(batch_size).cuda() * now_task,
                                                       scales)



                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    rmin, rmax, cmin, cmax = 0, 1024, 0, 1024

                    labels_onehot = one_hot_3D(labels.long())

                    for pi in range(len(images)):
                        # prediction = now_preds[pi]
                        prediction = smooth_segmentation_with_blur(preds[pi,1].detach().cpu().numpy(), lambda_term=50, sigma=5.0)

                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                        row = len(single_df_0)
                        single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]


                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

                'cortex'
                if task1_pool_image.num_imgs >= batch_size:
                    images = task1_pool_image.query(batch_size)
                    labels = task1_pool_mask.query(batch_size)
                    now_task = torch.tensor(1)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task1_scale.pop(0)
                        layers[bi] = task1_layer.pop(0)
                        filename.append(task1_name.pop(0))

                    preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                    preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels.long())
                    rmin, rmax, cmin, cmax = 0, 1024, 0, 1024


                    for pi in range(len(images)):
                        # prediction = now_preds[pi]
                        prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                                       lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_1)
                        single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

                'cortex_in'
                if task2_pool_image.num_imgs >= batch_size:
                    images = task2_pool_image.query(batch_size)
                    labels = task2_pool_mask.query(batch_size)
                    now_task = torch.tensor(2)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task2_scale.pop(0)
                        layers[bi] = task2_layer.pop(0)
                        filename.append(task2_name.pop(0))

                    preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                    preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels.long())
                    rmin, rmax, cmin, cmax = 0, 1024, 0, 1024


                    for pi in range(len(images)):
                        # prediction = now_preds[pi]
                        prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                                       lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_2)
                        single_df_2.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1


                'cortex_middle'
                if task3_pool_image.num_imgs >= batch_size:
                    images = task3_pool_image.query(batch_size)
                    labels = task3_pool_mask.query(batch_size)
                    now_task = torch.tensor(3)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task3_scale.pop(0)
                        layers[bi] = task3_layer.pop(0)
                        filename.append(task3_name.pop(0))

                    preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                    preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels.long())
                    rmin, rmax, cmin, cmax = 0, 1024, 0, 1024


                    for pi in range(len(images)):
                        # prediction = now_preds[pi]
                        prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                                       lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_3)
                        single_df_3.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1


                'cortex_out'
                if task4_pool_image.num_imgs >= batch_size:
                    images = task4_pool_image.query(batch_size)
                    labels = task4_pool_mask.query(batch_size)
                    now_task = torch.tensor(4)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task4_scale.pop(0)
                        layers[bi] = task4_layer.pop(0)
                        filename.append(task4_name.pop(0))

                    preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                    preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                       torch.ones(batch_size).cuda() * now_task, scales)
                    preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                       torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels.long())
                    rmin, rmax, cmin, cmax = 0, 1024, 0, 1024


                    for pi in range(len(images)):
                        # prediction = now_preds[pi]
                        prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                                       lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_4)
                        single_df_4.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1


                'dt'
                if task5_pool_image.num_imgs >= batch_size:
                    images = task5_pool_image.query(batch_size)
                    labels = task5_pool_mask.query(batch_size)
                    now_task = torch.tensor(5)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task5_scale.pop(0)
                        layers[bi] = task5_layer.pop(0)
                        filename.append(task5_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 128, 384, 128, 384


                    for pi in range(len(images)):
                        #prediction = now_preds[pi, 128:384, 128:384]

                        prediction = smooth_segmentation_with_blur(preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                                                                       lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_5)
                        single_df_5.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

                'pt'
                if task6_pool_image.num_imgs >= batch_size:
                    images = task6_pool_image.query(batch_size)
                    labels = task6_pool_mask.query(batch_size)
                    now_task = torch.tensor(6)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task6_scale.pop(0)
                        layers[bi] = task6_layer.pop(0)
                        filename.append(task6_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 128, 384, 128, 384


                    for pi in range(len(images)):
                        #prediction = now_preds[pi, 128:384, 128:384]

                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_6)
                        single_df_6.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

                'cap'
                if task7_pool_image.num_imgs >= batch_size:
                    images = task7_pool_image.query(batch_size)
                    labels = task7_pool_mask.query(batch_size)
                    now_task = torch.tensor(7)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task7_scale.pop(0)
                        layers[bi] = task7_layer.pop(0)
                        filename.append(task7_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 128, 384, 128, 384


                    for pi in range(len(images)):
                        # prediction = now_preds[pi, 128:384, 128:384]

                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_7)
                        single_df_7.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

                'tuft'
                if task8_pool_image.num_imgs >= batch_size:
                    images = task8_pool_image.query(batch_size)
                    labels = task8_pool_mask.query(batch_size)
                    now_task = torch.tensor(8)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task8_scale.pop(0)
                        layers[bi] = task8_layer.pop(0)
                        filename.append(task8_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 128, 384, 128, 384


                    for pi in range(len(images)):
                        # prediction = now_preds[pi, 128:384, 128:384]

                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_8)
                        single_df_8.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1


                'art'
                if task9_pool_image.num_imgs >= batch_size:
                    images = task9_pool_image.query(batch_size)
                    labels = task9_pool_mask.query(batch_size)
                    now_task = torch.tensor(9)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task9_scale.pop(0)
                        layers[bi] = task9_layer.pop(0)
                        filename.append(task9_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 128, 384, 128, 384


                    for pi in range(len(images)):
                        # prediction = now_preds[pi, 128:384, 128:384]
                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)

                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_9)
                        single_df_9.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1


                'ptc'
                if task10_pool_image.num_imgs >= batch_size:
                    images = task10_pool_image.query(batch_size)
                    labels = task10_pool_mask.query(batch_size)
                    now_task = torch.tensor(10)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task10_scale.pop(0)
                        layers[bi] = task10_layer.pop(0)
                        filename.append(task10_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)


                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 128, 384, 128, 384


                    for pi in range(len(images)):
                        # prediction = now_preds[pi, 128:384, 128:384]

                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                                   prediction, cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_10)
                        single_df_10.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1


                'mv'
                if task11_pool_image.num_imgs >= batch_size:
                    images = task11_pool_image.query(batch_size)
                    labels = task11_pool_mask.query(batch_size)
                    now_task = torch.tensor(11)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task11_scale.pop(0)
                        layers[bi] = task11_layer.pop(0)
                        filename.append(task11_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 0, 512, 0, 512

                    for pi in range(len(images)):
                        # prediction = now_preds[pi]

                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(
                            os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                            prediction, cmap = cm.gray)


                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_11)
                        single_df_11.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1




                'pod'
                if task12_pool_image.num_imgs >= batch_size:
                    images = task12_pool_image.query(batch_size)
                    labels = task12_pool_mask.query(batch_size)
                    now_task = torch.tensor(12)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task12_scale.pop(0)
                        layers[bi] = task12_layer.pop(0)
                        filename.append(task12_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 0, 512, 0, 512

                    for pi in range(len(images)):
                        # prediction = now_preds[pi]
                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)

                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(
                            os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                            prediction, cmap = cm.gray)


                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_12)
                        single_df_12.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

                'mes'
                if task13_pool_image.num_imgs >= batch_size:
                    images = task13_pool_image.query(batch_size)
                    labels = task13_pool_mask.query(batch_size)
                    now_task = torch.tensor(13)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task13_scale.pop(0)
                        layers[bi] = task13_layer.pop(0)
                        filename.append(task13_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task, scales)

                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 0, 512, 0, 512

                    for pi in range(len(images)):
                        # prediction = now_preds[pi]
                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)

                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(
                            os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                            prediction, cmap=cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_13)
                        single_df_13.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

                'smooth'
                if task14_pool_image.num_imgs >= batch_size:
                    images = task14_pool_image.query(batch_size)
                    labels = task14_pool_mask.query(batch_size)
                    now_task = torch.tensor(14)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[bi] = task14_scale.pop(0)
                        layers[bi] = task14_layer.pop(0)
                        filename.append(task14_name.pop(0))

                    preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                                  scales)

                    now_preds = torch.argmax(preds, 1) == 1
                    now_preds_onehot = one_hot_3D(now_preds.long())

                    labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                    rmin, rmax, cmin, cmax = 0, 512, 0, 512

                    for pi in range(len(images)):
                        # prediction = now_preds[pi]

                        prediction = smooth_segmentation_with_blur(
                            preds[pi, 1].detach().cpu().numpy(),
                            lambda_term=50, sigma=5.0)
                        num = len(glob.glob(os.path.join(output_folder, '*')))
                        out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap=cm.gray)
                        plt.imsave(
                            os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                            prediction, cmap=cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                        labels_onehot[pi].unsqueeze(0),
                                                        rmin, rmax, cmin, cmax)

                        row = len(single_df_14)
                        single_df_14.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[now_task] += F1
                        val_Dice[now_task] += DICE
                        val_HD[now_task] += HD
                        val_MSD[now_task] += MSD
                        cnt[now_task] += 1

            'last round to clean up the image pool'
            'medulla'
            if task0_pool_image.num_imgs > 0:
                batch_size = task0_pool_image.num_imgs
                images = task0_pool_image.query(batch_size)
                labels = task0_pool_mask.query(batch_size)
                now_task = torch.tensor(0)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task0_scale.pop(0)
                    layers[bi] = task0_layer.pop(0)
                    filename.append(task0_name.pop(0))

                preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                torch.ones(batch_size).cuda() * now_task,
                                                scales)
                preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                torch.ones(batch_size).cuda() * now_task,
                                                scales)
                preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                torch.ones(batch_size).cuda() * now_task,
                                                scales)
                preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                torch.ones(batch_size).cuda() * now_task,
                                                scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                rmin, rmax, cmin, cmax = 0, 1024, 0, 1024

                labels_onehot = one_hot_3D(labels.long())

                for pi in range(len(images)):
                    prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                               lambda_term=50, sigma=5.0)

                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0), rmin, rmax, cmin, cmax)
                    row = len(single_df_0)
                    single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'cortex'
            if task1_pool_image.num_imgs > 0:
                batch_size = task1_pool_image.num_imgs
                images = task1_pool_image.query(batch_size)
                labels = task1_pool_mask.query(batch_size)
                now_task = torch.tensor(1)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task1_scale.pop(0)
                    layers[bi] = task1_layer.pop(0)
                    filename.append(task1_name.pop(0))

                preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels.long())
                rmin, rmax, cmin, cmax = 0, 1024, 0, 1024

                for pi in range(len(images)):
                    # prediction = now_preds[pi]
                    prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                               lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)
                    row = len(single_df_1)
                    single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'cortex_in'
            if task2_pool_image.num_imgs > 0:
                batch_size = task2_pool_image.num_imgs
                images = task2_pool_image.query(batch_size)
                labels = task2_pool_mask.query(batch_size)
                now_task = torch.tensor(2)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task2_scale.pop(0)
                    layers[bi] = task2_layer.pop(0)
                    filename.append(task2_name.pop(0))

                preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels.long())
                rmin, rmax, cmin, cmax = 0, 1024, 0, 1024

                for pi in range(len(images)):
                    # prediction = now_preds[pi]
                    prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                               lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)
                    row = len(single_df_2)
                    single_df_2.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'cortex_middle'
            if task3_pool_image.num_imgs > 0:
                batch_size = task3_pool_image.num_imgs
                images = task3_pool_image.query(batch_size)
                labels = task3_pool_mask.query(batch_size)
                now_task = torch.tensor(3)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task3_scale.pop(0)
                    layers[bi] = task3_layer.pop(0)
                    filename.append(task3_name.pop(0))

                preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels.long())
                rmin, rmax, cmin, cmax = 0, 1024, 0, 1024

                for pi in range(len(images)):
                    # prediction = now_preds[pi]
                    prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                               lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)
                    row = len(single_df_3)
                    single_df_3.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'cortex_out'
            if task4_pool_image.num_imgs > 0:
                batch_size = task4_pool_image.num_imgs
                images = task4_pool_image.query(batch_size)
                labels = task4_pool_mask.query(batch_size)
                now_task = torch.tensor(4)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task4_scale.pop(0)
                    layers[bi] = task4_layer.pop(0)
                    filename.append(task4_name.pop(0))

                preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
                preds[:, :, :512, :512] = model(images[:, :, :512, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
                                                torch.ones(batch_size).cuda() * now_task, scales)
                preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
                                                torch.ones(batch_size).cuda() * now_task, scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels.long())
                rmin, rmax, cmin, cmax = 0, 1024, 0, 1024

                for pi in range(len(images)):
                    # prediction = now_preds[pi]
                    prediction = smooth_segmentation_with_blur(preds[pi, 1].detach().cpu().numpy(),
                                                               lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)
                    row = len(single_df_4)
                    single_df_4.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'dt'
            if task5_pool_image.num_imgs > 0:
                batch_size = task5_pool_image.num_imgs
                images = task5_pool_image.query(batch_size)
                labels = task5_pool_mask.query(batch_size)
                now_task = torch.tensor(5)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task5_scale.pop(0)
                    layers[bi] = task5_layer.pop(0)
                    filename.append(task5_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 128, 384, 128, 384

                for pi in range(len(images)):
                    # prediction = now_preds[pi, 128:384, 128:384]

                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_5)
                    single_df_5.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'pt'
            if task6_pool_image.num_imgs > 0:
                batch_size = task6_pool_image.num_imgs
                images = task6_pool_image.query(batch_size)
                labels = task6_pool_mask.query(batch_size)
                now_task = torch.tensor(6)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task6_scale.pop(0)
                    layers[bi] = task6_layer.pop(0)
                    filename.append(task6_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 128, 384, 128, 384

                for pi in range(len(images)):
                    # prediction = now_preds[pi, 128:384, 128:384]

                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_6)
                    single_df_6.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'cap'
            if task7_pool_image.num_imgs > 0:
                batch_size = task7_pool_image.num_imgs
                images = task7_pool_image.query(batch_size)
                labels = task7_pool_mask.query(batch_size)
                now_task = torch.tensor(7)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task7_scale.pop(0)
                    layers[bi] = task7_layer.pop(0)
                    filename.append(task7_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 128, 384, 128, 384

                for pi in range(len(images)):
                    # prediction = now_preds[pi, 128:384, 128:384]

                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_7)
                    single_df_7.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'tuft'
            if task8_pool_image.num_imgs > 0:
                batch_size = task8_pool_image.num_imgs
                images = task8_pool_image.query(batch_size)
                labels = task8_pool_mask.query(batch_size)
                now_task = torch.tensor(8)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task8_scale.pop(0)
                    layers[bi] = task8_layer.pop(0)
                    filename.append(task8_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 128, 384, 128, 384

                for pi in range(len(images)):
                    # prediction = now_preds[pi, 128:384, 128:384]

                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_8)
                    single_df_8.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'art'
            if task9_pool_image.num_imgs > 0:
                batch_size = task9_pool_image.num_imgs
                images = task9_pool_image.query(batch_size)
                labels = task9_pool_mask.query(batch_size)
                now_task = torch.tensor(9)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task9_scale.pop(0)
                    layers[bi] = task9_layer.pop(0)
                    filename.append(task9_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 128, 384, 128, 384

                for pi in range(len(images)):
                    # prediction = now_preds[pi, 128:384, 128:384]
                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)

                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_9)
                    single_df_9.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'ptc'
            if task10_pool_image.num_imgs > 0:
                batch_size = task10_pool_image.num_imgs
                images = task10_pool_image.query(batch_size)
                labels = task10_pool_mask.query(batch_size)
                now_task = torch.tensor(10)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task10_scale.pop(0)
                    layers[bi] = task10_layer.pop(0)
                    filename.append(task10_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 128, 384, 128, 384

                for pi in range(len(images)):
                    # prediction = now_preds[pi, 128:384, 128:384]

                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1, 128:384, 128:384].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 384:640, 384:640].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_10)
                    single_df_10.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'mv'
            if task11_pool_image.num_imgs > 0:
                batch_size = task11_pool_image.num_imgs
                images = task11_pool_image.query(batch_size)
                labels = task11_pool_mask.query(batch_size)
                now_task = torch.tensor(11)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task11_scale.pop(0)
                    layers[bi] = task11_layer.pop(0)
                    filename.append(task11_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 0, 512, 0, 512

                for pi in range(len(images)):
                    # prediction = now_preds[pi]

                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_11)
                    single_df_11.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'pod'
            if task12_pool_image.num_imgs > 0:
                batch_size = task12_pool_image.num_imgs
                images = task12_pool_image.query(batch_size)
                labels = task12_pool_mask.query(batch_size)
                now_task = torch.tensor(12)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task12_scale.pop(0)
                    layers[bi] = task12_layer.pop(0)
                    filename.append(task12_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 0, 512, 0, 512

                for pi in range(len(images)):
                    # prediction = now_preds[pi]
                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)

                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_12)
                    single_df_12.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'mes'
            if task13_pool_image.num_imgs > 0:
                batch_size = task13_pool_image.num_imgs
                images = task13_pool_image.query(batch_size)
                labels = task13_pool_mask.query(batch_size)
                now_task = torch.tensor(13)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task13_scale.pop(0)
                    layers[bi] = task13_layer.pop(0)
                    filename.append(task13_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 0, 512, 0, 512

                for pi in range(len(images)):
                    # prediction = now_preds[pi]
                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)

                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_13)
                    single_df_13.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            'smooth'
            if task14_pool_image.num_imgs > 0:
                batch_size = task14_pool_image.num_imgs
                images = task14_pool_image.query(batch_size)
                labels = task14_pool_mask.query(batch_size)
                now_task = torch.tensor(14)
                scales = torch.ones(batch_size).cuda()
                layers = torch.ones(batch_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[bi] = task14_scale.pop(0)
                    layers[bi] = task14_layer.pop(0)
                    filename.append(task14_name.pop(0))

                preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * now_task,
                              scales)

                now_preds = torch.argmax(preds, 1) == 1
                now_preds_onehot = one_hot_3D(now_preds.long())

                labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                rmin, rmax, cmin, cmax = 0, 512, 0, 512

                for pi in range(len(images)):
                    # prediction = now_preds[pi]

                    prediction = smooth_segmentation_with_blur(
                        preds[pi, 1].detach().cpu().numpy(),
                        lambda_term=50, sigma=5.0)
                    num = len(glob.glob(os.path.join(output_folder, '*')))
                    out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, 256:768, 256:768].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(
                        os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                        prediction, cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0),
                                                    labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)

                    row = len(single_df_14)
                    single_df_14.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[now_task] += F1
                    val_Dice[now_task] += DICE
                    val_HD[now_task] += HD
                    val_MSD[now_task] += MSD
                    cnt[now_task] += 1

            avg_val_F1 = val_F1 / cnt
            avg_val_Dice = val_Dice / cnt
            avg_val_HD = val_HD / cnt
            avg_val_MSD = val_MSD / cnt

            print('Validate \n 0medulla_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}'
                  ' \n 1cortex_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 2cortexin_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 3cortexmiddle_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 4cortexout_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 5dt_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 6pt_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 7cap_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 8tuft_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 9art_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 10ptc_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 11mv_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 12pod_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 13mes_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  ' \n 14smooth_f1={:.4} dsc={:.4} hd={:.4} msd={:.4}\n'
                  .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item(),
                          avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item(),
                          avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(), avg_val_MSD[2].item(),
                          avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(), avg_val_MSD[3].item(),
                          avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(), avg_val_MSD[4].item(),
                          avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item(),
                          avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(), avg_val_MSD[6].item(),
                          avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(), avg_val_MSD[7].item(),
                          avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_HD[8].item(), avg_val_MSD[8].item(),
                          avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_HD[9].item(), avg_val_MSD[9].item(),
                          avg_val_F1[10].item(), avg_val_Dice[10].item(), avg_val_HD[10].item(),avg_val_MSD[10].item(),
                          avg_val_F1[11].item(), avg_val_Dice[11].item(), avg_val_HD[11].item(),avg_val_MSD[11].item(),
                          avg_val_F1[12].item(), avg_val_Dice[12].item(), avg_val_HD[12].item(),avg_val_MSD[12].item(),
                          avg_val_F1[13].item(), avg_val_Dice[13].item(), avg_val_HD[13].item(),avg_val_MSD[13].item(),
                          avg_val_F1[14].item(), avg_val_Dice[14].item(), avg_val_HD[14].item(),avg_val_MSD[14].item(), ))

        df = pd.DataFrame(columns=['task', 'F1', 'Dice', 'HD', 'MSD'])
        df.loc[0] = ['0medulla', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(),avg_val_MSD[0].item()]
        df.loc[1] = ['1cortex', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(),avg_val_MSD[1].item()]
        df.loc[2] = ['2cortexin', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_HD[2].item(),avg_val_MSD[2].item()]
        df.loc[3] = ['3cortexmiddle', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_HD[3].item(),avg_val_MSD[3].item()]
        df.loc[4] = ['4cortexout', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_HD[4].item(),avg_val_MSD[4].item()]
        df.loc[5] = ['5dt', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_HD[5].item(), avg_val_MSD[5].item()]
        df.loc[6] = ['6pt', avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_HD[6].item(), avg_val_MSD[6].item()]
        df.loc[7] = ['7cap', avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_HD[7].item(), avg_val_MSD[7].item()]
        df.loc[8] = ['8tuft', avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_HD[8].item(),avg_val_MSD[8].item()]
        df.loc[9] = ['9art', avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_HD[9].item(), avg_val_MSD[9].item()]
        df.loc[10] = ['10ptc', avg_val_F1[10].item(), avg_val_Dice[10].item(), avg_val_HD[10].item(),avg_val_MSD[10].item()]
        df.loc[11] = ['11mv', avg_val_F1[11].item(), avg_val_Dice[11].item(), avg_val_HD[11].item(),avg_val_MSD[11].item()]
        df.loc[12] = ['12pod', avg_val_F1[12].item(), avg_val_Dice[12].item(), avg_val_HD[12].item(),avg_val_MSD[12].item()]
        df.loc[13] = ['13mes', avg_val_F1[13].item(), avg_val_Dice[13].item(), avg_val_HD[13].item(),avg_val_MSD[13].item()]
        df.loc[14] = ['14smooth', avg_val_F1[14].item(), avg_val_Dice[14].item(), avg_val_HD[14].item(),avg_val_MSD[14].item()]

        df.to_csv(os.path.join(output_folder, 'testing_result.csv'))
        single_df_0.to_csv(os.path.join(output_folder,'testing_result_0.csv'))
        single_df_1.to_csv(os.path.join(output_folder,'testing_result_1.csv'))
        single_df_2.to_csv(os.path.join(output_folder,'testing_result_2.csv'))
        single_df_3.to_csv(os.path.join(output_folder,'testing_result_3.csv'))
        single_df_4.to_csv(os.path.join(output_folder,'testing_result_4.csv'))
        single_df_5.to_csv(os.path.join(output_folder,'testing_result_5.csv'))
        single_df_6.to_csv(os.path.join(output_folder,'testing_result_6.csv'))
        single_df_7.to_csv(os.path.join(output_folder,'testing_result_7.csv'))
        single_df_8.to_csv(os.path.join(output_folder,'testing_result_8.csv'))
        single_df_9.to_csv(os.path.join(output_folder,'testing_result_9.csv'))
        single_df_10.to_csv(os.path.join(output_folder,'testing_result_10.csv'))
        single_df_11.to_csv(os.path.join(output_folder,'testing_result_11.csv'))
        single_df_12.to_csv(os.path.join(output_folder,'testing_result_12.csv'))
        single_df_13.to_csv(os.path.join(output_folder,'testing_result_13.csv'))
        single_df_14.to_csv(os.path.join(output_folder,'testing_result_14.csv'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
