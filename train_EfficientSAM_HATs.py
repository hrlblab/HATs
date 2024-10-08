import argparse
import os, sys

import pandas
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
import matplotlib.pyplot as plt

import torch.nn as nn
import imgaug.augmenters as iaa
import kornia
from torchvision import transforms
from PIL import Image, ImageOps

import os.path as osp

from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSDataSet as MOTSDataSet
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSValDataSet as MOTSValDataSet

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util_a.image_pool import ImagePool

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

from unet2D_Dodnet_scale_token import UNet2D as UNet2D_scale

def one_hot_3D(targets,C = 2):
	targets_extend=targets.clone()
	targets_extend.unsqueeze_(1) # convert to Nx1xHxW
	one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
	one_hot.scatter_(1, targets_extend, 1)
	return one_hot

	# parser.add_argument("--valset_dir", type=str, defau


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

	parser = argparse.ArgumentParser(description="DeepLabV3")
	# parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_trainingset_patch/data_list.csv')
	parser.add_argument("--trainset_dir", type=str, default='/Data/HATs/Data/train/data_list.csv')

	# parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/data_list.csv')
	parser.add_argument("--valset_dir", type=str, default='/Data/HATs/Data/val/data_list.csv')


	parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
	parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
	parser.add_argument("--edge_weight", type=float, default=1.0)

	parser.add_argument("--scale", type=str2bool, default=False)
	parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/EfficientSAM_dynamichead_logits_HATs/')
	parser.add_argument("--reload_path", type=str, default='snapshots_2D/EfficientSAM_dynamichead_logits_0201_15class/MOTS_DynConv_EfficientSAM_dynamichead_logits_0201_15class_e50.pth')
	parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
	parser.add_argument("--input_size", type=str, default='512,512')
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--num_gpus", type=int, default=1)
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument("--FP16", type=str2bool, default=False)
	parser.add_argument("--num_epochs", type=int, default=101)
	parser.add_argument("--itrs_each_epoch", type=int, default=250)
	parser.add_argument("--patience", type=int, default=3)
	parser.add_argument("--start_epoch", type=int, default=0)
	parser.add_argument("--val_pred_every", type=int, default=10)
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--num_classes", type=int, default=2)
	parser.add_argument("--num_workers", type=int, default=0)
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


def count_score(preds, labels, rmin, rmax, cmin, cmax):

	Val_F1 = 0
	Val_DICE = 0
	Val_TPR = 0
	Val_PPV = 0
	cnt = 0

	for ki in range(len(preds)):
		cnt += 1


		pred = preds[ki,:,rmin:rmax,cmin:cmax]
		label = labels[ki,:,rmin:rmax,cmin:cmax]

		Val_DICE += dice_score(pred, label)
		preds1 = pred[1, ...].flatten().detach().cpu().numpy()
		labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

		cnf_matrix = confusion_matrix(preds1, labels1)

		try:
			FP = cnf_matrix[1,0]
			FN = cnf_matrix[0,1]
			TP = cnf_matrix[1,1]
			TN = cnf_matrix[0,0]

		except:
			FP = np.array(1)
			FN = np.array(1)
			TP = np.array(1)
			TN = np.array(1)


		FP = FP.astype(float)
		FN = FN.astype(float)
		TP = TP.astype(float)
		TN = TN.astype(float)

		Val_TPR += TP / (TP + FN)
		Val_PPV += TP / (TP + FP)

		Val_F1 += f1_score(preds1, labels1, average='macro')

	return Val_F1/cnt, Val_DICE/cnt, Val_TPR/cnt, Val_PPV/cnt

def dice_score(preds, labels):  # on GPU
	assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
	predict = preds.contiguous().view(preds.shape[0], -1)
	target = labels.contiguous().view(labels.shape[0], -1)

	num = torch.sum(torch.mul(predict, target), dim=1)
	den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

	dice = 2 * num / den

	return dice.mean()


def get_loss(images, preds, labels, weight, loss_seg_DICE, loss_seg_CE):

	term_seg_Dice = 0
	term_seg_BCE = 0
	term_all = 0

	term_seg_Dice += loss_seg_DICE.forward(preds, labels, weight)
	term_seg_BCE += loss_seg_CE.forward(preds, labels, weight)
	term_all += (term_seg_Dice + term_seg_BCE)

	return term_seg_Dice, term_seg_BCE, term_all

def supervise_learning(images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE, loss_seg_CE):

	preds = model(images, torch.ones(batch_size).cuda() * now_task, scales)

	labels = one_hot_3D(labels.long())

	term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds, labels, weight, loss_seg_DICE, loss_seg_CE)

	return term_seg_Dice, term_seg_BCE, term_all


def HATs_learning(images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE, loss_seg_CE, term_seg_Dice, term_seg_BCE, term_all, HATs_matrix, semi_ratio, area_ratio):

	for ii in range(len(HATs_matrix[1])):
		now_task_semi = ii
		if now_task_semi == now_task:
			continue
		now_relative = HATs_matrix[now_task][now_task_semi]
		now_area_ratio = area_ratio[now_task][now_task_semi]

		if now_relative == 0:
			continue

		semi_preds = model(images, torch.ones(batch_size).cuda() * now_task_semi, scales)

		'Only use dice rather than bce in semi-supervised learning'

		if now_relative == 1:
			semi_labels = 1 - labels                        # Background from this label should not have any overlap with the pred, --> 0
			semi_labels = one_hot_3D(semi_labels.long())
			semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE)
			term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
			term_all -= semi_ratio * semi_seg_Dice * now_area_ratio


		elif now_relative == -1:
			semi_labels = labels
			semi_preds = semi_labels.unsqueeze(1).repeat(1,2,1,1) * semi_preds           # Only supervised the regions which have label  --> 1
			semi_labels = one_hot_3D(semi_labels.long())
			semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE)
			term_seg_Dice += semi_ratio * semi_seg_Dice * now_area_ratio
			term_all += semi_ratio * semi_seg_Dice * now_area_ratio

		elif now_relative == 2:
			semi_labels = labels                            # Foreground from this label should not have any overlap with the pred, --> 0
			semi_labels = one_hot_3D(semi_labels.long())
			semi_seg_Dice, semi_seg_BCE, semi_all = get_loss(images, semi_preds, semi_labels, weight, loss_seg_DICE, loss_seg_CE)
			term_seg_Dice -= semi_ratio * semi_seg_Dice * now_area_ratio
			term_all -= semi_ratio * semi_seg_Dice * now_area_ratio

	return term_seg_Dice, term_seg_BCE, term_all


def division_ratio(a, b):
	if a > b:
		return b / a
	else:
		return a / b

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
		# HATs backbone
		model = build_efficient_sam_vits(task_num = 15, scale_num = 4)
		model.image_encoder.requires_grad_(False)

		# PrPSeg backbone
		# model = UNet2D_scale(num_classes=15, num_scale = 4, weight_std=False)

		check_wo_gpu = 0

		if not check_wo_gpu:
			device = torch.device('cuda:{}'.format(args.local_rank))
			model.to(device)

		optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

		if not check_wo_gpu:
			if args.FP16:
				print("Note: Using FP16 during training************")
				model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

			if args.num_gpus > 1:
				model = engine.data_parallel(model)

		# load checkpoint...a
		if args.reload_from_checkpoint:
			print('loading from checkpoint: {}'.format(args.reload_path))
			if os.path.exists(args.reload_path):
				if args.FP16:
					checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
					model.load_state_dict(checkpoint['model'])
					optimizer.load_state_dict(checkpoint['optimizer'])
					amp.load_state_dict(checkpoint['amp'])
				else:
					model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')),strict = False)
			else:
				print('File not exists in the reload path: {}'.format(args.reload_path))

		if not check_wo_gpu:
			weights = [1., 1.]
			class_weights = torch.FloatTensor(weights).cuda()
			loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
			loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=args.num_classes, ignore_index=255).to(device)
			loss_KL = nn.KLDivLoss().to(device)
			loss_MSE = nn.MSELoss().to(device)


		else:
			weights = [1., 1.]
			class_weights = torch.FloatTensor(weights)
			loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
			loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=args.num_classes, ignore_index=255)
			loss_KL = nn.KLDivLoss()
			loss_MSE = nn.MSELoss()


		if not os.path.exists(args.snapshot_dir):
			os.makedirs(args.snapshot_dir)

		edge_weight = args.edge_weight

		num_worker = 8

		trainloader = DataLoader(
			MOTSDataSet(args.trainset_dir, args.train_list, max_iters=args.itrs_each_epoch * args.batch_size,
						crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
						edge_weight=edge_weight),batch_size=4,shuffle=True,num_workers=num_worker)

		valloader = DataLoader(
			MOTSValDataSet(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
						   crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
						   edge_weight=edge_weight),batch_size=4,shuffle=False,num_workers=num_worker)

		all_tr_loss_supervise = []
		all_tr_loss_all = []

		all_tr_loss = []
		all_va_loss = []
		train_loss_MA = None
		val_loss_MA = None

		val_best_loss = 999999

		# layer_num = [0,2,6]
		'extend from 8 to 15'
		layer_num = [0,5,12]
		semi_ratio = 0.1

		#HATs_matrix = np.zeros((8,8))
		'extend from 8 to 15'
		HATs_matrix = np.zeros((15, 15))

		Area = np.zeros((15))
		Area[0] = 2.434
		Area[1] = 2.600
		Area[2] = 1.760
		Area[3] = 1.853
		Area[4] = 1.844
		Area[5] = 0.097
		Area[6] = 0.360
		Area[7] = 0.619
		Area[8] = 0.466
		Area[9] = 0.083
		Area[10] = 0.002
		Area[11] = 0.012
		Area[12] = 0.001
		Area[13] = 0.001
		Area[14] = 0.002

		Area_ratio = np.zeros((15, 15))
		for xi in range(0,15):
			for yi in range(0,15):
				Area_ratio[xi,yi] = division_ratio(Area[xi], Area[yi])

		'0_medulla'
		HATs_matrix[0, 1] = 2  # 1_cortex
		HATs_matrix[0, 2] = 2  # 2_cortexin
		HATs_matrix[0, 3] = 2  # 3_cortexmiddle
		HATs_matrix[0, 4] = 2  # 4_cortexout

		HATs_matrix[0, 7] = 2  #7_cap
		HATs_matrix[0, 8] = 2  #8_tuft
		HATs_matrix[0, 10] = 2  #10_ptc
		HATs_matrix[0, 11] = 1  #11_mv    medulla cover mv

		HATs_matrix[0, 12] = 2  #12_pod
		HATs_matrix[0, 13] = 2  #13_mes


		'1_cortex'
		HATs_matrix[1, 0] = 2  #0_medulla
		HATs_matrix[1, 2] = 1  # 2_cortexin cortex cover cortexin
		HATs_matrix[1, 3] = 1  # 3_cortexmiddle cortex cover cortexmiddle
		HATs_matrix[1, 4] = 1  # 4_cortexout cortex cover cortexout

		HATs_matrix[1, 7] = 1  # 7_cap cortex cover cap
		HATs_matrix[1, 8] = 1  # 8_tuft cortex cover tuft
		HATs_matrix[1, 10] = 1  # 10_ptc cortex cover ptc
		HATs_matrix[1, 11] = 2  # 11_mv

		HATs_matrix[1, 12] = 1  #12_pod cortex cover pod
		HATs_matrix[1, 13] = 1  #13_mes cortex cover mes


		'2_cortexin'
		HATs_matrix[2, 0] = 2  #0_medulla
		HATs_matrix[2, 1] = -1  # 1_cortex cortexin is covered by cortex
		HATs_matrix[2, 3] = 2
		HATs_matrix[2, 4] = 2


		'3_cortexmiddle'
		HATs_matrix[3, 0] = 2  #0_medulla
		HATs_matrix[3, 1] = -1  # 1_cortex cortexmiddle is covered by cortex
		HATs_matrix[3, 2] = 2
		HATs_matrix[3, 4] = 2


		'4_cortexout'
		HATs_matrix[4, 0] = 2  #0_medulla
		HATs_matrix[4, 1] = -1  # 1_cortex cortexout is covered by cortex
		HATs_matrix[4, 2] = 2
		HATs_matrix[4, 3] = 2


		'5_dt'
		HATs_matrix[5, 6] = 2  #6_pt
		HATs_matrix[5, 7] = 2  #7_cap
		HATs_matrix[5, 8] = 2  #8_tuft
		HATs_matrix[5, 9] = 2  #9_art
		HATs_matrix[5, 10] = 2  #10_ptc
		HATs_matrix[5, 11] = 2  #11_mv

		HATs_matrix[5, 12] = 2  #12_pod
		HATs_matrix[5, 13] = 2  #13_mes
		HATs_matrix[5, 14] = 2  #14_smooth


		'6_pt'
		HATs_matrix[6, 5] = 2  #5_dt
		HATs_matrix[6, 7] = 2  #7_cap
		HATs_matrix[6, 8] = 2  #8_tuft
		HATs_matrix[6, 9] = 2  #9_art
		HATs_matrix[6, 10] = 2  #10_ptc
		HATs_matrix[6, 11] = 2  #11_mv

		HATs_matrix[6, 12] = 2  #12_pod
		HATs_matrix[6, 13] = 2  #13_mes
		HATs_matrix[6, 14] = 2  #14_smooth


		'7_cap'
		HATs_matrix[7, 0] = 2  #0_medulla
		HATs_matrix[7, 1] = -1  #1_cortex  cap is covered by cortex but don't know between in/middle/out

		HATs_matrix[7, 5] = 2  #5_dt
		HATs_matrix[7, 6] = 2  #6_pt
		HATs_matrix[7, 8] = 1  #8_tuft  cap covers tuft
		HATs_matrix[7, 9] = 2  #9_art
		HATs_matrix[7, 10] = 2  #10_ptc
		HATs_matrix[7, 11] = 2  #11_mv

		HATs_matrix[7, 12] = 1  #12_pod   cap cover pod
		HATs_matrix[7, 13] = 1  #13_mes   cap cover mes
		HATs_matrix[7, 14] = 2  #14_smooth


		'8_tuft'
		HATs_matrix[8, 0] = 2  #0_medulla
		HATs_matrix[8, 1] = -1  #1_cortex  tuft is covered by cortex but don't know between in/middle/out

		HATs_matrix[8, 5] = 2  #5_dt
		HATs_matrix[8, 6] = 2  #6_pt
		HATs_matrix[8, 7] = -1  #7_cap  tuft is covered by cap
		HATs_matrix[8, 9] = 2  #9_art
		HATs_matrix[8, 10] = 2  #10_ptc
		HATs_matrix[8, 11] = 2  #11_mv

		HATs_matrix[8, 12] = 1  #12_pod   tuft cover pod
		HATs_matrix[8, 13] = 1  #13_mes   tuft cover mes
		HATs_matrix[8, 14] = 2  #14_smooth


		'9_art'
		HATs_matrix[9, 5] = 2  #5_dt
		HATs_matrix[9, 6] = 2  #6_pt
		HATs_matrix[9, 7] = 2  #7_cap
		HATs_matrix[9, 8] = 2  #8_tuft
		HATs_matrix[9, 10] = 2  #10_ptc
		HATs_matrix[9, 11] = 2  #11_mv

		HATs_matrix[9, 12] = 2  #12_pod
		HATs_matrix[9, 13] = 2  #13_mes
		HATs_matrix[9, 14] = 1  #14_smooth art cover smooth


		'10_ptc'
		HATs_matrix[10, 0] = 2  #0_medulla
		HATs_matrix[10, 1] = -1  #1_cortex  PTC is covered by cortex but don't know between in/middle/out

		HATs_matrix[10, 5] = 2  #5_dt
		HATs_matrix[10, 6] = 2  #6_pt
		HATs_matrix[10, 7] = 2  #7_cap
		HATs_matrix[10, 8] = 2  #8_tuft
		HATs_matrix[10, 9] = 2  #9_art
		HATs_matrix[10, 11] = 2  #11_mv

		HATs_matrix[10, 12] = 2  #12_pod
		HATs_matrix[10, 13] = 2  #13_mes
		HATs_matrix[10, 14] = 2  #14_smooth


		'11_mv'
		HATs_matrix[11, 0] = -1  #0_medulla mv is covered by medulla
		HATs_matrix[11, 1] = 2  #1_cortex

		HATs_matrix[11, 5] = 2  #5_dt
		HATs_matrix[11, 6] = 2  #6_pt
		HATs_matrix[11, 7] = 2  #7_cap
		HATs_matrix[11, 8] = 2  #8_tuft
		HATs_matrix[11, 9] = 2  #9_art
		HATs_matrix[11, 10] = 2  #10_ptc

		HATs_matrix[11, 12] = 2  #12_pod
		HATs_matrix[11, 13] = 2  #13_mes
		HATs_matrix[11, 14] = 2  #14_smooth


		'12_pod'
		HATs_matrix[12, 0] = 2  #0_medulla
		HATs_matrix[12, 1] = -1  #1_cortex  pod is covered by cortex but don't know between in/middle/out

		HATs_matrix[12, 5] = 2  #5_dt
		HATs_matrix[12, 6] = 2  #6_pt
		HATs_matrix[12, 7] = -1  #7_cap     pod is covered by cap
		HATs_matrix[12, 8] = -1  #8_tuft    pod is covered by tuft
		HATs_matrix[12, 9] = 2  #9_art
		HATs_matrix[12, 10] = 2  #10_ptc
		HATs_matrix[12, 11] = 2  #11_mv

		HATs_matrix[12, 13] = 2  #13_mes
		HATs_matrix[12, 14] = 2  #14_smooth


		'13_mes'
		HATs_matrix[13, 0] = 2  #0_medulla
		HATs_matrix[13, 1] = -1  #1_cortex  pod is covered by cortex but don't know between in/middle/out

		HATs_matrix[13, 5] = 2  #5_dt
		HATs_matrix[13, 6] = 2  #6_pt
		HATs_matrix[13, 7] = -1  #7_cap     med is covered by cap
		HATs_matrix[13, 8] = -1  #8_tuft    med is covered by tuft
		HATs_matrix[13, 9] = 2  #9_art
		HATs_matrix[13, 10] = 2  #10_ptc
		HATs_matrix[13, 11] = 2  #11_mv

		HATs_matrix[13, 12] = 2  #12_pod
		HATs_matrix[13, 14] = 2  #14_smooth


		'14_smooth'
		HATs_matrix[14, 5] = 2  #5_dt
		HATs_matrix[14, 6] = 2  #6_pt
		HATs_matrix[14, 7] = 2  #7_cap
		HATs_matrix[14, 8] = 2  #8_tuft
		HATs_matrix[14, 9] = -1  #9_art     smooth is covered by art
		HATs_matrix[14, 10] = 2  #10_ptc
		HATs_matrix[14, 11] = 2  #11_mv

		HATs_matrix[14, 12] = 2  #12_pod
		HATs_matrix[14, 13] = 2  #13_mes


		df_loss = pandas.DataFrame(columns= ['epoch','epoch_loss_supervise_mean','semi_all'])

		for epoch in range(50,args.num_epochs):
			model.train()

			task0_pool_image = ImagePool(8)
			task0_pool_mask = ImagePool(8)
			task0_pool_weight = ImagePool(8)
			task0_scale = []
			task0_layer = []

			task1_pool_image = ImagePool(8)
			task1_pool_mask = ImagePool(8)
			task1_pool_weight = ImagePool(8)
			task1_scale = []
			task1_layer = []

			task2_pool_image = ImagePool(8)
			task2_pool_mask = ImagePool(8)
			task2_pool_weight = ImagePool(8)
			task2_scale = []
			task2_layer = []

			task3_pool_image = ImagePool(8)
			task3_pool_mask = ImagePool(8)
			task3_pool_weight = ImagePool(8)
			task3_scale = []
			task3_layer = []

			task4_pool_image = ImagePool(8)
			task4_pool_mask = ImagePool(8)
			task4_pool_weight = ImagePool(8)
			task4_scale = []
			task4_layer = []

			task5_pool_image = ImagePool(8)
			task5_pool_mask = ImagePool(8)
			task5_pool_weight = ImagePool(8)
			task5_scale = []
			task5_layer = []

			task6_pool_image = ImagePool(8)
			task6_pool_mask = ImagePool(8)
			task6_pool_weight = ImagePool(8)
			task6_scale = []
			task6_layer = []

			task7_pool_image = ImagePool(8)
			task7_pool_mask = ImagePool(8)
			task7_pool_weight = ImagePool(8)
			task7_scale = []
			task7_layer = []

			'extend to 15 classes'
			task8_pool_image = ImagePool(8)
			task8_pool_mask = ImagePool(8)
			task8_pool_weight = ImagePool(8)
			task8_scale = []
			task8_layer = []

			task9_pool_image = ImagePool(8)
			task9_pool_mask = ImagePool(8)
			task9_pool_weight = ImagePool(8)
			task9_scale = []
			task9_layer = []

			task10_pool_image = ImagePool(8)
			task10_pool_mask = ImagePool(8)
			task10_pool_weight = ImagePool(8)
			task10_scale = []
			task10_layer = []

			task11_pool_image = ImagePool(8)
			task11_pool_mask = ImagePool(8)
			task11_pool_weight = ImagePool(8)
			task11_scale = []
			task11_layer = []

			task12_pool_image = ImagePool(8)
			task12_pool_mask = ImagePool(8)
			task12_pool_weight = ImagePool(8)
			task12_scale = []
			task12_layer = []

			task13_pool_image = ImagePool(8)
			task13_pool_mask = ImagePool(8)
			task13_pool_weight = ImagePool(8)
			task13_scale = []
			task13_layer = []

			task14_pool_image = ImagePool(8)
			task14_pool_mask = ImagePool(8)
			task14_pool_weight = ImagePool(8)
			task14_scale = []
			task14_layer = []

			if epoch < args.start_epoch:
				continue

			if engine.distributed:
				train_sampler.set_epoch(epoch)

			epoch_loss = []
			adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

			batch_size = args.batch_size
			task_num = 15
			each_loss = torch.zeros((task_num)).cuda()
			count_batch = torch.zeros((task_num)).cuda()
			loss_weight = torch.ones((task_num)).cuda()
			supervised_loss = torch.zeros((task_num)).cuda()

			for iter, batch in enumerate(trainloader):

				# if iter > 10:
				# 	break

				'dataloader'
				imgs = batch[0].cuda()
				lbls = batch[1].cuda()
				wt = batch[2].cuda().float()
				volumeName = batch[3]
				l_ids = batch[4].cuda()
				t_ids = batch[5].cuda()
				s_ids = batch[6].cuda()

				sum_loss = 0

				for ki in range(len(imgs)):
					now_task = layer_num[l_ids[ki]] + t_ids[ki]

					# if now_task!= 14:
					# 	continue

					if now_task == 0:
						task0_pool_image.add(imgs[ki].unsqueeze(0))
						task0_pool_mask.add(lbls[ki].unsqueeze(0))
						task0_pool_weight.add(wt[ki].unsqueeze(0))
						task0_scale.append((s_ids[ki]))
						task0_layer.append((l_ids[ki]))
					elif now_task == 1:
						task1_pool_image.add(imgs[ki].unsqueeze(0))
						task1_pool_mask.add(lbls[ki].unsqueeze(0))
						task1_pool_weight.add(wt[ki].unsqueeze(0))
						task1_scale.append((s_ids[ki]))
						task1_layer.append((l_ids[ki]))
					elif now_task == 2:
						task2_pool_image.add(imgs[ki].unsqueeze(0))
						task2_pool_mask.add(lbls[ki].unsqueeze(0))
						task2_pool_weight.add(wt[ki].unsqueeze(0))
						task2_scale.append((s_ids[ki]))
						task2_layer.append((l_ids[ki]))
					elif now_task == 3:
						task3_pool_image.add(imgs[ki].unsqueeze(0))
						task3_pool_mask.add(lbls[ki].unsqueeze(0))
						task3_pool_weight.add(wt[ki].unsqueeze(0))
						task3_scale.append((s_ids[ki]))
						task3_layer.append((l_ids[ki]))
					elif now_task == 4:
						task4_pool_image.add(imgs[ki].unsqueeze(0))
						task4_pool_mask.add(lbls[ki].unsqueeze(0))
						task4_pool_weight.add(wt[ki].unsqueeze(0))
						task4_scale.append((s_ids[ki]))
						task4_layer.append((l_ids[ki]))
					elif now_task == 5:
						task5_pool_image.add(imgs[ki].unsqueeze(0))
						task5_pool_mask.add(lbls[ki].unsqueeze(0))
						task5_pool_weight.add(wt[ki].unsqueeze(0))
						task5_scale.append((s_ids[ki]))
						task5_layer.append((l_ids[ki]))
					elif now_task == 6:
						task6_pool_image.add(imgs[ki].unsqueeze(0))
						task6_pool_mask.add(lbls[ki].unsqueeze(0))
						task6_pool_weight.add(wt[ki].unsqueeze(0))
						task6_scale.append((s_ids[ki]))
						task6_layer.append((l_ids[ki]))
					elif now_task == 7:
						task7_pool_image.add(imgs[ki].unsqueeze(0))
						task7_pool_mask.add(lbls[ki].unsqueeze(0))
						task7_pool_weight.add(wt[ki].unsqueeze(0))
						task7_scale.append((s_ids[ki]))
						task7_layer.append((l_ids[ki]))

						'extend to 15 classes'
					elif now_task == 8:
						task8_pool_image.add(imgs[ki].unsqueeze(0))
						task8_pool_mask.add(lbls[ki].unsqueeze(0))
						task8_pool_weight.add(wt[ki].unsqueeze(0))
						task8_scale.append((s_ids[ki]))
						task8_layer.append((l_ids[ki]))

					elif now_task == 9:
						task9_pool_image.add(imgs[ki].unsqueeze(0))
						task9_pool_mask.add(lbls[ki].unsqueeze(0))
						task9_pool_weight.add(wt[ki].unsqueeze(0))
						task9_scale.append((s_ids[ki]))
						task9_layer.append((l_ids[ki]))

					elif now_task == 10:
						task10_pool_image.add(imgs[ki].unsqueeze(0))
						task10_pool_mask.add(lbls[ki].unsqueeze(0))
						task10_pool_weight.add(wt[ki].unsqueeze(0))
						task10_scale.append((s_ids[ki]))
						task10_layer.append((l_ids[ki]))

					elif now_task == 11:
						task11_pool_image.add(imgs[ki].unsqueeze(0))
						task11_pool_mask.add(lbls[ki].unsqueeze(0))
						task11_pool_weight.add(wt[ki].unsqueeze(0))
						task11_scale.append((s_ids[ki]))
						task11_layer.append((l_ids[ki]))

					elif now_task == 12:
						task12_pool_image.add(imgs[ki].unsqueeze(0))
						task12_pool_mask.add(lbls[ki].unsqueeze(0))
						task12_pool_weight.add(wt[ki].unsqueeze(0))
						task12_scale.append((s_ids[ki]))
						task12_layer.append((l_ids[ki]))

					elif now_task == 13:
						task13_pool_image.add(imgs[ki].unsqueeze(0))
						task13_pool_mask.add(lbls[ki].unsqueeze(0))
						task13_pool_weight.add(wt[ki].unsqueeze(0))
						task13_scale.append((s_ids[ki]))
						task13_layer.append((l_ids[ki]))

					elif now_task == 14:
						task14_pool_image.add(imgs[ki].unsqueeze(0))
						task14_pool_mask.add(lbls[ki].unsqueeze(0))
						task14_pool_weight.add(wt[ki].unsqueeze(0))
						task14_scale.append((s_ids[ki]))
						task14_layer.append((l_ids[ki]))


				if task0_pool_image.num_imgs >= batch_size:
					images = task0_pool_image.query(batch_size)
					labels = task0_pool_mask.query(batch_size)
					wts = task0_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task0_scale.pop(0)
						layers[bi] = task0_layer.pop(0)

					now_task = 0
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales,model, now_task, weight, loss_seg_DICE, loss_seg_CE, term_seg_Dice, term_seg_BCE, Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all, Sup_term_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all
					#sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))


				if task1_pool_image.num_imgs >= batch_size:
					images = task1_pool_image.query(batch_size)
					labels = task1_pool_mask.query(batch_size)
					wts = task1_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task1_scale.pop(0)
						layers[bi] = task1_layer.pop(0)


					now_task = 1
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all, Sup_term_all)
					supervised_loss[now_task] += supervise_all


					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task2_pool_image.num_imgs >= batch_size:
					images = task2_pool_image.query(batch_size)
					labels = task2_pool_mask.query(batch_size)
					wts = task2_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task2_scale.pop(0)
						layers[bi] = task2_layer.pop(0)

					now_task = 2
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all, Sup_term_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task3_pool_image.num_imgs >= batch_size:
					images = task3_pool_image.query(batch_size)
					labels = task3_pool_mask.query(batch_size)
					wts = task3_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task3_scale.pop(0)
						layers[bi] = task3_layer.pop(0)

					now_task = 3
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all, Sup_term_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task4_pool_image.num_imgs >= batch_size:
					images = task4_pool_image.query(batch_size)
					labels = task4_pool_mask.query(batch_size)
					wts = task4_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task4_scale.pop(0)
						layers[bi] = task4_layer.pop(0)

					now_task = 4
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all, Sup_term_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))


				if task5_pool_image.num_imgs >= batch_size:
					images = task5_pool_image.query(batch_size)
					labels = task5_pool_mask.query(batch_size)
					wts = task5_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task5_scale.pop(0)
						layers[bi] = task5_layer.pop(0)

					now_task = 5
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all, Sup_term_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))


				if task6_pool_image.num_imgs >= batch_size:
					images = task6_pool_image.query(batch_size)
					labels = task6_pool_mask.query(batch_size)
					wts = task6_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task6_scale.pop(0)
						layers[bi] = task6_layer.pop(0)

					now_task = 6
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all, Sup_term_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task7_pool_image.num_imgs >= batch_size:
					images = task7_pool_image.query(batch_size)
					labels = task7_pool_mask.query(batch_size)
					wts = task7_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task7_scale.pop(0)
						layers[bi] = task7_layer.pop(0)

					now_task = 7
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					# print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				'extend to 15'
				if task8_pool_image.num_imgs >= batch_size:
					images = task8_pool_image.query(batch_size)
					labels = task8_pool_mask.query(batch_size)
					wts = task8_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task8_scale.pop(0)
						layers[bi] = task8_layer.pop(0)

					now_task = 8
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))


				if task9_pool_image.num_imgs >= batch_size:
					images = task9_pool_image.query(batch_size)
					labels = task9_pool_mask.query(batch_size)
					wts = task9_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task9_scale.pop(0)
						layers[bi] = task9_layer.pop(0)

					now_task = 9
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task10_pool_image.num_imgs >= batch_size:
					images = task10_pool_image.query(batch_size)
					labels = task10_pool_mask.query(batch_size)
					wts = task10_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task10_scale.pop(0)
						layers[bi] = task10_layer.pop(0)

					now_task = 10
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task11_pool_image.num_imgs >= batch_size:
					images = task11_pool_image.query(batch_size)
					labels = task11_pool_mask.query(batch_size)
					wts = task11_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task11_scale.pop(0)
						layers[bi] = task11_layer.pop(0)

					now_task = 11
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task12_pool_image.num_imgs >= batch_size:
					images = task12_pool_image.query(batch_size)
					labels = task12_pool_mask.query(batch_size)
					wts = task12_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task12_scale.pop(0)
						layers[bi] = task12_layer.pop(0)

					now_task = 12
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																			   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

				if task13_pool_image.num_imgs >= batch_size:
					images = task13_pool_image.query(batch_size)
					labels = task13_pool_mask.query(batch_size)
					wts = task13_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task13_scale.pop(0)
						layers[bi] = task13_layer.pop(0)

					now_task = 13
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))


				if task14_pool_image.num_imgs >= batch_size:
					images = task14_pool_image.query(batch_size)
					labels = task14_pool_mask.query(batch_size)
					wts = task14_pool_weight.query(batch_size)
					scales = torch.ones(batch_size).cuda()
					layers = torch.ones(batch_size).cuda()
					for bi in range(len(scales)):
						scales[bi] = task14_scale.pop(0)
						layers[bi] = task14_layer.pop(0)

					now_task = 14
					weight = edge_weight ** wts

					'supervise_learning'
					term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
																				   scales,
																				   model, now_task, weight,
																				   loss_seg_DICE, loss_seg_CE)

					term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
																			  now_task, weight, loss_seg_DICE,
																			  loss_seg_CE, term_seg_Dice, term_seg_BCE,
																			  Sup_term_all, HATs_matrix, semi_ratio, Area_ratio)

					reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
					reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
					reduce_all = engine.all_reduce_tensor(All_term_all)

					optimizer.zero_grad()
					reduce_all.backward()
					optimizer.step()

					print(
						'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
							epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
							reduce_BCE.item(), reduce_all.item()))

					supervise_all = engine.all_reduce_tensor(Sup_term_all)
					#print(supervise_all)
					supervised_loss[now_task] += supervise_all

					term_all = reduce_all

					# sum_loss += term_all
					each_loss[now_task] += term_all
					count_batch[now_task] += 1

					epoch_loss.append(float(term_all))

			'last round clean the image pool'

			if task0_pool_image.num_imgs > 0:
				batch_size = task0_pool_image.num_imgs
				images = task0_pool_image.query(batch_size)
				labels = task0_pool_mask.query(batch_size)
				wts = task0_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task0_scale.pop(0)
					layers[bi] = task0_layer.pop(0)

				now_task = 0
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE, loss_seg_CE,
				                                                          term_seg_Dice, term_seg_BCE, Sup_term_all,
				                                                          HATs_matrix, semi_ratio, Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all, Sup_term_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all
				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task1_pool_image.num_imgs > 0:
				batch_size = task1_pool_image.num_imgs
				images = task1_pool_image.query(batch_size)
				labels = task1_pool_mask.query(batch_size)
				wts = task1_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task1_scale.pop(0)
					layers[bi] = task1_layer.pop(0)

				now_task = 1
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all, Sup_term_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task2_pool_image.num_imgs > 0:
				batch_size = task2_pool_image.num_imgs
				images = task2_pool_image.query(batch_size)
				labels = task2_pool_mask.query(batch_size)
				wts = task2_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task2_scale.pop(0)
					layers[bi] = task2_layer.pop(0)

				now_task = 2
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all, Sup_term_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task3_pool_image.num_imgs > 0:
				batch_size = task3_pool_image.num_imgs
				images = task3_pool_image.query(batch_size)
				labels = task3_pool_mask.query(batch_size)
				wts = task3_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task3_scale.pop(0)
					layers[bi] = task3_layer.pop(0)

				now_task = 3
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all, Sup_term_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task4_pool_image.num_imgs > 0:
				batch_size = task4_pool_image.num_imgs
				images = task4_pool_image.query(batch_size)
				labels = task4_pool_mask.query(batch_size)
				wts = task4_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task4_scale.pop(0)
					layers[bi] = task4_layer.pop(0)

				now_task = 4
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all, Sup_term_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task5_pool_image.num_imgs > 0:
				batch_size = task5_pool_image.num_imgs
				images = task5_pool_image.query(batch_size)
				labels = task5_pool_mask.query(batch_size)
				wts = task5_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task5_scale.pop(0)
					layers[bi] = task5_layer.pop(0)

				now_task = 5
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all, Sup_term_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task6_pool_image.num_imgs > 0:
				batch_size = task6_pool_image.num_imgs
				images = task6_pool_image.query(batch_size)
				labels = task6_pool_mask.query(batch_size)
				wts = task6_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task6_scale.pop(0)
					layers[bi] = task6_layer.pop(0)

				now_task = 6
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all, Sup_term_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task7_pool_image.num_imgs > 0:
				batch_size = task7_pool_image.num_imgs
				images = task7_pool_image.query(batch_size)
				labels = task7_pool_mask.query(batch_size)
				wts = task7_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task7_scale.pop(0)
					layers[bi] = task7_layer.pop(0)

				now_task = 7
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			'extend to 15'
			if task8_pool_image.num_imgs > 0:
				batch_size = task8_pool_image.num_imgs
				images = task8_pool_image.query(batch_size)
				labels = task8_pool_mask.query(batch_size)
				wts = task8_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task8_scale.pop(0)
					layers[bi] = task8_layer.pop(0)

				now_task = 8
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task9_pool_image.num_imgs > 0:
				batch_size = task9_pool_image.num_imgs
				images = task9_pool_image.query(batch_size)
				labels = task9_pool_mask.query(batch_size)
				wts = task9_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task9_scale.pop(0)
					layers[bi] = task9_layer.pop(0)

				now_task = 9
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task10_pool_image.num_imgs > 0:
				batch_size = task10_pool_image.num_imgs
				images = task10_pool_image.query(batch_size)
				labels = task10_pool_mask.query(batch_size)
				wts = task10_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task10_scale.pop(0)
					layers[bi] = task10_layer.pop(0)

				now_task = 10
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task11_pool_image.num_imgs > 0:
				batch_size = task11_pool_image.num_imgs
				images = task11_pool_image.query(batch_size)
				labels = task11_pool_mask.query(batch_size)
				wts = task11_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task11_scale.pop(0)
					layers[bi] = task11_layer.pop(0)

				now_task = 11
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task12_pool_image.num_imgs > 0:
				batch_size = task12_pool_image.num_imgs
				images = task12_pool_image.query(batch_size)
				labels = task12_pool_mask.query(batch_size)
				wts = task12_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task12_scale.pop(0)
					layers[bi] = task12_layer.pop(0)

				now_task = 12
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task13_pool_image.num_imgs > 0:
				batch_size = task13_pool_image.num_imgs
				images = task13_pool_image.query(batch_size)
				labels = task13_pool_mask.query(batch_size)
				wts = task13_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task13_scale.pop(0)
					layers[bi] = task13_layer.pop(0)

				now_task = 13
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			if task14_pool_image.num_imgs > 0:
				batch_size = task14_pool_image.num_imgs
				images = task14_pool_image.query(batch_size)
				labels = task14_pool_mask.query(batch_size)
				wts = task14_pool_weight.query(batch_size)
				scales = torch.ones(batch_size).cuda()
				layers = torch.ones(batch_size).cuda()
				for bi in range(len(scales)):
					scales[bi] = task14_scale.pop(0)
					layers[bi] = task14_layer.pop(0)

				now_task = 14
				weight = edge_weight ** wts

				'supervise_learning'
				term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
				                                                               scales,
				                                                               model, now_task, weight,
				                                                               loss_seg_DICE, loss_seg_CE)

				term_seg_Dice, term_seg_BCE, All_term_all = HATs_learning(images, labels, batch_size, scales, model,
				                                                          now_task, weight, loss_seg_DICE,
				                                                          loss_seg_CE, term_seg_Dice, term_seg_BCE,
				                                                          Sup_term_all, HATs_matrix, semi_ratio,
				                                                          Area_ratio)

				reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
				reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
				reduce_all = engine.all_reduce_tensor(All_term_all)

				optimizer.zero_grad()
				reduce_all.backward()
				optimizer.step()

				print(
					'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
						epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
						reduce_BCE.item(), reduce_all.item()))

				supervise_all = engine.all_reduce_tensor(Sup_term_all)
				# print(supervise_all)
				supervised_loss[now_task] += supervise_all

				term_all = reduce_all

				# sum_loss += term_all
				each_loss[now_task] += term_all
				count_batch[now_task] += 1

				epoch_loss.append(float(term_all))

			epoch_loss = np.mean(epoch_loss)
			supervised_loss = np.mean(supervised_loss.detach().cpu().numpy())

			print('loss sumary', epoch_loss, supervised_loss)

			all_tr_loss_supervise.append(supervised_loss)
			all_tr_loss_all.append(epoch_loss)


			all_tr_loss.append(epoch_loss)

			if (args.local_rank == 0):
				print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
																		  epoch_loss.item()))
				writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
				writer.add_scalar('Train_loss', epoch_loss.item(), epoch)

				plt.plot(all_tr_loss_supervise, label="Supervise")
				plt.plot(all_tr_loss_all, label="Supervise + Psuedo")
				plt.xlabel('Epoch')
				plt.ylabel('Loss')
				plt.title('Loss')
				plt.legend()
				plt.savefig('TrainingLoss_HATs_omni-seg_15classes_pretrained_cls_sls_%s.png' % (os.path.basename(args.snapshot_dir)))
				plt.clf()

				row = len(df_loss)
				df_loss.loc[row] = [epoch, supervised_loss, epoch_loss]
				df_loss.to_csv('TrainingLoss_HATs_omni-seg_15classes_pretrained_cls_sls.csv')


			if (epoch >= 0) and (args.local_rank == 0) and (((epoch % 10 == 0) and (epoch >= 800)) or (epoch % 1 == 0)):
				print('save validation image ...')

				model.eval()
				# semi_pool_image = ImagePool(8 * 6)
				task0_pool_image = ImagePool(8)
				task0_pool_mask = ImagePool(8)
				task0_scale = []
				task0_layer = []

				task1_pool_image = ImagePool(8)
				task1_pool_mask = ImagePool(8)
				task1_scale = []
				task1_layer = []

				task2_pool_image = ImagePool(8)
				task2_pool_mask = ImagePool(8)
				task2_scale = []
				task2_layer = []

				task3_pool_image = ImagePool(8)
				task3_pool_mask = ImagePool(8)
				task3_scale = []
				task3_layer = []

				task4_pool_image = ImagePool(8)
				task4_pool_mask = ImagePool(8)
				task4_scale = []
				task4_layer = []

				task5_pool_image = ImagePool(8)
				task5_pool_mask = ImagePool(8)
				task5_scale = []
				task5_layer = []

				task6_pool_image = ImagePool(8)
				task6_pool_mask = ImagePool(8)
				task6_scale = []
				task6_layer = []

				task7_pool_image = ImagePool(8)
				task7_pool_mask = ImagePool(8)
				task7_scale = []
				task7_layer = []


				'extend to 15 classes'
				task8_pool_image = ImagePool(8)
				task8_pool_mask = ImagePool(8)
				task8_scale = []
				task8_layer = []

				task9_pool_image = ImagePool(8)
				task9_pool_mask = ImagePool(8)
				task9_scale = []
				task9_layer = []


				task10_pool_image = ImagePool(8)
				task10_pool_mask = ImagePool(8)
				task10_scale = []
				task10_layer = []


				task11_pool_image = ImagePool(8)
				task11_pool_mask = ImagePool(8)
				task11_scale = []
				task11_layer = []


				task12_pool_image = ImagePool(8)
				task12_pool_mask = ImagePool(8)
				task12_scale = []
				task12_layer = []


				task13_pool_image = ImagePool(8)
				task13_pool_mask = ImagePool(8)
				task13_scale = []
				task13_layer = []


				task14_pool_image = ImagePool(8)
				task14_pool_mask = ImagePool(8)
				task14_scale = []
				task14_layer = []


				val_loss = np.zeros((task_num))
				val_F1 = np.zeros((task_num))
				val_Dice = np.zeros((task_num))
				val_TPR = np.zeros((task_num))
				val_PPV = np.zeros((task_num))
				cnt = np.zeros((task_num))

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

							# if now_task <= 9 :
							#     continue

							if now_task == 0:
								task0_pool_image.add(imgs[ki].unsqueeze(0))
								task0_pool_mask.add(lbls[ki].unsqueeze(0))
								task0_scale.append((s_ids[ki]))
								task0_layer.append((l_ids[ki]))
							elif now_task == 1:
								task1_pool_image.add(imgs[ki].unsqueeze(0))
								task1_pool_mask.add(lbls[ki].unsqueeze(0))
								task1_scale.append((s_ids[ki]))
								task1_layer.append((l_ids[ki]))
							elif now_task == 2:
								task2_pool_image.add(imgs[ki].unsqueeze(0))
								task2_pool_mask.add(lbls[ki].unsqueeze(0))
								task2_scale.append((s_ids[ki]))
								task2_layer.append((l_ids[ki]))
							elif now_task == 3:
								task3_pool_image.add(imgs[ki].unsqueeze(0))
								task3_pool_mask.add(lbls[ki].unsqueeze(0))
								task3_scale.append((s_ids[ki]))
								task3_layer.append((l_ids[ki]))
							elif now_task == 4:
								task4_pool_image.add(imgs[ki].unsqueeze(0))
								task4_pool_mask.add(lbls[ki].unsqueeze(0))
								task4_scale.append((s_ids[ki]))
								task4_layer.append((l_ids[ki]))
							elif now_task == 5:
								task5_pool_image.add(imgs[ki].unsqueeze(0))
								task5_pool_mask.add(lbls[ki].unsqueeze(0))
								task5_scale.append((s_ids[ki]))
								task5_layer.append((l_ids[ki]))
							elif now_task == 6:
								task6_pool_image.add(imgs[ki].unsqueeze(0))
								task6_pool_mask.add(lbls[ki].unsqueeze(0))
								task6_scale.append((s_ids[ki]))
								task6_layer.append((l_ids[ki]))
							elif now_task == 7:
								task7_pool_image.add(imgs[ki].unsqueeze(0))
								task7_pool_mask.add(lbls[ki].unsqueeze(0))
								task7_scale.append((s_ids[ki]))
								task7_layer.append((l_ids[ki]))


								'extend to 15 classes'
							elif now_task == 8:
								task8_pool_image.add(imgs[ki].unsqueeze(0))
								task8_pool_mask.add(lbls[ki].unsqueeze(0))
								task8_scale.append((s_ids[ki]))
								task8_layer.append((l_ids[ki]))


							elif now_task == 9:
								task9_pool_image.add(imgs[ki].unsqueeze(0))
								task9_pool_mask.add(lbls[ki].unsqueeze(0))
								task9_scale.append((s_ids[ki]))
								task9_layer.append((l_ids[ki]))

							elif now_task == 10:
								task10_pool_image.add(imgs[ki].unsqueeze(0))
								task10_pool_mask.add(lbls[ki].unsqueeze(0))
								task10_scale.append((s_ids[ki]))
								task10_layer.append((l_ids[ki]))

							elif now_task == 11:
								task11_pool_image.add(imgs[ki].unsqueeze(0))
								task11_pool_mask.add(lbls[ki].unsqueeze(0))
								task11_scale.append((s_ids[ki]))
								task11_layer.append((l_ids[ki]))

							elif now_task == 12:
								task12_pool_image.add(imgs[ki].unsqueeze(0))
								task12_pool_mask.add(lbls[ki].unsqueeze(0))
								task12_scale.append((s_ids[ki]))
								task12_layer.append((l_ids[ki]))

							elif now_task == 13:
								task13_pool_image.add(imgs[ki].unsqueeze(0))
								task13_pool_mask.add(lbls[ki].unsqueeze(0))
								task13_scale.append((s_ids[ki]))
								task13_layer.append((l_ids[ki]))

							elif now_task == 14:
								task14_pool_image.add(imgs[ki].unsqueeze(0))
								task14_pool_mask.add(lbls[ki].unsqueeze(0))
								task14_scale.append((s_ids[ki]))
								task14_layer.append((l_ids[ki]))


						output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/','/Data4/HATs/validation_'), str(epoch))
						#output_folder = os.path.join('/Data/DoDNet/a_DynConv/validation_noscale_0829', str(epoch))
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
							for bi in range(len(scales)):
								scales[bi] = task0_scale.pop(0)
								layers[bi] = task0_layer.pop(0)

							preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
							preds[:,:,:512,:512] = model(images[:,:,:512,:512], torch.ones(batch_size).cuda()*0, scales)
							preds[:,:,:512,512:] = model(images[:,:,:512,512:], torch.ones(batch_size).cuda()*0, scales)
							preds[:,:,512:,512:] = model(images[:,:,512:,512:], torch.ones(batch_size).cuda()*0, scales)
							preds[:,:,512:,:512] = model(images[:,:,512:,:512], torch.ones(batch_size).cuda()*0, scales)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels.long())

							rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi, ...].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
										   prediction.detach().cpu().numpy())

						'cortex'
						if task1_pool_image.num_imgs >= batch_size:
							images = task1_pool_image.query(batch_size)
							labels = task1_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task1_scale.pop(0)
								layers[bi] = task1_layer.pop(0)

							preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
							preds[:, :, :512, :512] = model(images[:, :, :512, :512],
															   torch.ones(batch_size).cuda() * 1, scales)
							preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
															   torch.ones(batch_size).cuda() * 1, scales)
							preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
															   torch.ones(batch_size).cuda() * 1, scales)
							preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
															   torch.ones(batch_size).cuda() * 1, scales)

							now_task = torch.tensor(1)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels.long())
							rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi, ...].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
										   prediction.detach().cpu().numpy())


						'cortex_in'
						if task2_pool_image.num_imgs >= batch_size:
							images = task2_pool_image.query(batch_size)
							labels = task2_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task2_scale.pop(0)
								layers[bi] = task2_layer.pop(0)

							preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
							preds[:, :, :512, :512] = model(images[:, :, :512, :512],
															   torch.ones(batch_size).cuda() * 2, scales)
							preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
															   torch.ones(batch_size).cuda() * 2, scales)
							preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
															   torch.ones(batch_size).cuda() * 2, scales)
							preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
															   torch.ones(batch_size).cuda() * 2, scales)

							now_task = torch.tensor(2)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels.long())
							rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi, ...].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
										   prediction.detach().cpu().numpy())


						'cortex_middle'
						if task3_pool_image.num_imgs >= batch_size:
							images = task3_pool_image.query(batch_size)
							labels = task3_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task3_scale.pop(0)
								layers[bi] = task3_layer.pop(0)

							preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
							preds[:, :, :512, :512] = model(images[:, :, :512, :512],
															   torch.ones(batch_size).cuda() * 3, scales)
							preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
															   torch.ones(batch_size).cuda() * 3, scales)
							preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
															   torch.ones(batch_size).cuda() * 3, scales)
							preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
															   torch.ones(batch_size).cuda() * 3, scales)

							now_task = torch.tensor(3)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels.long())
							rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi, ...].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
										   prediction.detach().cpu().numpy())


						'cortex_out'
						if task4_pool_image.num_imgs >= batch_size:
							images = task4_pool_image.query(batch_size)
							labels = task4_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task4_scale.pop(0)
								layers[bi] = task4_layer.pop(0)

							preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
							preds[:, :, :512, :512] = model(images[:, :, :512, :512],
															   torch.ones(batch_size).cuda() * 4, scales)
							preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
															   torch.ones(batch_size).cuda() * 4, scales)
							preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
															   torch.ones(batch_size).cuda() * 4, scales)
							preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
															   torch.ones(batch_size).cuda() * 4, scales)

							now_task = torch.tensor(4)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels.long())
							rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi, ...].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
										   prediction.detach().cpu().numpy())


						'dt'
						if task5_pool_image.num_imgs >= batch_size:
							images = task5_pool_image.query(batch_size)
							labels = task5_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task5_scale.pop(0)
								layers[bi] = task5_layer.pop(0)
							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*5, scales)

							now_task = torch.tensor(5)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 128, 384, 128, 384
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi,128:384,128:384]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,384:640,384:640].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
										   prediction.detach().cpu().numpy())

						'pt'
						if task6_pool_image.num_imgs >= batch_size:
							images = task6_pool_image.query(batch_size)
							labels = task6_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task6_scale.pop(0)
								layers[bi] = task6_layer.pop(0)

							preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 6, scales)

							now_task = torch.tensor(6)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
							rmin, rmax, cmin, cmax = 128, 384, 128, 384
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi, 128:384, 128:384]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi, 384:640, 384:640].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
										   prediction.detach().cpu().numpy())

						'cap'
						if task7_pool_image.num_imgs >= batch_size:
							images = task7_pool_image.query(batch_size)
							labels = task7_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task7_scale.pop(0)
								layers[bi] = task7_layer.pop(0)

							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*7, scales)

							now_task = torch.tensor(7)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 128, 384, 128, 384
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi,128:384,128:384]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,384:640,384:640].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
										   prediction.detach().cpu().numpy())

						'tuft'
						if task8_pool_image.num_imgs >= batch_size:
							images = task8_pool_image.query(batch_size)
							labels = task8_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task8_scale.pop(0)
								layers[bi] = task8_layer.pop(0)

							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*8, scales)

							now_task = torch.tensor(8)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 128, 384, 128, 384
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi,128:384,128:384]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,384:640,384:640].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
										   prediction.detach().cpu().numpy())


						'art'
						if task9_pool_image.num_imgs >= batch_size:
							images = task9_pool_image.query(batch_size)
							labels = task9_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task9_scale.pop(0)
								layers[bi] = task9_layer.pop(0)

							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*9, scales)

							now_task = torch.tensor(9)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 128, 384, 128, 384
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi,128:384,128:384]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,384:640,384:640].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
										   prediction.detach().cpu().numpy())


						'ptc'
						if task10_pool_image.num_imgs >= batch_size:
							images = task10_pool_image.query(batch_size)
							labels = task10_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task10_scale.pop(0)
								layers[bi] = task10_layer.pop(0)

							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*10, scales)

							now_task = torch.tensor(10)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 128, 384, 128, 384
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi,128:384,128:384]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,384:640,384:640].detach().cpu().numpy())
								plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
										   prediction.detach().cpu().numpy())


						'mv'
						if task11_pool_image.num_imgs >= batch_size:
							images = task11_pool_image.query(batch_size)
							labels = task11_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task11_scale.pop(0)
								layers[bi] = task11_layer.pop(0)
							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda() * 11, scales)

							now_task = torch.tensor(11)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 0, 512, 0, 512
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
															 cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,256:768,256:768].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,256:768,256:768].detach().cpu().numpy())
								plt.imsave(
									os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
									prediction.detach().cpu().numpy())


						'pod'
						if task12_pool_image.num_imgs >= batch_size:
							images = task12_pool_image.query(batch_size)
							labels = task12_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task12_scale.pop(0)
								layers[bi] = task12_layer.pop(0)
							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda() * 12, scales)

							now_task = torch.tensor(12)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 0, 512, 0, 512
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
															 cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,256:768,256:768].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,256:768,256:768].detach().cpu().numpy())
								plt.imsave(
									os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
									prediction.detach().cpu().numpy())

						'mes'
						if task13_pool_image.num_imgs >= batch_size:
							images = task13_pool_image.query(batch_size)
							labels = task13_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task13_scale.pop(0)
								layers[bi] = task13_layer.pop(0)
							preds = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda() * 13, scales)

							now_task = torch.tensor(13)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
							rmin, rmax, cmin, cmax = 0, 512, 0, 512
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
															 cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi,:,256:768,256:768].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi,256:768,256:768].detach().cpu().numpy())
								plt.imsave(
									os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
									prediction.detach().cpu().numpy())

						'smooth'
						if task14_pool_image.num_imgs >= batch_size:
							images = task14_pool_image.query(batch_size)
							labels = task14_pool_mask.query(batch_size)
							scales = torch.ones(batch_size).cuda()
							layers = torch.ones(batch_size).cuda()
							for bi in range(len(scales)):
								scales[bi] = task14_scale.pop(0)
								layers[bi] = task14_layer.pop(0)
							preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 14, scales)

							now_task = torch.tensor(14)

							now_preds = torch.argmax(preds, 1) == 1
							now_preds_onehot = one_hot_3D(now_preds.long())

							labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
							rmin, rmax, cmin, cmax = 0, 512, 0, 512
							F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
															 cmax)

							val_F1[now_task] += F1
							val_Dice[now_task] += DICE
							val_TPR[now_task] += TPR
							val_PPV[now_task] += PPV
							cnt[now_task] += 1

							for pi in range(len(images)):
								prediction = now_preds[pi]
								num = len(glob.glob(os.path.join(output_folder, '*')))
								out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
								img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
								plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
										   img)
								plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
										   labels[pi, 256:768, 256:768].detach().cpu().numpy())
								plt.imsave(
									os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
									prediction.detach().cpu().numpy())


					'last round clean up the image pool'

					'medulla'
					if task0_pool_image.num_imgs > 0:
						batch_size = task0_pool_image.num_imgs
						images = task0_pool_image.query(batch_size)
						labels = task0_pool_mask.query(batch_size)
						now_task = torch.tensor(0)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task0_scale.pop(0)
							layers[bi] = task0_layer.pop(0)

						preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
						preds[:, :, :512, :512] = model(images[:, :, :512, :512], torch.ones(batch_size).cuda() * 0,
						                                scales)
						preds[:, :, :512, 512:] = model(images[:, :, :512, 512:], torch.ones(batch_size).cuda() * 0,
						                                scales)
						preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:], torch.ones(batch_size).cuda() * 0,
						                                scales)
						preds[:, :, 512:, :512] = model(images[:, :, 512:, :512], torch.ones(batch_size).cuda() * 0,
						                                scales)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels.long())

						rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, ...].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'cortex'
					if task1_pool_image.num_imgs > 0:
						batch_size = task1_pool_image.num_imgs
						images = task1_pool_image.query(batch_size)
						labels = task1_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task1_scale.pop(0)
							layers[bi] = task1_layer.pop(0)

						preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
						preds[:, :, :512, :512] = model(images[:, :, :512, :512],
						                                torch.ones(batch_size).cuda() * 1, scales)
						preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
						                                torch.ones(batch_size).cuda() * 1, scales)
						preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
						                                torch.ones(batch_size).cuda() * 1, scales)
						preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
						                                torch.ones(batch_size).cuda() * 1, scales)

						now_task = torch.tensor(1)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels.long())
						rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, ...].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'cortex_in'
					if task2_pool_image.num_imgs > 0:
						batch_size = task2_pool_image.num_imgs
						images = task2_pool_image.query(batch_size)
						labels = task2_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task2_scale.pop(0)
							layers[bi] = task2_layer.pop(0)

						preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
						preds[:, :, :512, :512] = model(images[:, :, :512, :512],
						                                torch.ones(batch_size).cuda() * 2, scales)
						preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
						                                torch.ones(batch_size).cuda() * 2, scales)
						preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
						                                torch.ones(batch_size).cuda() * 2, scales)
						preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
						                                torch.ones(batch_size).cuda() * 2, scales)

						now_task = torch.tensor(2)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels.long())
						rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, ...].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'cortex_middle'
					if task3_pool_image.num_imgs > 0:
						batch_size = task3_pool_image.num_imgs
						images = task3_pool_image.query(batch_size)
						labels = task3_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task3_scale.pop(0)
							layers[bi] = task3_layer.pop(0)

						preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
						preds[:, :, :512, :512] = model(images[:, :, :512, :512],
						                                torch.ones(batch_size).cuda() * 3, scales)
						preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
						                                torch.ones(batch_size).cuda() * 3, scales)
						preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
						                                torch.ones(batch_size).cuda() * 3, scales)
						preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
						                                torch.ones(batch_size).cuda() * 3, scales)

						now_task = torch.tensor(3)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels.long())
						rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, ...].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'cortex_out'
					if task4_pool_image.num_imgs > 0:
						batch_size = task4_pool_image.num_imgs
						images = task4_pool_image.query(batch_size)
						labels = task4_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task4_scale.pop(0)
							layers[bi] = task4_layer.pop(0)

						preds = torch.zeros((batch_size, 2, 1024, 1024)).cuda()
						preds[:, :, :512, :512] = model(images[:, :, :512, :512],
						                                torch.ones(batch_size).cuda() * 4, scales)
						preds[:, :, :512, 512:] = model(images[:, :, :512, 512:],
						                                torch.ones(batch_size).cuda() * 4, scales)
						preds[:, :, 512:, 512:] = model(images[:, :, 512:, 512:],
						                                torch.ones(batch_size).cuda() * 4, scales)
						preds[:, :, 512:, :512] = model(images[:, :, 512:, :512],
						                                torch.ones(batch_size).cuda() * 4, scales)

						now_task = torch.tensor(4)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels.long())
						rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, ...].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'dt'
					if task5_pool_image.num_imgs > 0:
						batch_size = task5_pool_image.num_imgs
						images = task5_pool_image.query(batch_size)
						labels = task5_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task5_scale.pop(0)
							layers[bi] = task5_layer.pop(0)
						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 5, scales)

						now_task = torch.tensor(5)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 128, 384, 128, 384
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi, 128:384, 128:384]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 384:640, 384:640].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'pt'
					if task6_pool_image.num_imgs > 0:
						batch_size = task6_pool_image.num_imgs
						images = task6_pool_image.query(batch_size)
						labels = task6_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task6_scale.pop(0)
							layers[bi] = task6_layer.pop(0)

						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 6, scales)

						now_task = torch.tensor(6)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 128, 384, 128, 384
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi, 128:384, 128:384]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 384:640, 384:640].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'cap'
					if task7_pool_image.num_imgs > 0:
						batch_size = task7_pool_image.num_imgs
						images = task7_pool_image.query(batch_size)
						labels = task7_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task7_scale.pop(0)
							layers[bi] = task7_layer.pop(0)

						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 7, scales)

						now_task = torch.tensor(7)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 128, 384, 128, 384
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi, 128:384, 128:384]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 384:640, 384:640].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'tuft'
					if task8_pool_image.num_imgs > 0:
						batch_size = task8_pool_image.num_imgs
						images = task8_pool_image.query(batch_size)
						labels = task8_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task8_scale.pop(0)
							layers[bi] = task8_layer.pop(0)

						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 8, scales)

						now_task = torch.tensor(8)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 128, 384, 128, 384
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi, 128:384, 128:384]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 384:640, 384:640].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'art'
					if task9_pool_image.num_imgs > 0:
						batch_size = task9_pool_image.num_imgs
						images = task9_pool_image.query(batch_size)
						labels = task9_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task9_scale.pop(0)
							layers[bi] = task9_layer.pop(0)

						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 9, scales)

						now_task = torch.tensor(9)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 128, 384, 128, 384
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi, 128:384, 128:384]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 384:640, 384:640].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'ptc'
					if task10_pool_image.num_imgs > 0:
						batch_size = task10_pool_image.num_imgs
						images = task10_pool_image.query(batch_size)
						labels = task10_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task10_scale.pop(0)
							layers[bi] = task10_layer.pop(0)

						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 10, scales)

						now_task = torch.tensor(10)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 128, 384, 128, 384
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi, 128:384, 128:384]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 384:640, 384:640].detach().cpu().numpy())
							plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
							           prediction.detach().cpu().numpy())

					'mv'
					if task11_pool_image.num_imgs > 0:
						batch_size = task11_pool_image.num_imgs
						images = task11_pool_image.query(batch_size)
						labels = task11_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task11_scale.pop(0)
							layers[bi] = task11_layer.pop(0)
						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 11, scales)

						now_task = torch.tensor(11)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 0, 512, 0, 512
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
						                                 cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 256:768, 256:768].detach().cpu().numpy())
							plt.imsave(
								os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
								prediction.detach().cpu().numpy())

					'pod'
					if task12_pool_image.num_imgs > 0:
						batch_size = task12_pool_image.num_imgs
						images = task12_pool_image.query(batch_size)
						labels = task12_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task12_scale.pop(0)
							layers[bi] = task12_layer.pop(0)
						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 12, scales)

						now_task = torch.tensor(12)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 0, 512, 0, 512
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
						                                 cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 256:768, 256:768].detach().cpu().numpy())
							plt.imsave(
								os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
								prediction.detach().cpu().numpy())

					'mes'
					if task13_pool_image.num_imgs > 0:
						batch_size = task13_pool_image.num_imgs
						images = task13_pool_image.query(batch_size)
						labels = task13_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task13_scale.pop(0)
							layers[bi] = task13_layer.pop(0)
						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 13, scales)

						now_task = torch.tensor(13)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 0, 512, 0, 512
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
						                                 cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 256:768, 256:768].detach().cpu().numpy())
							plt.imsave(
								os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
								prediction.detach().cpu().numpy())

					'smooth'
					if task14_pool_image.num_imgs > 0:
						batch_size = task14_pool_image.num_imgs
						images = task14_pool_image.query(batch_size)
						labels = task14_pool_mask.query(batch_size)
						scales = torch.ones(batch_size).cuda()
						layers = torch.ones(batch_size).cuda()
						for bi in range(len(scales)):
							scales[bi] = task14_scale.pop(0)
							layers[bi] = task14_layer.pop(0)
						preds = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 14, scales)

						now_task = torch.tensor(14)

						now_preds = torch.argmax(preds, 1) == 1
						now_preds_onehot = one_hot_3D(now_preds.long())

						labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
						rmin, rmax, cmin, cmax = 0, 512, 0, 512
						F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
						                                 cmax)

						val_F1[now_task] += F1
						val_Dice[now_task] += DICE
						val_TPR[now_task] += TPR
						val_PPV[now_task] += PPV
						cnt[now_task] += 1

						for pi in range(len(images)):
							prediction = now_preds[pi]
							num = len(glob.glob(os.path.join(output_folder, '*')))
							out_image = images[pi, :, 256:768, 256:768].permute([1, 2, 0]).detach().cpu().numpy()
							img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
							plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
							           img)
							plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
							           labels[pi, 256:768, 256:768].detach().cpu().numpy())
							plt.imsave(
								os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
								prediction.detach().cpu().numpy())


					avg_val_F1 = val_F1 / cnt
					avg_val_Dice = val_Dice / cnt
					avg_val_TPR = val_TPR / cnt
					avg_val_PPV = val_PPV / cnt

					print('Validate \n 0medulla_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}'
						  ' \n 1cortex_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 2cortexin_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 3cortexmiddle_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 4cortexout_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 5dt_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 6pt_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 7cap_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 8tuft_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 9art_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 10ptc_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 11mv_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 12pod_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 13mes_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  ' \n 14smooth_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
						  .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item(),
								  avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item(),
								  avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item(),
								  avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item(),
								  avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item(),
								  avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item(),
								  avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_TPR[6].item(), avg_val_PPV[6].item(),
								  avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_TPR[7].item(), avg_val_PPV[7].item(),
								  avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_TPR[8].item(), avg_val_PPV[8].item(),
								  avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_TPR[9].item(), avg_val_PPV[9].item(),
								  avg_val_F1[10].item(), avg_val_Dice[10].item(), avg_val_TPR[10].item(), avg_val_PPV[10].item(),
								  avg_val_F1[11].item(), avg_val_Dice[11].item(), avg_val_TPR[11].item(), avg_val_PPV[11].item(),
								  avg_val_F1[12].item(), avg_val_Dice[12].item(), avg_val_TPR[12].item(), avg_val_PPV[12].item(),
								  avg_val_F1[13].item(), avg_val_Dice[13].item(), avg_val_TPR[13].item(), avg_val_PPV[13].item(),
								  avg_val_F1[14].item(), avg_val_Dice[14].item(), avg_val_TPR[14].item(), avg_val_PPV[14].item(),))

					df = pd.DataFrame(columns = ['task','F1','Dice','TPR','PPV'])
					df.loc[0] = ['0medulla', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item()]
					df.loc[1] = ['1cortex', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item()]
					df.loc[2] = ['2cortexin', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item()]
					df.loc[3] = ['3cortexmiddle', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item()]
					df.loc[4] = ['4cortexout', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item()]
					df.loc[5] = ['5dt', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item()]
					df.loc[6] = ['6pt', avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_TPR[6].item(), avg_val_PPV[6].item()]
					df.loc[7] = ['7cap', avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_TPR[7].item(), avg_val_PPV[7].item()]
					df.loc[8] = ['8tuft', avg_val_F1[8].item(), avg_val_Dice[8].item(), avg_val_TPR[8].item(), avg_val_PPV[8].item()]
					df.loc[9] = ['9art', avg_val_F1[9].item(), avg_val_Dice[9].item(), avg_val_TPR[9].item(), avg_val_PPV[9].item()]
					df.loc[10] = ['10ptc', avg_val_F1[10].item(), avg_val_Dice[10].item(), avg_val_TPR[10].item(), avg_val_PPV[10].item()]
					df.loc[11] = ['11mv', avg_val_F1[11].item(), avg_val_Dice[11].item(), avg_val_TPR[11].item(), avg_val_PPV[11].item()]
					df.loc[12] = ['12pod', avg_val_F1[12].item(), avg_val_Dice[12].item(), avg_val_TPR[12].item(), avg_val_PPV[12].item()]
					df.loc[13] = ['13mes', avg_val_F1[13].item(), avg_val_Dice[13].item(), avg_val_TPR[13].item(),avg_val_PPV[13].item()]
					df.loc[14] = ['14smooth', avg_val_F1[14].item(), avg_val_Dice[14].item(), avg_val_TPR[14].item(),avg_val_PPV[14].item()]

					df.to_csv(os.path.join(output_folder,'validation_result.csv'))


				print('save model ...')
				if args.FP16:
					checkpoint = {
						'model': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'amp': amp.state_dict()
					}
					torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
				else:
					torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))

			if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
				print('save model ...')
				if args.FP16:
					checkpoint = {
						'model': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'amp': amp.state_dict()
					}
					torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
				else:
					torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
				break

		end = timeit.default_timer()
		print(end - start, 'seconds')


if __name__ == '__main__':
	main()
