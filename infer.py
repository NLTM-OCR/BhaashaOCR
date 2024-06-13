from __future__ import division, print_function

import argparse
import random
import json
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import Compose

import datasets.dataset as dataset
import utils
from datasets.ae_transforms import *
from datasets.imprint_dataset import Rescale as IRescale
from models.model import ModelBuilder
from layout import main as layout_main

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
cudnn.deterministic = True
cudnn.benchmark = False
cudnn.enabled = True

class BaseHTR(object):
	def __init__(self, opt):
		self.opt = opt
		self.test_transforms = Compose([
			IRescale(max_width=256, height=96),
			ToTensor()
		])
		self.identity_matrix = torch.tensor(
			[1, 0, 0, 0, 1, 0],
			dtype=torch.float
		).cuda()
		
		##################################################################
		self.test_root = opt.test_root

		if torch.cuda.is_available() and not self.opt.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			self.opt.gpu_id = list(map(int, self.opt.gpu_id.split(',')))
			torch.cuda.set_device(self.opt.gpu_id[0])

		self.test_data = dataset.ImageDataset(
			root=self.test_root,
			voc=self.opt.alphabet,
			transform=self.test_transforms,
			voc_type='file',
			return_list=True
		)
		self.converter = utils.strLabelConverter(
			self.test_data.id2char,
			self.test_data.char2id,
			self.test_data.ctc_blank
		)
		self.nclass = self.test_data.rec_num_classes

		crnn = ModelBuilder(
			96, 256,
			[48,128], [96,256],
			20, [0.05, 0.05],
			'none',
			256, 1, 1,
			self.nclass,
			STN_type='TPS',
			nheads=1,
			stn_attn=None,
			use_loc_bn=False,
			loc_block = 'LocNet',
			CNN='ResCRNN'
		)
		if self.opt.cuda:
			crnn.cuda()
			crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id, dim=1)
		else:
			crnn = torch.nn.DataParallel(crnn, device_ids=self.opt.gpu_id)
		print('Using pretrained model', self.opt.ocr_pretrained)
		crnn.load_state_dict(torch.load(self.opt.ocr_pretrained))
		self.model = crnn
		self.model.eval()
		print('Model loading complete')

		self.init_variables()
		print('Classes: ', self.test_data.voc)
		print('#Test Samples: ', self.test_data.nSamples)

		data_loader = torch.utils.data.DataLoader(
			self.test_data,
			batch_size=64,
			num_workers=2,
			pin_memory=True,
			collate_fn=dataset.collatedict(),
			drop_last=False
		)
		gts = []
		decoded_preds = []
		val_iter = iter(data_loader)
		max_iter = min(np.inf, len(data_loader))
		with torch.no_grad():
			for i in range(max_iter):
				cpu_images, cpu_texts = next(val_iter)
				utils.loadData(self.image, cpu_images)
				output_dict = self.model(self.image)
				batch_size = cpu_images.size(0)

				preds = F.log_softmax(output_dict['probs'], 2)

				preds_size = torch.IntTensor([preds.size(0)] * batch_size)
				_, preds = preds.max(2)
				preds = preds.transpose(1, 0).contiguous().view(-1)
				decoded_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

				gts += list(cpu_texts)
				decoded_preds += list(decoded_pred)

		writepath1 = join(self.opt.out_dir, 'out.json')
		print(writepath1)
		output = {}
		for target, pred in zip(gts, decoded_preds):         
			print(target, pred)
			if str(target) == 'test_-1.jpg':
				print('Discarding the test image used to workaround the single char output bug')
				continue
			output[str(target)] = str(pred)
		with open(writepath1, 'w', encoding='utf-8') as f:
			f.write(json.dumps(output, indent=4))
		return


	def init_variables(self):
		self.image = torch.FloatTensor(64, 3, 96, 256)
		self.text = torch.LongTensor(64 * 5)
		self.length = torch.LongTensor(64)
		if self.opt.cuda:
			self.image = self.image.cuda()
			self.text = self.text.cuda()
			self.length = self.length.cuda()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_path', help='path to the page image')
	parser.add_argument('--ocr_pretrained', default='', help="path to OCR pretrained model")
	parser.add_argument('--layout_pretrained', default='', help="path to Layout pretrained model")
	parser.add_argument('--out_dir', type=str, default="out", help='path to the output folder')
	parser.add_argument('--language', help='Language of inference model to be called')

	parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
	parser.add_argument('--gpu_id', type=str, default='0', help='gpu device ids')
	parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz*')

	opt = parser.parse_args()
	opt.alphabet = f'alphabet/{opt.language}_lexicon.txt'

	opt.test_root = layout_main(opt.image_path, opt.layout_pretrained, opt.out_dir)
	obj = BaseHTR(opt)
