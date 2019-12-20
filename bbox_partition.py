import os
import torch.utils.data as data
import torch
from PIL import Image
import random
from tqdm import tqdm
import shutil

DATASET_BASE = '/scratch/ag7387/cv/project/cat_data/Img'
FOLDER_BASE = '/scratch/ag7387/cv/project/cat_data'
crop = True
bbox = {}
partition_dict = {}

def read_lines(path):
	with open(path) as fin:
		lines = fin.readlines()[2:]
		lines = list(filter(lambda x: len(x) > 0, lines))
	pairs = list(map(lambda x: x.strip().split(), lines))
	return pairs

def read_bbox():
	list_bbox = os.path.join(FOLDER_BASE, r'Anno', r'list_bbox.txt')
	pairs = read_lines(list_bbox)
	for k, x1, y1, x2, y2 in pairs:
		bbox[k] = list(map(int, [x1, y1, x2, y2]))
		bbox[k.replace('img/', 'img_highres/')] = list(map(int, [x1, y1, x2, y2]))

def read_crop(img_path):
	img_full_path = os.path.join(DATASET_BASE, img_path)
	with open(img_full_path, 'rb') as f:
		with Image.open(f) as img:
			img = img.convert('RGB')
	if crop:
		x1, y1, x2, y2 = bbox[img_path]
		if x1 < x2 <= img.size[0] and y1 < y2 <= img.size[1]:
			img = img.crop((x1, y1, x2, y2))
	return img

def read_partition():
	with open(os.path.join(FOLDER_BASE, r'Eval', r'list_eval_partition.txt')) as fin:
		lines = fin.readlines()[2:]
		for line in lines:
			key, val = line.split()
			partition_dict[key] = val

read_bbox()
read_partition()

if 'train' in os.listdir(DATASET_BASE):
	shutil.rmtree(os.path.join(DATASET_BASE, 'train'))

if 'test' in os.listdir(DATASET_BASE):
	shutil.rmtree(os.path.join(DATASET_BASE, 'test'))

if 'val' in os.listdir(DATASET_BASE):
	shutil.rmtree(os.path.join(DATASET_BASE, 'val'))

if 'unk' in os.listdir(DATASET_BASE):
	shutil.rmtree(os.path.join(DATASET_BASE, 'unk'))

os.mkdir(os.path.join(DATASET_BASE, 'train'))
os.mkdir(os.path.join(DATASET_BASE, 'test'))
os.mkdir(os.path.join(DATASET_BASE, 'val'))

#folder 50 cats

folders = [folder for folder in os.listdir(os.path.join(DATASET_BASE, 'img')) if not folder.startswith('.') and not folder.endswith('.zip)')]
cats = list(set([i.split('_')[-1] for i in folders if not i.startswith('.')]))
for cat in cats:
	os.mkdir(os.path.join(DATASET_BASE, 'train', cat))
	os.mkdir(os.path.join(DATASET_BASE, 'test', cat))
	os.mkdir(os.path.join(DATASET_BASE, 'val', cat))

count = 0
for folder in tqdm(folders):
	cat = folder.split('_')[-1]
	if folder!='img.zip':
		images = os.listdir(os.path.join(DATASET_BASE, 'img', folder))
		for image in images:
			path = os.path.join(DATASET_BASE, 'img', folder, image)
			pathfile = os.path.join('img', folder, image)
			if pathfile in partition_dict and pathfile in bbox:
				trainbool = partition_dict[pathfile]
				read_crop(pathfile).save(os.path.join(DATASET_BASE, trainbool, cat, folder+'_'+image))
			else:
				count = count +1
				pass
		
print("Error count")
print(count)
