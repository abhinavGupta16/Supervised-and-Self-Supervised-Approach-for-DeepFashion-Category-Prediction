import shutil
import os
from tqdm import tqdm


basedir = '/scratch/ag7387/cv/project/cat_data/Img/'

allcats = os.listdir(basedir+'img_highres')
topcats = list(set([i.split('_')[-1] for i in allcats if not i.startswith('.')]))

if not 'rearranged_img' in os.listdir(basedir):
	#shutil.rmtree(basedir + 'rearranged_img')
	os.mkdir(basedir+'rearranged_img')

for cat in topcats:
	if not os.path.isdir(basedir+'rearranged_img/'+cat):
		os.mkdir(basedir+'rearranged_img/'+cat)

for folder in tqdm(allcats):
	if folder.startswith('.'):
		continue
	allimages = os.listdir(basedir+'img_highres/'+folder)
	for image in allimages:
		if not image.startswith('.') and not folder.startswith('.'):
			source = basedir+'img_highres/'+folder+'/'+image
			dest = basedir+'rearranged_img/'+folder.split('_')[-1]+'/'+folder+'_'+image
			#print(dest)
			# print(os.path.exists(dest))
			if not os.path.exists(dest):
				shutil.copyfile(source, dest)
