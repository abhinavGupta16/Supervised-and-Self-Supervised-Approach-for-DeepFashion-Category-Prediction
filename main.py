from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from awesome_net import AWESOME_NET
from awesome_net2 import AWESOME_NET2
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
from cae import CAE
import os
from awesome_net3 import AWESOME_NET3
from sklearn.cluster import KMeans
import torchvision
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--test_path', type=str, default=None, metavar='D')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--checkpoint', type=str, default=None, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")

args = parser.parse_args()

img_path = args.test_path


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Data Initialization and Loading
from data import initialize_data, data_transforms, train_transforms, jitter_hue,jitter_brightness,jitter_saturation,jitter_contrast,rotate,hvflip,shear,translate,center,hflip,vflip,grayscale # data.py in the same folder

mnist = '/scratch/ag7387/cv/project/mnist'
#train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root=mnist,train=True, transform=data_transforms, download=True))

#val_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root=mnist,train=False, transform=data_transforms,download = True))

fashion_folder = '/scratch/ag7387/cv/project/dummy_data'
fashion_dataset = "/scratch/ag7387/cv/project/cat_data/Img"
stl_10 = '/scratch/ag7387/cv/project/stl-10/binary'
nclasses = 10
#train_loader = torch.utils.data.DataLoader( torchvision.datasets.STL10(stl_10, split='train', transform=data_transforms))

#val_loader = torch.utils.data.DataLoader( torchvision.datasets.STL10(stl_10, split='test', transform=data_transforms))
'''
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(stl_10,#args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(stl_10,#args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4)
'''

data_f = []

def filter_dataset():
    global data_f
    if data_f:
       return data_f
    dataset = datasets.ImageFolder(fashion_dataset + '/val',                                                                                                                           transform=data_transforms)
    useful = set([1,2,7,13,19,20,25,34,36,37,38,41,42,43])
    dataset_filter = [i for i in tqdm(dataset) if i[1] in useful]
    data_f = dataset_filter
    return dataset_filter


train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(fashion_dataset + '/train',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4)

val_loader = torch.utils.data.DataLoader(filter_dataset(),#datasets.ImageFolder(fashion_dataset + '/val',                                                                                                                           transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=4)

print(args.lr)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script

from model import Net, SpatialNet
#model = Net()
#model = SpatialNet()

#model = AWESOME_NET()
model = AWESOME_NET3()
#model = AWESOME_NET2()
#model = CAE()

print(model)

if args.checkpoint is not None:
    print("using checkpoint")
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

#print(model)

model = model.cuda()
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=3,factor=0.1,verbose=True)
#loss = FocalLoss(class_num = 43, gamma=1.5, size_average = False)


def train(epoch):
    model.train()
    correct = 0
    train_loss = 0
    train_limit = 0
    error = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        try:
            batch_num = batch_idx+1
            data = Variable(data).cuda()
            optimizer.zero_grad()
            encoder_output,output = model(data)
            loss = F.mse_loss(output, data).cuda()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if train_limit >= 100:
                break
        except:
            error = error+1
            print("error")
    if batch_num % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_num * len(data), len(train_loader.dataset),
            100. * batch_num / len(train_loader), loss.item()))
        
    return 100. * correct / len(train_loader.dataset) ,train_loss / len(train_loader.dataset)

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()
        encoder_output,output = model(data)
        validation_loss += F.mse_loss(output, data).item() # sum up batch loss
        #pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        #correct += pred.eq(data.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss,2))
    return 100. * correct / len(val_loader.dataset), validation_loss


def show_image(x):
    plt.imshow(np.clip(x, 0, 1))


def visualize(img, out, encoder_out, name):
    """Draws original, encoded and decoded images"""
    # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
    plt.figure()
    plt.subplot(1,3,1)
    plt.title("Image 1")
    show_image(img/255)
    
    plt.subplot(1,3,2)
    plt.title("Image 2")
    if encoder_out is not None:
        show_image(encoder_out)
        
    plt.subplot(1,3,3)
    plt.title("Image 3")
    show_image(out)
    
    plt.savefig(name + ".jpeg")


def test(img_path):
    #dataset = torchvision.datasets.STL10(stl_10, split='test')
    dataset = datasets.ImageFolder(fashion_dataset + '/val')
    #print(np.array(dataset[0][0]).shape)
    image = dataset[2][0]
    model.eval()
    #image = Image.open(img_path)
    #x = TF.to_tensor(image)
    x = data_transforms(image).cuda()
    print('before')
    x.unsqueeze_(0)
    print(x.shape)
    with torch.no_grad():
        encoder_output,output = model(x)
    print('output')
    output_image = output.cpu().detach().numpy()
    print(output_image.shape)
    output_image = np.squeeze(output_image)
    #print(output_image.shape)
    output_image = output_image.transpose(1,2,0)
    print('encoder_output')
    encoder_output_image = encoder_output.cpu().detach().numpy()
    print(encoder_output_image.shape)
    encoder_output_image = np.squeeze(encoder_output_image)
    #print(output_image.shape)
    encoder_output_image = encoder_output_image.transpose(1,2,0)
    visualize(np.array(image), output_image, None, "test")
    #print(output_image.shape)
    #matplotlib.image.imsave('name.png', output_image)
    #im = Image.fromarray(output_image, mode='RGB')
    #im.save("new_image.jpeg")


def kmeans():
    print("---------kmeans--------------")
    model.eval()
    kmeans_list = []
    error = 0
    for data, target in tqdm(val_loader):
        try:        
            #print(target.item())
            data = Variable(data).cuda()
            encoder_output,output = model(data)
            #print(encoder_output)
            encoder_out = encoder_output.cpu().detach().numpy()
            kmeans_list.append(np.reshape(encoder_out, [-1,16*16*32]))
        except Exception as e:
            error = error + 1
            print("error")
            print(e)
    print("errorcount")
    print(error) 
    print(kmeans_list[-1].shape)
    print(len(kmeans_list))
    pca = PCA(n_components=32*4)
    X = np.vstack(kmeans_list)
    pca_X = pca.fit_transform(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_)
    print("-----cumsum----")
    print(cumsum)
    print(pca_X.shape)
    kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(pca_X) 
    #kmeans = FuzzyKMeans(10).fit(X)
    labels = kmeans.labels_
   
    print(labels)
#    dataset = datasets.ImageFolder(stl_10)#args.data + '/val_images')
    #dataset = torchvision.datasets.STL10(stl_10, split='test')
    #dataset = data_f#filter_dataset(datasets.ImageFolder(fashion_dataset + '/val'))
    dataset = torchvision.datasets.MNIST(root=mnist,train=False, download=True)
    plt.figure()
    sel = np.where(labels == 1)[0]
    accuracy(labels, dataset)
    #quit()
    print(sel[0])
    print(np.array(dataset[sel[0]][0]).shape)
    print(np.array(dataset[sel[1]][0]).shape)
    print(np.array(dataset[sel[2]][0]).shape)
    print(np.array(dataset[sel[2]][0]).transpose(1,2,0).shape)
    output_image = np.array(dataset[0][0])
    visualize(np.array(dataset[sel[0]][0]),np.array(dataset[sel[1]][0])/255,np.array(dataset[sel[2]][0])/255,"kmeans")
    #show_image(output_image/255)
    #plt.title("Reconstructed")

    #plt.savefig("kmeans.jpeg")

def accuracy(labels, dataset):
    i = 0
    actual = 0
    counted = []
    for i in range(nclasses):
        sel = np.where(labels == i)[0]
        targets = []
        for j in sel:
            if dataset[j][1] not in counted:
                targets.append(dataset[j][1])
        if len(targets) == 0:
            continue
        majority = max(targets, key=targets.count)
        print(majority)
        counted.append(majority)
       # for val in targets:
       #     if val == majority:
       #         actual += 1
        actual += targets.count(majority)
     
            
    acc = actual/len(labels) * 100
    print(acc)
        

if img_path is not None:
    #test(img_path)
    kmeans()
    quit()


# step=10
tran_arr=[]
val_arr=[]
tran_acc_arr=[]
val_acc_arr=[]
for epoch in range(1, args.epochs + 1):
    tran_acc, tran_loss = train(epoch)
    val_acc, val_loss = validation()
    # if epoch % step :
    # print("train: " , tran_loss)
    # print("val:" , val)
    tran_arr.append(tran_loss)
    val_arr.append(val_loss)
    tran_acc_arr.append(tran_acc)
    val_acc_arr.append(val_acc)
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
