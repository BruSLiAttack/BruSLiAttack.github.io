import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.utils.data as data
import numpy as np
from xmodels import *
import torch.backends.cudnn as cudnn
import cv2
import pandas as pd

# ======================== Dataset ========================

def load_data(dataset, data_path=None, batch_size=1):

    if dataset == 'imagenet' and data_path==None:
        data_path = '../../datasets/ImageNet-val'
    elif dataset == 'stl10'and data_path==None:
        data_path = '../../datasets/stl10' 
    elif dataset == 'cifar10'and data_path==None:
        data_path = '../../datasets/cifar10'
    elif dataset == 'cifar100'and data_path==None:
        data_path = '../../datasets/cifar100' 

    if dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'stl10':
        transform = transforms.Compose([transforms.ToTensor()])
        testset = datasets.STL10(root=data_path, split ='test', download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif dataset == 'imagenet':
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])

        testset = datasets.ImageNet(root=data_path, split='val', transform=transform)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader, testset

# ======================== Model ========================

class PretrainedModel():
    def __init__(self,model,dataset='imagenet'):
        self.model = model
        self.dataset = dataset
        
        # ======= CIFAR10 ==========
        if self.dataset == 'cifar10':
            self.mu = torch.Tensor([0.4914, 0.4822, 0.4465]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([0.2023, 0.1994, 0.2010]).float().view(1, 3, 1, 1).cuda()

        # ======= CIFAR100 =========
        elif self.dataset == 'cifar100':
            self.mu = torch.Tensor([0.5071, 0.4865, 0.4409]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([0.2673, 0.2564, 0.2762]).float().view(1, 3, 1, 1).cuda()

        # ======= STL10 ==========
        if self.dataset == 'stl10':
            self.mu = torch.Tensor([0.4467, 0.4398, 0.4066]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([0.2241, 0.2215, 0.2239]).float().view(1, 3, 1, 1).cuda()

        # ======= ImageNet =========
        elif self.dataset == 'imagenet':
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()              


    def predict(self, x):
        
        # shape: [n,c,w,h]
        img = (x - self.mu) / self.sigma
        out = self.model(img)
        return  out

    def predict_label(self, x):
        img = (x - self.mu) / self.sigma
        out = self.model(img)
        out = torch.max(out,1)
        return out[1]

    def __call__(self, x):
        return self.predict(x)

def load_model(net,model_path=None):
    if net == 'resnet50':
        net = models.resnet50(pretrained=True).cuda()
 
    elif net == 'resnet18' :
        if model_path == None:
            model_path = '../models/cifar10_ResNet18.pth'
        net = ResNet18()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        net = net.to(device)
        checkpoint = torch.load(model_path,map_location='cuda:0')
        if 'net' in checkpoint:
            if device == 'cuda:0':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True
            net.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(checkpoint)
    
    elif net == 'resnet9':
        if model_path == None:
            model_path = '../models/stl10-resnet9-norm-ba256-ep500.pth'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = ResNet9(in_channels=3, num_classes=10)
        net = net.to(device)
        checkpoint = torch.load(model_path,map_location='cuda')
        net.load_state_dict(checkpoint)

    net.eval()
    return net

# ======================== Generate starting img ========================

def search_space_init(img,seed=0,scale=1,mode='uni',scale_mode='INTER_LINEAR'):
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # 1. find the distribution of reveresed image (opposite)
    c = img.shape[1]
    w = img.shape[2]//scale
    h = img.shape[3]//scale

    xo = img[0].cpu().numpy().reshape(c,-1)
    x_pd=pd.DataFrame(np.round(1-xo.transpose()), columns=['r', 'g','b'])

    p = (x_pd.sum()/x_pd.shape[0]).to_numpy()
    distri = np.stack((p,1-p),axis = 1)

    # 2. generate customized distribution image in lower scale if scale > 1
    val = [1,0]
    x = np.zeros((c,w,h))

    # =================== generate ===================
    if mode == 'uni': #final selection used for the paper
        for i in(range(c)):
            x[i] = np.random.choice(val, size=(w, h))
    elif mode == 'custom':
        for i in(range(c)):
            x[i] = np.random.choice(val, p=distri[i], size=(w, h))
    elif mode == 'sp':
        x[0] = np.random.choice(val, size=(w, h))
        x[1] = x[0]
        x[2] = x[0]
    elif mode == 'sp_custom':
        x[0] = np.random.choice(val, p=(distri[0]),size=(w, h))
        x[1] = x[0]
        x[2] = x[0]
    elif mode == 'inverted':
        m = 2**8-1 # scale level
        val = np.arange(m+1)
        xt = np.floor(((1-xo)*m)).astype(int)
        distri = np.zeros((c,len(val)))
        for i in range(c):
            for j in (val):
                distri[i,j] = np.count_nonzero(xt[i] == j)
            
        distri /= (w*h)
        for i in(range(c)):
            x[i] = np.random.choice(val, p=distri[i], size=(w, h))
        x /=m
    
    # 3. =============== Upscale ===================
    x = np.transpose(x,(1,2,0))

    if scale_mode == 'INTER_LINEAR':
        x = cv2.resize(x, (0, 0), fx=scale, fy=scale)
    elif scale_mode == 'INTER_NEAREST':
        x = cv2.resize(x, (scale*w,scale*h), interpolation=cv2.INTER_NEAREST)    
    elif scale_mode == 'INTER_AREA':
        x = cv2.resize(x, (scale*w,scale*h), interpolation=cv2.INTER_AREA) 
    elif scale_mode == 'INTER_CUBIC':
        x = cv2.resize(x, (scale*w,scale*h), interpolation=cv2.INTER_CUBIC) 

    xo = torch.tensor(np.transpose(x,(2,0,1)),dtype=torch.float).cuda().unsqueeze(0)
            
    return xo

# ======================== Measurement ========================

def l0b(img1,img2):
    xo = torch.abs(img1-img2)
    d = torch.sum(xo,1)>0.0
    return d.sum().item()

# ================ l0_projection for HSJA adapted ==============
def project_l0(original_image, perturbed_images, k):
    '''
    1. Clone "https://github.com/Jianbo-Lab/HSJA"
    2. Replace projection step built for l_2 and l_inf in the original code
    '''
    
    x = np.abs(original_image[0] - perturbed_images[0])
    wi = original_image.shape[2]
    x2 = x**2
    x2 = np.sum(x2,axis=0)
    x2 = x2.reshape(1,-1)
    n_same_px = len(np.where(x2==0)[0])
    out_images = original_image.copy()
    
    if n_same_px+k<wi*wi:
        idxs = np.argsort(x2)[:,n_same_px :n_same_px +k]
        c1 = idxs //wi
        c2 = idxs - c1 * wi
        out_images[:,:,c1,c2] = perturbed_images[:,:,c1,c2]

    return out_images
