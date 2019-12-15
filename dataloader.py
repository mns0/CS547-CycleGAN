from __future__ import print_function
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy 
import os



def get_data(**opts):
    location = opts['folder'] + '/' +  opts['dataset']
    mode = opts['mode']

    #define transforms
    transform = transforms.Compose([
        transforms.Resize(opts['res']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    transformed_dataset = CycleGanDataLoader(location,mode,transform)
    #test_plot(data)
    dataloader = DataLoader(transformed_dataset, batch_size=opts['batch_size'], shuffle=True, num_workers=1)
    return dataloader

def test_plot(data):
    sidx = random.randint(0,len(data)-9)
    fig, ax = plt.subplots(4,2)
    c = 0
    for i in range(sidx, sidx+4):
        ax[c][0].imshow(data[i]['imageA'].numpy().transpose(2,1,0))
        ax[c][1].imshow(data[i]['imageB'].numpy().transpose(2,1,0))
        c +=1
    plt.show()



class cycle_data(Dataset):

    def __init__(self,path):

        self.monet = []
        self.photo = []
        monet_path = path+'trainA/'
        photo_path = path+'trainB/'
        for monet_name in os.listdir(monet_path):
            self.monet.append(monet_path+monet_name)
        for photo_name in os.listdir(photo_path):
            self.photo.append(photo_path+photo_name)

    def __len__(self):

        assert len(self.monet) != len(self.photo), "Size Different!"

        return len(self.monet)

    def __getitem__(self, idx):

        x,y = plt.imread(self.monet[idx]),plt.imread(self.photo[idx])
        x = (x-np.min(x))/np.ptp(x)
        y = (y-np.min(y))/np.ptp(y)
        x = np.moveaxis(x,(0,1,2),(1,2,0))
        y = np.moveaxis(y,(0,1,2),(1,2,0))
        return x,y






class CycleGanDataLoader(Dataset):
    """Data loader for cyclegan """ 
    def __init__(self, root_dir, mode,transform=None):
        """
        Args: 
            set_a/b (string): Path to photoset A/B
            transform (transforms): Pytorch transfroms
        """
        dir_a = root_dir+"/"+mode+"A"
        dir_b = root_dir+"/"+mode+"B"

        self.a_set = glob.glob(dir_a+"/*")
        self.b_set = glob.glob(dir_b+"/*")

        self.a_size = len(self.a_set)
        self.b_size = len(self.b_set)
        self.transform = transform

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #randomize the index for different pairs
        path_A = self.a_set[idx % len(self.a_set)]
        #path_B = self.b_set[idx % len(self.b_set)]
        path_B = self.b_set[random.randint(0,self.b_size-1)]

        image_A = Image.open(path_A)
        image_B = Image.open(path_B)

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        #return {'imageA' : image_A, 'imageB' : image_B }
        return  image_A,  image_B

    def __len__(self):
        return max(self.a_size, self.b_size)



class Sample_from_Pool(object):
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.cur_elements = 0
        self.items = []

    def __call__(self, in_items):
        return_items = []
        for in_item in in_items:
            if self.cur_elements < self.max_elements:
                self.items.append(in_item)
                self.cur_elements = self.cur_elements + 1
                return_items.append(in_item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elements)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = in_item
                    return_items.append(tmp)
                else:
                    return_items.append(in_item)
        return return_items



class cycle_data(Dataset):

    def __init__(self,path):

        self.monet = []
        self.photo = []
        monet_path = path+'trainA/'
        photo_path = path+'trainB/'
        for monet_name in os.listdir(monet_path):
            self.monet.append(monet_path+monet_name)
        for photo_name in os.listdir(photo_path):
            self.photo.append(photo_path+photo_name)

    def __len__(self):

        assert len(self.monet) != len(self.photo), "Size Different!"

        return len(self.monet)

    def __getitem__(self, idx):

        x,y = plt.imread(self.monet[idx]),plt.imread(self.photo[idx])
        x = (x-np.min(x))/np.ptp(x)
        y = (y-np.min(y))/np.ptp(y)
        x = np.moveaxis(x,(0,1,2),(1,2,0))
        y = np.moveaxis(y,(0,1,2),(1,2,0))
        return x,y

