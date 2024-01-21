import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import Counter


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,subset = 'train', transform = None):
        
        self.classes = os.listdir("data/train") #  each directory is a class
        self.classes.remove('.DS_Store')
        self.pics   = [] # paths of each image
        self.labels = [] # labels from 0 => 6
        self.transform = transform
        self.train = (subset == "train") # whether the object is a training set or a test set
        self.map = dict(enumerate(self.classes)) # {0:'happy',1:'sad'.... }
    
        if subset == "train":
            path = "data/train"

            # looping over class names and their indices
            for i,c in enumerate(self.classes):
                
                classPath = os.path.join(path,c)
                picsInClassC = os.listdir(classPath) 
                
                for pic in picsInClassC:
                    if pic[-4:] == '.jpg':
                        #get pic path
                        picPath = os.path.join(classPath,pic)

                        # add path
                        self.pics.append(picPath)

                        # add label
                        self.labels.append(i)
                    
        elif subset == "val" : 
            path = "data/test"
            
            # looping over class names and their indices           
            for i,c in enumerate(self.classes):
                
                classPath = os.path.join(path,c)
                picsInClassC = os.listdir(classPath) 

                # first half of test set
                valLength = len(picsInClassC)//2
                
                for j in range(valLength):
                    # get pic path
                    pic = picsInClassC[j]
                    if pic[-4:] == '.jpg':
                        #get pic path
                        picPath = os.path.join(classPath,pic)

                        # add path
                        self.pics.append(picPath)

                        # add label
                        self.labels.append(i)
                    
        elif subset == "test" :
            path = "data/test"

            # looping over class names and their indices            
            for i,c in enumerate(self.classes):
                
                classPath = os.path.join(path,c)
                picsInClassC = os.listdir(classPath)

                # second half in the test set
                valLength = len(picsInClassC)//2
                testLength = len(picsInClassC) - valLength
                
                for j in range(testLength):
                    #get pic path
                    # start adding starting from the last val pic
                    pic = picsInClassC[valLength+j] 
                    if pic[-4:] == '.jpg':
                        #get pic path
                        picPath = os.path.join(classPath,pic)

                        # add path
                        self.pics.append(picPath)

                        # add label
                        self.labels.append(i)
            
    
    # used by Dataloader
    def __len__(self):
        return len(self.pics)

    #used by Dataloader
    def __getitem__(self,index):
        # open Image
        pic = Image.open(self.pics[index])
        
        # ADDED
        pic = transforms.Resize((48,48))(pic)
        # PIL.Image object to Tensor
        pic = transforms.ToTensor()(pic)

        if pic.shape[0] != 1:
            pic = Image.open(self.pics[index]).convert("L")
            pic = transforms.Resize((48, 48))(pic)
            pic = transforms.ToTensor()(pic)


        label  = self.labels[index]
        return pic,label
        
    # number of samples in each class
    def getClassCounts(self):
        return Counter(self.labels)

    
    # maps label integer into class name
    def mapLabel(self,label):
         return self.map[label]

    def plotImage(self,index):
        # pic shape = [1,48,48] (gray scale)
        pic = Image.open(self.pics[index]) 
        #print(self.pics[index])
        # resize just for better resolution in the visualization
        pic = transforms.Resize((1080, 1080))(pic)
        
        #crop in the y axis, x axis
        #pic = np.array(pic)[200:850,200:950] 
        
        
        print(f"Label : ",self.mapLabel(self.labels[index]))
        plt.imshow(pic, cmap='gray')