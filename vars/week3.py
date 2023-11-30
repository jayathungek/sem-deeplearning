MODEL_DIR = "./saved_models"
DATASET_PREFIX = "./data/MNIST"
DATA_PATH = f"{DATASET_PREFIX}/raw/mnist_dataset.csv"


import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import PIL
from PIL import Image


def load_image_tensor(filepath, threshold):
    img = Image.open(filepath)
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    img_tensor = transform(img)
    img_tensor = torch.where(img_tensor<threshold, 0, img_tensor) # ask class how we can fix this?
    return img_tensor



def custom_collate_fn(batch):
    image_batch_tensor = torch.FloatTensor(len(batch), 28, 28) # We define a tensor of the same size as our image batch to store loaded images into
    image_tensors = []
    labels = []
    for item in batch:
        image_tensor = load_image_tensor(f"{DATASET_PREFIX}/{item.iloc[0]}", threshold=50) # load a single image
        image_tensors.append(image_tensor) # put image into a list 
        labels.append(item.iloc[1]) # put the same image's label into another list


    torch.cat(image_tensors, out=image_batch_tensor) # torch.cat simply concatenates a list of individual tensors (image_tensors) into a single Pytorch tensor (image_batch_tensor)
    label_batch_tensor = torch.LongTensor(labels) # use the label list to create a torch tensor of ints
    return (image_batch_tensor, label_batch_tensor)





def load_data(data_path, batch_sz=100, train_val_test_split=[0.3, 0.2, 0.5]):
    # This is a convenience funtion that returns dataset splits of train, val and test according to the fractions specified in the arguments
    assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
    train_dataset = MNISTDataset(data_path)  # Instantiating our previously defined dataset
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    train_val_split = list(
        map( # map applies a given function to each element of a list
            lambda frac: round(frac * len(train_dataset)), # anonymous function that multiplies the fraction by total length of dataset and rounds to the nearest integer
            train_val_test_split # the list to apply the function to
        )
    )
    
    # split dataset into train, val and test
    train_split, val_split, test_split = random_split(train_dataset, train_val_split)
    
    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
    n_cpus = mp.cpu_count() # returns number of CPU cores on this machine
    train_dl = DataLoader(train_split, 
                          batch_size=batch_sz, 
                          shuffle=True, 
                          collate_fn=custom_collate_fn,
                          num_workers=n_cpus)            
    val_dl = DataLoader(val_split, 
                        batch_size=batch_sz, 
                        shuffle=True, 
                        collate_fn=custom_collate_fn,
                        num_workers=n_cpus)
    test_dl = DataLoader(test_split,
                         batch_size=batch_sz,
                         shuffle=False,
                         collate_fn=custom_collate_fn,
                         num_workers=n_cpus)
    return train_dl, val_dl, test_dl


class MNISTDataset(Dataset):
    def __init__(self, filepath: str): 
        super().__init__()
        self.dataframe = pd.read_csv(filepath) # Load data from CSV filepath defined earlier into a Pandas dataframe
    
    def __len__(self):
        return len(self.dataframe) # Return size of our dataframe
        
    def __getitem__(self, i):
        return self.dataframe.iloc[i] # Return the `i`th item in our dataframe
    

    
    


# Visualisation helper functions. When working with image data, it can be helpful to define such functions to
# make sure that the data visually "looks right". It is also a pretty good indication that you probably got all
# the dataloading code correct!

def image_grid(batch, ncols=4):
    height, width = batch[0].shape
    nrows = len(batch)//ncols # calculate the number of rows based on the number of columns needed by the user
    
    img_grid = (batch.reshape(nrows, ncols, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols))
    
    return img_grid


def show_batch(batch, title="Image batch", cols=4):
    N = len(batch)
    if N > cols:
        assert N % cols == 0, "Number of cols must be a multiple of N"
    
    result = image_grid(batch, cols)
    fig = plt.figure(figsize=(5., 5.))
    plt.suptitle(f"{title} [{int(N/cols)}x{cols}]")
    plt.imshow(result)
    
    
