# Week 3 imports
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import PIL
from PIL import Image
from vars.week3 import *

# Week 4 imports
import torch.nn as nn
from torch.optim import SGD
from torchsummary import summary
from vars.week4 import *
# Week 5 imports
from torch.optim.lr_scheduler import ExponentialLR

gamma = 0.5
NUM_CLASSES = 10

def get_simple_conv_net():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), # no need for nn.Flatten: Conv2d expects a 2d array
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)), # output is out_channels * image_width/kernel_size * image_width/kernel_size
        
        nn.Flatten(), # output of MaxPool2d is multidimensional, so it needs to be flattened
        nn.Linear(16 * 14 * 14, 128), # Linear layer like before
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES)
    )


# Because the model now expects a 3-d input (channels * width * height), we need to modify our training function:
def train_model_gpu_lr_conv(model, epochs, train_dl, optimiser, lr_scheduler):
    msg = ""
    for epoch in range(epochs):
        total_steps = len(train_dl)
        correct = 0
        total = 0

        model.train()
        for batch_num, (image_batch, label_batch) in enumerate(train_dl):
            batch_sz = len(image_batch)
            label_batch = label_batch.to(DEVICE)
            image_batch = image_batch.to(DEVICE).reshape(batch_sz, 1, 28, 28)  # 1 channel, 28 * 28 pixels
            output = model(image_batch)
            losses = nn.CrossEntropyLoss()(output, label_batch)
                        
            optimiser.zero_grad()
            losses.backward()
            optimiser.step()  
            
            preds = torch.argmax(output, dim=1)
            correct += int(torch.eq(preds, label_batch).sum())
            total += batch_sz
            minibatch_accuracy = 100 * correct / total

            #### Fancy printing stuff, you can ignore this! ######
            if (batch_num + 1) % 5 == 0:
                print(" " * len(msg), end='\r')
                msg = f'Train epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps}], Loss: {losses.item():.5f}, Acc: {minibatch_accuracy:.5f}, LR: {lr_scheduler.get_last_lr()[0]:.5f}'
                print (msg, end='\r' if epoch < epochs else "\n",flush=True)
            #### Fancy printing stuff, you can ignore this! ######
            
        lr_scheduler.step()
        
        

def train_model_gpu_lr_conv_valid(model, epochs, dataloaders, optimiser, lr_scheduler):
    msg = ""
    for epoch in range(epochs):        
        #######################TRAINING STEP###################################
        model.train()  # set model to training mode 
        train_dl = dataloaders['train'] # select train dataloader
        
        total_steps_train = len(train_dl)
        correct_train = 0
        total_train = 0
        
        for batch_num, (image_batch, label_batch) in enumerate(train_dl):
            batch_sz = len(image_batch)
            label_batch = label_batch.to(DEVICE)
            image_batch = image_batch.to(DEVICE).reshape(batch_sz, 1, 28, 28) 
            output = model(image_batch)
            losses = nn.CrossEntropyLoss()(output, label_batch)
                        
            optimiser.zero_grad()
            losses.backward()
            optimiser.step()
            
            preds_train = torch.argmax(output, dim=1)
            correct_train += int(torch.eq(preds_train, label_batch).sum())
            total_train += batch_sz
            minibatch_accuracy_train = 100 * correct_train / total_train
            
            #### Fancy printing stuff, you can ignore this! ######
            if (batch_num + 1) % 5 == 0:
                print(" " * len(msg), end='\r')
                msg = f'Train epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps_train}], Loss: {losses.item():.5f}, Acc: {minibatch_accuracy_train:.5f}, LR: {lr_scheduler.get_last_lr()[0]:.5f}'
                print (msg, end='\r' if epoch < epochs else "\n",flush=True)
            #### Fancy printing stuff, you can ignore this! ######
        lr_scheduler.step()
        ########################################################################
        print("") # Create newline between progress bars
        #######################VALIDATION STEP##################################
        model.eval()  # set model to evaluation mode. This is very important, we do not want to update model weights in eval mode
        val_dl = dataloaders['val'] # select val dataloader
        
        total_steps_val = len(val_dl)
        correct_val = 0
        total_val = 0
        
        for batch_num, (image_batch, label_batch) in enumerate(val_dl):
            batch_sz = len(image_batch)
            label_batch = label_batch.to(DEVICE)
            image_batch = image_batch.to(DEVICE).reshape(batch_sz, 1, 28, 28) 
            
            with torch.no_grad(): # no_grad disables gradient calculations, which are not needed when evaluating the model. This speeds up the calculations
                output = model(image_batch)
                losses = nn.CrossEntropyLoss()(output, label_batch)

                preds_val = torch.argmax(output, dim=1)
                correct_val += int(torch.eq(preds_val, label_batch).sum())
                total_val += batch_sz
                minibatch_accuracy_val = 100 * correct_val / total_val
                
                #### Fancy printing stuff, you can ignore this! ######
                if (batch_num + 1) % 5 == 0:
                    print(" " * len(msg), end='\r')
                    msg = f'Eval epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps_val}], Loss: {losses.item():.5f}, Acc: {minibatch_accuracy_val:.5f}'
                    print (msg, end='\r' if epoch < epochs else "\n",flush=True)
                #### Fancy printing stuff, you can ignore this! ######
        ########################################################################
        print("")  # Create newline between progress bars
     