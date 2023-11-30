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

epochs = 5
batch_sz = 32
learning_rate = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining models
# The Sequential class is very simple: it accepts a sequence of neural network Modules as 
# arguments and arranges them such that the output of one is automatically sent to the input of the next in line. This saves
# us a bit of time writing some code, but has some drawbacks, as we shall see shortly.
# The following is the simplest possible neural network. It consists only of an input layer, 1 hidden layer 
# and an output layer
def get_simple_linear_net():
    return nn.Sequential(
        nn.Flatten(),                # Input is a 2d array of pixel values, so we flatten it out to 1d
        nn.Linear(28*28, 128),       # Input layer connects each input node to each hidden node. MNIST images are 28*28 pixels, hidden size can be anything we want
        nn.ReLU(),                   # ReLU activation only lets a signal through if it is > 0
        nn.Linear(128, 10)  # Output connects each node in the hidden layer to 10 output classes - the number of digits we want to classify!
        
    )


# Now that we've defined a network, we can start training it! Let's define the simplest possible training loop, for 
# which we only require the model, number of training epochs, the dataloader and an optimisation function
def train_model_simple(model, epochs, train_dl, optimiser):
    msg = ""
    for epoch in range(epochs):
        total_steps = len(train_dl)
        correct = 0
        total = 0

        model.train()  # set model to training mode
        for batch_num, (image_batch, label_batch) in enumerate(train_dl):
            # Prepare data and label batches 
            batch_sz = len(image_batch)
#             image_batch = image_batch.reshape(batch_sz, 1, 28, 28)
            output = model(image_batch)
            losses = nn.CrossEntropyLoss()(output, label_batch)
            
            # Zero gradients and backpropagate the losses
            optimiser.zero_grad()
            losses.backward()
            optimiser.step()  # update model weights based on loss gradients

            # Update the total number of correct predictions and calculate accuracy
            preds = torch.argmax(output, dim=1)
            correct += int(torch.eq(preds, label_batch).sum())
            total += batch_sz
            minibatch_accuracy = 100 * correct / total

            #### Fancy printing stuff, you can ignore this! ######
            if (batch_num + 1) % 5 == 0:
                print(" " * len(msg), end='\r')
                msg = f'Train epoch[{epoch+1}/{epochs}], MiniBatch[{batch_num + 1}/{total_steps}], Loss: {losses.item():.5f}, Acc: {minibatch_accuracy:.5f}'
                print (msg, end='\r' if epoch < epochs else "\n",flush=True)
            #### Fancy printing stuff, you can ignore this! ######
            
            
            
            




def train_model_lr(model, epochs, train_dl, optimiser, lr_scheduler):
    msg = ""
    for epoch in range(epochs):
        total_steps = len(train_dl)
        correct = 0
        total = 0

        model.train()
        for batch_num, (image_batch, label_batch) in enumerate(train_dl):
            batch_sz = len(image_batch)
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
            
        lr_scheduler.step() # Call the LR scheduler every epoch so that it can update the learning rate used by the optimiser
        
        
        


# We also need to modify our training loop to transfer input tensors to the GPU device
def train_model_lr_gpu(model, epochs, train_dl, optimiser, lr_scheduler):
    msg = ""
    for epoch in range(epochs):
        total_steps = len(train_dl)
        correct = 0
        total = 0

        model.train()
        for batch_num, (image_batch, label_batch) in enumerate(train_dl):
            batch_sz = len(image_batch)
            
            # Transferring image and label tensors to GPU #
            image_batch = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            ###############################################
            
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
            
        lr_scheduler.step() # Call the LR scheduler every epoch so that it can update the learning rate used by the optimiser