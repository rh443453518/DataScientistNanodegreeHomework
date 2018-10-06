from utils import *
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_classification_model(data_direct, use_gpu, model_hypers):
    # Assertion check if model parameter is correct
    assert model_hypers['architecture'] == 'inception_v3' or model_hypers['architecture'] == 'vgg16',  'Please enter a valid architecture: inception_v3 or vgg16'
    assert model_hypers['hidden_units'] > 0, 'Please enter a non-negative number of hidden units'
    assert model_hypers['learning_rate'] > 0, 'Please enter a non-negative number of learning rate'
    assert model_hypers['epochs'] > 0, 'Please enter a non-negative number of epochs'
    assert model_hypers['print_every'] > 0, 'Please enter a non-negative number of print_every'
    
    # Set up training device
    device = 'cuda' if use_gpu else 'cpu'
    
    # Set up model architecture
    dataloader, class_to_idx = load_image_data(data_direct, model_hypers['architecture'])
    model = pretrained_model_arch(model_hypers['architecture'], model_hypers['hidden_units'], model_hypers['dropout_prob'])
    criterion = nn.NLLLoss()
    if model_hypers['architecture'] == 'inception_v3':
        optimizer = optim.Adam(model.fc.parameters(), lr=model_hypers['learning_rate'])
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=model_hypers['learning_rate'])
    model.architecture = model_hypers['architecture']
    model = model.to(device)
    
    
    # Training process
    steps = 0
    for e in range(model_hypers['epochs']):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(dataloader['train']):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            if model_hypers['architecture'] == 'inception_v3':
                loss = criterion(outputs[0], labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Print model process
            if steps % model_hypers['print_every'] == 0:
                # Make sure network is in eval mode for inference
                # Disable dropout
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    _, train_accuracy = validation(model, dataloader['train'], criterion, device)
                    valid_loss, valid_accuracy = validation(model, dataloader['valid'], criterion, device)
                    
                print("Epoch: {}/{}... ".format(e+1, model_hypers['epochs']),
                      "Training Loss: {:.4f}".format(running_loss/model_hypers['print_every']),
                      "Valid Loss: {:.3f}.. ".format(valid_loss/len(dataloader['valid'])),
                      "Train Accuracy: {:.3f}".format(train_accuracy/len(dataloader['train'])),
                      "Valid Accuracy: {:.3f}".format(valid_accuracy/len(dataloader['valid'])))
                
                # Make sure training is back on
                running_loss = 0
                model.train()
                
    # Save dictionary of class_to_idx            
    model.class_to_idx = class_to_idx             
    return model
            

# ----------------------------------------------------------------------
# Predicting
# ----------------------------------------------------------------------
def predict_image_class(processed_image, model, topk, cat_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    tensor_image = torch.tensor(processed_image, dtype= torch.float).unsqueeze(0).to(device)
    prob_list = torch.exp(model.forward(tensor_image)).cpu().detach().numpy()[0]
    top_k_idx =  prob_list.argsort()[-topk:][::-1]
    probs =  prob_list[top_k_idx]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = list(idx_to_class[i] for i in top_k_idx)
    classes = list(str(x) for x in classes)
    predicted_k_flowers = [cat_name[x] for x in classes]
    return probs, predicted_k_flowers


# ----------------------------------------------------------------------
# Model utils
# ----------------------------------------------------------------------
def validation(model, dataloader, criterion, device):
    valid_loss = 0
    accuracy = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model.forward(inputs)
        valid_loss += criterion(outputs, labels).item()

        ps = torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy


def pretrained_model_arch(model_arch, hidden_units, dropout_prob):
    
    if model_arch == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(2048, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(dropout_prob)),
                        ('fc2', nn.Linear(hidden_units, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                ]))
        model.fc = classifier
    else:
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(25088, hidden_units)),
                        ('relu', nn.ReLU()),
                        ('dropout', nn.Dropout(dropout_prob)),
                        ('fc2', nn.Linear(hidden_units, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                ]))
        model.classifier = classifier
    return model