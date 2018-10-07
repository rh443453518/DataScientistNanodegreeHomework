import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image
import os
import errno


# ----------------------------------------------------------------------
# Data Processing
# ----------------------------------------------------------------------
def load_image_data(data_dir, model_arch):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    crop_size = 299 if model_arch=='inception_v3' else 224
    transformers = {'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(crop_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                    'valid': transforms.Compose([transforms.RandomResizedCrop(crop_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                    'test' : transforms.Compose([transforms.Resize(crop_size),
                                      transforms.CenterCrop(crop_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])}
    
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=transformers['train']),
                'valid': datasets.ImageFolder(valid_dir, transform=transformers['valid']),
                'test' : datasets.ImageFolder(test_dir,  transform=transformers['test'])}
    
    dataloader = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                  'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
                  'test' : torch.utils.data.DataLoader(image_datasets['test'],  batch_size=32)}
    
    return dataloader, image_datasets['train'].class_to_idx


def process_image(image, model_architecture):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    crop_size = 299 if model_architecture=='inception_v3' else 224
    expand_size = 310 if model_architecture=='inception_v3' else 240
    
    # Process a PIL image for use in a PyTorch model
    w,h = image.size
    ratio = h/w if h>w else w/h
    s = int(expand_size*ratio)
    image.thumbnail((s, s))
    w,h = image.size
    
    left = int((w - crop_size)/2)
    upper = int((h - crop_size)/2)
    image = image.crop((left, upper, left + crop_size, upper + crop_size))
    
    # return image
    np_image = np.array(image)
    np_image = (np_image/255 - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    np_image = np_image.T
    return np_image


# ----------------------------------------------------------------------
# Save and load model
# ----------------------------------------------------------------------
def save_model(model, save_dir):
    assert model.architecture == 'inception_v3' or model.architecture == 'vgg16',  'Please enter a valid architecture: inception_v3 or vgg16'
    checkpoint = {
        'architecture': model.architecture,
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'output_size': 102
    }
    checkpoint['classifier'] = model.fc if model.architecture=='inception_v3' else model.classifier
    
    # Make sure the saving directory exist. Create one if not.
    save_path = save_dir + model.architecture + '_checkpoint.pth'
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    torch.save(checkpoint, save_path)
    print('Successfully save model check point to:', save_path)


def load_model(checkpoint_file_path, compute_device):
    checkpoint = torch.load(checkpoint_file_path)
    assert checkpoint['architecture'] == 'inception_v3' or checkpoint['architecture'] == 'vgg16',  'Please load a valid architecture: inception_v3 or vgg16'
    
    if checkpoint['architecture'] == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        model.fc = checkpoint['classifier']
    else:
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.architecture = checkpoint['architecture']
    model = model.to(compute_device)
    return model
    
# ----------------------------------------------------------------------
# Get arguments from commind line prompt
# ----------------------------------------------------------------------
def get_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, default=None, 
                    help="data directory")
    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='save checkpoints to directory')
    parser.add_argument('--arch', type=str, default='inception_v3',
                        help='model architecture, can support VGG16 and inception_v3, default inception_v3')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate, default 0.001')
    parser.add_argument('--hidden_units', type=int, default=500,
                        help='hidden units, default 500')
    parser.add_argument('--print_every', type=int, default=10,
                        help='print every iterations, default 10')    
    parser.add_argument('--dropout_prob', type=int, default=0.1,
                        help='print every iterations, default 0.1')   
    parser.add_argument('--epochs', type=int, default=6,
                        help='epochs, default 6')
    parser.add_argument('--gpu', action='store_true',
                        default= False, help='change to cuda gpu')
    return parser.parse_args()


def get_args_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_image', type=str, default=None,
                    help='path of image file to predict')
    parser.add_argument('checkpoint', type=str, default='checkpoints/inception_v3_checkpoint.pth',
                    help='path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='top k classes the image most likely belongs to')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='mapping from category to class names')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='change to cuda gpu')
    return parser.parse_args()