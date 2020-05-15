import argparse

def train_input_args():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('data_directory', type=str, default='flowers/', 
                        help='path to data folder of images')
    
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', 
                        help='path to saved directory')
    parser.add_argument('--arch', type=str, default='vgg16',choices=['vgg16', 'alexnet'],
                        help='select a network model; vgg16 or alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='insert a learing rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='set numbers of units in the hidden layer')
    parser.add_argument('--epochs', type=int, default=20, help='set number of epochs to train the network')
    parser.add_argument('--gpu', type=str, default='cuda',  help='choose cuda or cpu')
    
    return parser.parse_args()