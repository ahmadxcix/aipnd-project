import argparse
import torch

def predict_input_args():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('image_path', type=str,
                        help='path to image file')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth',
                        help='checkpoint.pth file')
    parser.add_argument('--topk', type=int, default=3,
                        help='top k hight classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='json file of the categories names')
    
    if torch.cuda.is_available():
        parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                            help='use GPU if available')
    else:
        parser.add_argument('--gpu', dest='gpu', default=False, action='store_false',
                            help='use GPU if available')
    
    return parser.parse_args()