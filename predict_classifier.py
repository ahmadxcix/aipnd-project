import torch
from torchvision import datasets, transforms, models
from PIL import Image
import pandas as pd
import json




def load(path):
    """
    load the model form the path
    
    parameter:
        path (str)
    
    return:
        model from the path
    """
    
    checkpoint = torch.load(path)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    
    model.features = checkpoint['features']
    model.classifier = checkpoint['classifier']
    model.modules = checkpoint['modules']
    model.children = checkpoint['children']
    model.parameters = checkpoint['parameters']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['indices']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # DONE: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img_tensor = preprocess(img_pil)
    
    return img_tensor.numpy()


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to("cpu")
    model.eval()

    img = process_image(image_path)
    img_tensor = torch.from_numpy(img)
    img_tensor.unsqueeze_(0)
    with torch.no_grad():
        output = model.forward(img_tensor)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    
    return top_p, top_class



def classifier(in_arg):
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    img_path = in_arg.image_path
    model = load(in_arg.checkpoint)
    device = torch.device('cuda' if in_arg.gpu else 'cpu')
    model.to(device)

    topk = in_arg.topk
    
    ps, classes = predict(img_path, model, topk)
    ps, classes = ps.numpy(), classes.numpy()
    
    classes_names = []
    probilities = []
    dic = {}
    for cls in range(len(classes[0])):
        classes_names.append(cat_to_name[f"{classes[0][cls]}"])
        probilities.append(ps[0][cls])
    
    ser = {'name': pd.Series(data = classes_names), 'ps': pd.Series(data = probilities)}
    data = pd.DataFrame(ser)
    print(data)
