import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image


vgg16 = models.vgg16(pretrained=True)
alexnet = models.alexnet(pretrained=True)

models = {'vgg16': vgg16, 'alexnet': alexnet}
num_inputs = {'vgg16': 25088, 'alexnet': 9216}

def classifier(in_arg):
    data_path = in_arg.data_directory
    model_name = in_arg.arch
    
    train_dir = data_path + 'train/'
    valid_dir = data_path + 'valid/'
    test_dir = data_path + 'test/'
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation (30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    data_transforms = {'train': train_transforms, 'valid': valid_transforms, 'test': test_transforms}

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    image_datasets = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}

    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle = True)
    dataloaders = {'train': trainloader, 'valid': validloader, 'test': testloader}


    

    model = models[model_name]
    init_inputs = num_inputs[model_name]
    hidden_units = in_arg.hidden_units

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(init_inputs, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102)
        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    epochs = in_arg.epochs
    steps = 0
    print_every = 32
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            steps += 1

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for v_inputs, v_labels in dataloaders['valid']:
                        v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                        v_logps = model.forward(v_inputs)
                        batch_loss = criterion(v_logps, v_labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(v_logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

    accuracy = 0
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad ():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)
            test_loss += loss.item()

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    accuracy = (accuracy/len(dataloaders['test'])) * 100
    print(f"the accuracy for the model is: {accuracy:.1f}%")
    
    model.to('cpu')
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'features': model.features,
        'classifier': model.classifier,
        'modules': model.modules,
        'children': model.children,
        'parameters': model.parameters,
        'state_dict': model.state_dict(),
        'indices': model.class_to_idx,
        'epochs': epochs,
        'op_stat_dict': optimizer.state_dict
    }

    torch.save(checkpoint, in_arg.save_dir)