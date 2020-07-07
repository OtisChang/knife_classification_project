import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.resnet import resnet50
from model.Model_VGG16 import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from utils import parse_args
import copy
import time
import os

##REPRODUCIBILITY
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parse_args()
CUDA_DEVICES = args.cuda_devices
DATASET_ROOT = args.root

# Initial learning rate
init_lr = args.learning_rate
#opech
num_epochs = args.epochs
#batch size
batch_size = args.batch_size


# Save model every 5 epochs
checkpoint_interval = 5
if not os.path.isdir('Checkpoint/'):
    os.mkdir('Checkpoint/')

def draw_plot(loss_record, acc_record):

    plt.plot(loss_record, color='red')
    plt.title('Loss curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()

    plt.plot(acc_record, color='blue')
    plt.title('accuracy curve')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['acc'], loc='upper left')
    plt.show()

# Setting learning rate operation
def adjust_lr(optimizer, epoch):
    
    # 1/10 learning rate every 5 epochs
    lr = init_lr * (0.1 ** (epoch // 5))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(p=0.2),
        #transforms.ColorJitter(contrast=1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    print("dataset root", DATASET_ROOT)
    train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
    print("train_set.numclasses: {}".format(train_set.num_classes))
    print("len of train_set", len(train_set))
    # If out of memory , adjusting the batch size smaller
    data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    
    #print(train_set.num_classes)
    #model = VGG16(num_classes=train_set.num_classes)
    #model = resnet50(pretrained=False, num_classes=train_set.num_classes)
    model = resnet50(pretrained=True)

    # transfer learning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_set.num_classes)

    model = model.cuda(CUDA_DEVICES)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training epochs
    criterion = nn.CrossEntropyLoss()
    # Optimizer setting
    optimizer = torch.optim.SGD(params=model.parameters(), lr=init_lr, momentum=0.9)
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)


    # Log 
    with open('TrainingAccuracy.txt','w') as fAcc:
        print('Accuracy\n', file = fAcc)
    with open('TrainingLoss.txt','w') as fLoss:
        print('Loss\n', file = fLoss)

    #record loss & accuracy
    loss_record = list()
    acc_record = list()

    for epoch in range(num_epochs):
        localtime = time.asctime( time.localtime(time.time()) )
        print('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime))
        print('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,num_epochs,localtime)))

        training_loss = 0.0
        training_corrects = 0
        adjust_lr(optimizer, epoch)

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            optimizer.zero_grad()

            outputs = model(inputs)
            #print("outputs: {}, label: {}".format(outputs.size(), labels))


            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += float(loss.item() * inputs.size(0))
            training_corrects += torch.sum(preds == labels.data)
            print("preds : ", preds)
            print("labels : ", labels)
        
        training_loss = training_loss / len(train_set)
        training_acc = training_corrects.double() /len(train_set)
        print('Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss,training_acc))
        loss_record.append(training_loss)
        acc_record.append(training_acc)

        #save model each 10 epochs
        if epoch % 10 == 0:
            epoch_model = copy.deepcopy(model.state_dict())
            model.load_state_dict(epoch_model)
            model_name = './weights/epoch_models/model-{:.1f}epoch.pth'.format(epoch)
            torch.save(model, model_name)

        # Check best accuracy model ( but not the best on test )
        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())


        with open('TrainingAccuracy.txt','a') as fAcc:
            print('{:.4f} '.format(training_acc), file = fAcc)
        with open('TrainingLoss.txt','a') as fLoss:
            print('{:.4f} '.format(training_loss), file = fLoss)

        # Checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(model, 'Checkpoint/model-epoch-{:d}-train.pth'.format(epoch + 1))

    #draw the loss & acc curve
    draw_plot(loss_record, acc_record)

    # Save best training/valid accuracy model ( not the best on test )
    model.load_state_dict(best_model_params)
    best_model_name = './weights/model-{:.2f}-best_train_acc.pth'.format(best_acc)
    torch.save(model, best_model_name)
    print("Best model name : " + best_model_name)


if __name__ == '__main__':
    train()
