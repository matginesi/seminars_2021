import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import warnings
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional
import time
import sys

DATA_DIR = 'new_data'
BATCH_SIZE = 64  # 32
NUM_WORKERS = 12
RESIZE_PARAM = 256  # 32
EPOCHS = 50
MODEL_NAME = 'ResNet50'  # 'FaceMaskDetectorCNN'
LEARNING_RATE = 0.001  # 0.0005  # 0.001

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('PyTorch device in use: ', device)


class FaceMaskDetectorCNN(nn.Module):
    def __init__(self):
        super(FaceMaskDetectorCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x: Tensor):
        """ forward pass
        """
        out = functional.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = functional.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(RESIZE_PARAM),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize(RESIZE_PARAM),
    transforms.CenterCrop(RESIZE_PARAM),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_split_train_test(datadir, valid_size=.2):
    train_data = datasets.ImageFolder(
        datadir,
        transform=train_transforms,
        loader=custom_loader)
    test_data = datasets.ImageFolder(
        datadir,
        transform=test_transforms,
        loader=custom_loader)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=BATCH_SIZE)
    return trainloader, testloader


trainloader, testloader = load_split_train_test(DATA_DIR, .2)
classes = trainloader.dataset.classes
print(classes)

if MODEL_NAME == 'ResNet50':
    model = models.resnet50(pretrained=True)
    # model.to(device)
    # print(model)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 10),
        nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    # model.to(device)
elif MODEL_NAME == 'FaceMaskDetectorCNN':
    model = FaceMaskDetectorCNN()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # model.to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)

if __name__ == '__main__':
    epochs = EPOCHS
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    total_accuracy = []
    graph_training_loss, graph_valid_loss, graph_acc = [], [], []

    print(' :: Training phase...', end='\n')
    print(' Epochs: {}, test evaluation at every {} steps'.format(
        epochs, print_every),
        end='\n\n')
    sys.stdout.flush()

    since = time.time()
    for epoch in range(epochs):
        for inputs, labels in tqdm(trainloader):
            steps += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    total_accuracy.append(accuracy)

                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                """
                print(f"\nTest evaluation: Epoch {epoch + 1}/{epochs}\t"
                      f"Train loss: {running_loss / print_every:.3f}\t "
                      f"Test loss: {test_loss / len(testloader):.3f}\t "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                """
                running_loss = 0
                model.train()

        time_elapsed = time.time() - since
        print('\nEpoch: {}/{} in {:4.0f}:{:2.0f}  '.format(
            epoch, epochs, time_elapsed // 60, time_elapsed % 60))
        print('Training loss: {:.4f}\tTest loss: {:.4f}\tAccuracy: {:.4f}'.format(
            sum(test_losses) / len(test_losses),
            sum(train_losses) / len(train_losses),
            sum(total_accuracy) / len(total_accuracy)
        ))
        graph_valid_loss.append(sum(test_losses) / len(test_losses))
        graph_training_loss.append(sum(train_losses) / len(train_losses))
        graph_acc.append(sum(total_accuracy) / len(total_accuracy))
        sys.stdout.flush()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // (60 * 60),
        time_elapsed // 60,
        time_elapsed % 60))

    torch.save(model, MODEL_NAME + '.pth')

    plt.plot(graph_training_loss, label='Training loss')
    plt.plot(graph_valid_loss, label='Validation loss')
    plt.plot(graph_acc, label='Accuracy')
    plt.legend(frameon=False)
    plt.savefig(MODEL_NAME + '_metrics.png')
    plt.show()
