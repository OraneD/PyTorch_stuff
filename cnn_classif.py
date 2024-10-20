import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import copy

def init_params(m, seed=0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data, generator=torch.manual_seed(seed))
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02, generator=torch.manual_seed(seed))
        m.bias.data.fill_(0.01)
    return
##############################################Data Processing###############################################################
data_dir = "../DNN_PyTorch/datasets/"
train_data = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=torchvision.transforms.ToTensor())
num_classes = len(train_data.classes)

train_data = Subset(train_data, torch.arange(500))
test_data = Subset(train_data, torch.arange(50))

n_train_examples = int(len(train_data)*0.8)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples], generator=torch.manual_seed(0))

batch_size = 8
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(0))
valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

image_batch_example, labels_batch_example = next(iter(train_dataloader))

# plt.figure(figsize = (10,6))
# for ib in range(batch_size):
#     plt.subplot(batch_size // 4, 4, ib+1)
#     plt.imshow(image_batch_example[ib, :].squeeze().detach(), cmap='gray_r')
#     plt.xticks([]), plt.yticks([])
#     plt.title('Image label = ' + str(labels_batch_example[ib].item()))
# plt.show()

####################################### CNN #####################################################################
cnn_layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding=2),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=2))
out1 = cnn_layer1(image_batch_example)
print(out1.shape)

cnn_layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, padding=2),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=2))
out2 = cnn_layer2(out1)
print(out2.shape)

out_vec = torch.flatten(out2, start_dim=1)#On laisse le batch size tranquille c'est juste height et width qu'on flatten
inp_linear = out_vec.shape[1]

############################## CNN instance #####################################################
class CNNClassif(nn.Module):
    def __init__(self, input_size_linear, num_channels1=16, num_channels2=32, num_classes=10):
        super().__init__()
        
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_channels1),
            nn.MaxPool2d(kernel_size=2))
            
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(num_channels1, num_channels2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_channels2),
            nn.MaxPool2d(kernel_size=2))
        
        self.lin_layer = nn.Linear(input_size_linear, num_classes)

        
    def forward(self, x):

        out = self.cnn_layer1(x)
        out = self.cnn_layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.lin_layer(out)
        return out 
        
num_channels1 = 16
num_channels2 = 32
num_classes = 10
model = CNNClassif(inp_linear, num_channels1, num_channels2, num_classes)

print('Total number of parameters: ', sum(p.numel() for p in model.parameters()))

model.apply(init_params)

################################# Eval and Training #####################################
num_epochs = 40
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001

from typing import Tuple, List, List

def eval_cnn_classifier(model, eval_dataloader):
    model.eval()
    correct_labels = 0
    total_labels = 0
    with torch.no_grad():
        for batch_images, labels in eval_dataloader:
            y = model(batch_images)
            _, label_predicted = torch.max(y.data, 1)
            total_labels += labels.size(0)
            correct_labels += (label_predicted == labels).sum().item()
    
    accuracy = 100 * correct_labels / total_labels
    return  accuracy

def train_cnn(model: nn.Module, 
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              num_epochs: int,
              learning_rate: float,
              loss_fn = nn.CrossEntropyLoss,
              verbose: bool = True):
    model_tr = copy.deepcopy(model)
    model_tr.train()

    optimizer = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)
    train_losses, val_accuracies = [], []
    val_acc_opt = 0

    for epoch in range(num_epochs):
        train_loss = 0
        for batch_index, (images, labels) in enumerate(train_dataloader):
            y = model_tr(images)
            loss = loss_fn(y, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # We get the average training loss a the end of each epoch
        train_loss = train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        val_acc = eval_cnn_classifier(model_tr, val_dataloader)
        val_accuracies.append(val_acc)

        if verbose :
            print(f"Epoch {epoch + 1}, Training loss : {train_loss:.4f}; Validation accuracy : {val_acc:.4f}")

        if val_acc > val_acc_opt:
         model_opt = copy.deepcopy(model_tr)
         val_acc_opt = val_acc

    return model_opt, train_losses, val_accuracies

model_tr, train_losses, val_accuracies = train_cnn(model, train_dataloader, valid_dataloader, num_epochs, learning_rate, loss_fn, verbose=True)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(torch.arange(num_epochs)+1, train_losses)
plt.title('Training loss')
plt.xlabel('Epochs')
plt.subplot(1, 2, 2)
plt.plot(torch.arange(num_epochs)+1, val_accuracies)
plt.title('Validation accuracy')
plt.xlabel('Epochs')
plt.show()

accuracy = eval_cnn_classifier(model_tr, test_dataloader)
print('Accuracy of the network on the test images: ', accuracy, '%')