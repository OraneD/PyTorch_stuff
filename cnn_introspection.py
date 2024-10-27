import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import copy
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

data_dir = '../DNN_PyTorch/datasets/'
train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())
num_classes = len(train_data.classes)
len_dataset = 2000
train_data = Subset(train_data, torch.arange(len_dataset))
batch_size = 8
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)

class CNNClassif_bnorm(nn.Module):
    def __init__(self, num_channels1=16, num_channels2=32, num_classes=10):
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
        
        self.lin_layer = nn.Linear(7*7*num_channels2, num_classes)
    
    def forward(self, x):
        
        out = self.cnn_layer1(x)
        out = self.cnn_layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.lin_layer(out)
        
        return out
    


num_channels1 = 16
num_channels2 = 32
model = CNNClassif_bnorm(num_channels1, num_channels2, num_classes)
model.load_state_dict(torch.load('../DNN_PyTorch/notebooks/lab6/model_cnn_classif_introspection.pt'))
model.eval()

image_batch_example = next(iter(train_dataloader))[0]
image_example = image_batch_example[0].squeeze()
plt.figure()
plt.imshow(image_example, cmap='gray_r')
plt.xticks([]), plt.yticks([])
#plt.show()

with torch.no_grad():
    out_conv1 = model.cnn_layer1(image_batch_example)

features_maps1 = out_conv1[0]
print(out_conv1[0].shape)
plt.figure(figsize = (10, 6))
for ic in range(3):
    plt.subplot(1, 3, ic+1)
    plt.imshow(features_maps1[ic,:], cmap='gray_r')
    plt.xticks([]), plt.yticks([])
    plt.title('Feature map ' + str(ic+1))
plt.show()

with torch.no_grad():
    out_conv2 = model.cnn_layer2(out_conv1)
features_maps2 = out_conv2[0]
#print(out_conv2)
print(out_conv2[0].shape)
plt.figure(figsize = (10, 6))
for ic in range(3):
    plt.subplot(1, 3, ic+1)
    plt.imshow(features_maps2[ic,:], cmap='gray_r')
    plt.xticks([]), plt.yticks([])
    plt.title('Feature map ' + str(ic+1))
plt.show()

##############################################T-SNE#################################
all_images, all_labels = [], []

model.eval()
with torch.no_grad():
    for images, labels in train_dataloader:
        all_images.append(images)
        all_labels.append(labels)

all_images = torch.cat(all_images, dim=0)
all_labels = torch.cat(all_labels, dim=0)
print('Size of the tensor containing all input images:', all_images.shape)
print('Size of the tensor containing all labels:', all_labels.shape)

all_images = all_images.reshape(all_images.shape[0], -1)
print('Size of the tensor containing all vectorized input images:', all_images.shape)

all_images = all_images.numpy()
all_labels = all_labels.numpy()

images_TSNE = TSNE(n_components=2, init='pca').fit_transform(all_images)
print('Size TSNE embeddings (input images):', images_TSNE.shape)

def plot_tsne_embeddings(X, y, title):
    
    #y = y.astype(int)
    X = QuantileTransformer().fit_transform(X)
    
    plt.figure(figsize = (5,5))
    for i in range(X.shape[0]):        
        plt.text(X[i, 0],
                 X[i, 1],
                 str(y[i]),
                 color=plt.cm.Dark2(int(y[i])),
                 fontdict={"weight": "bold", "size": 9})
    plt.xticks([]), plt.yticks([])
    plt.title('t-SNE - ' + title, fontsize=16)
    plt.show()
    
    return

plot_tsne_embeddings(images_TSNE, all_labels, 'Input images')


all_feature_maps = []
for images, labels in train_dataloader:
    feature_maps = model.cnn_layer1(images)
    all_feature_maps.append(feature_maps)


all_featuremaps1 = torch.cat(all_feature_maps, dim=0)
print('Size of the tensor containing all feature maps 1:', all_featuremaps1.shape)

all_featuremaps1 = all_featuremaps1.reshape(all_featuremaps1.shape[0], -1)
print('Size of the tensor containing all vectorized feature maps 1:', all_featuremaps1.shape)

all_featuremaps1 = all_featuremaps1.detach().numpy()

all_featuremaps2 = []
model.eval()
with torch.no_grad():
    for images, labels in train_dataloader:
        featuremaps1 = model.cnn_layer1(images)
        featuremaps2 = model.cnn_layer2(featuremaps1)
        all_featuremaps2.append(featuremaps2)

all_featuremaps2 = torch.cat(all_featuremaps2, dim=0)
all_featuremaps2 = all_featuremaps2.reshape(all_featuremaps2.shape[0], -1)
all_featuremaps2 = all_featuremaps2.detach().numpy()

feature_maps2_TSNE = TSNE(n_components=2, init='pca').fit_transform(all_featuremaps2)
print('Size TSNE embeddings (feature maps 1):', feature_maps2_TSNE.shape)
plot_tsne_embeddings(feature_maps2_TSNE, all_labels, 'Feature maps (layer 2)')