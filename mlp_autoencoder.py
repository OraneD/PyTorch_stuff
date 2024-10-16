import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
import matplotlib.pyplot as plt
import copy
import os

def init_params(m, seed=0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, generator=torch.manual_seed(seed))
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    return


data_dir = "DNN_PyTorch/datasets/"
data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])
train_data = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=data_transforms)
test_data = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=data_transforms)
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

image_batch_example = next(iter(train_dataloader))[0]


#############################ENCODER##################################
class Encoder(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU())
        self.second_layer = nn.Sequential(nn.Linear(128,64), nn.ReLU())
        self.third_layer = nn.Sequential(nn.Linear(64,32), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.third_layer(self.second_layer(self.input_layer(x)))

image_batch_example_vec = image_batch_example.reshape([batch_size, -1])
input_size = image_batch_example_vec.shape[-1]
encoder = Encoder(input_size=input_size)
z = encoder(image_batch_example_vec)
print("Size of `z` (latent representation): ", z.shape)
print("Size of `image_batch_example_vec`: ", image_batch_example_vec.shape)

#############################DECODER##################################
class Decoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, output_size), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)
    
decoder = Decoder(output_size=input_size)
y = decoder(z)
print("Size of `y`: ", y.shape)
assert y.shape[-1] == image_batch_example_vec.shape[-1]

##########################AUTOENCODER####################################
class AutoEncoder(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.encoder = Encoder(input_size=input_size)
        self.decoder = Decoder(output_size=input_size)
    
    def forward(self, x):
        y_encoded = self.encoder(x)
        y_decoded = self.decoder(y_encoded)
        return y_decoded
    
mlp_autoencoder = AutoEncoder(input_size = image_batch_example_vec.shape[-1])
mlp_autoencoder.apply(init_params)
print("Number of parameters :", sum(p.numel() for p in mlp_autoencoder.parameters()))

###########################TRAINING#######################################
from typing import Tuple, List, List

def eval_model(model: nn.Module,
               eval_dataloader : DataLoader,
               eval_fn=nn.MSELoss()
               )-> float:
    model.eval() #Eval mode

    with torch.no_grad():
        total_mse = 0
        for batch_images, _ in eval_dataloader:
            batch_images_vec = batch_images.reshape([batch_images.shape[0], -1])
            y = model(batch_images_vec)
            total_mse += eval_fn(y, batch_images_vec).item()

    return total_mse / len(eval_dataloader)

def train_mlp_autoencoder(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    loss_fn=nn.MSELoss(),
    verbose: bool = True
) -> Tuple[nn.Module, List, List]:

    model_tr = copy.deepcopy(model)
    model_tr.train()

    optimizer = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)

    train_losses, mse_vals = [], []
    best_mse, best_model = None, copy.deepcopy(model_tr)

    for epoch in range(num_epochs):
        train_loss = 0        
        
        for batch_images, _ in train_dataloader:
            batch_images_vec = batch_images.reshape([batch_images.shape[0], -1])
            y = model_tr(batch_images_vec)
            loss = loss_fn(y, batch_images_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_dataloader)
        train_losses.append(train_loss)

        mse = eval_model(model_tr, eval_dataloader)
        if best_mse is None:
            best_mse = mse
        if mse < best_mse:
            best_mse, best_model = mse, copy.deepcopy(model_tr)
        mse_vals.append(mse)

        if verbose:
            print(
                f"Epoch[{epoch}/{num_epochs}]:",
                "train_loss:{:.4f}, MSE validation dataset:{:.4f}".format(train_loss, mse)
            )

    return best_model, train_losses, mse_vals

mlp_autoencoder.apply(init_params)

mlp_autoencoder_tr, train_losses, mse_vals = train_mlp_autoencoder(
    mlp_autoencoder, train_dataloader, valid_dataloader,
    50, 0.001,
)

print("Result on the test dataset: ", eval_model(mlp_autoencoder_tr, test_dataloader))

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(train_losses, color="blue")
plt.xlabel("Epoch n°"), plt.ylabel("Train loss")
plt.subplot(1, 2, 2)
plt.plot(mse_vals, color="red")
plt.xlabel("Epoch n°"), plt.ylabel("MSE on Validation Dataset")
plt.tight_layout()
plt.show()

##############################Visualisation################################
test_batch = next(iter(test_dataloader))[0]
bsize = test_batch.shape[0]

# Vectorize and apply the model
test_batch_vec = test_batch.reshape(bsize, -1)
with torch.no_grad():
    test_batch_vec_pred = mlp_autoencoder_tr(test_batch_vec)

# Reshape the prediction as a black-and-white image (3D tensor)
test_batch_pred = test_batch_vec_pred.reshape(bsize, 1, 28, 28)

# Plot the original and predicted images
for ib in range(batch_size):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(test_batch[ib, :].squeeze(), cmap='gray_r')
    plt.xticks([]), plt.yticks([])
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(test_batch_pred[ib, :].squeeze(), cmap='gray_r')
    plt.xticks([]), plt.yticks([])
    plt.title('Predicted image')
    plt.show()


