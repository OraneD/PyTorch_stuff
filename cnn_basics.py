import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import os 


############################Load Image##################################
def myplot(x):
    plt.figure()
    plt.imshow(x.detach().squeeze().numpy(), cmap='gray')
    plt.show()
    return

image_t = torchvision.io.read_image('../DNN_PyTorch/notebooks/lab5/tdp.jpeg', mode=torchvision.io.ImageReadMode.RGB)
image_t = torchvision.transforms.functional.rgb_to_grayscale(image_t)

#Normalization
image_t = image_t.float()
image_t /= image_t.max()
print('Image shape:', image_t.shape)

image_t = image_t.unsqueeze(dim=0)
print(image_t.shape)
#myplot(image_t)


#####################Convolution and kernel#########################
# To create a convolution function in Pytorch, we specify:
# - the number of input channels (= the depth of the input image, usually 1 for black and white images, and 3 for RGB color images
# - the number of output channels (which is equal to the number of kernels)
# - the kernel size (the dimension(s) of the filter)
# - wether to use bias or not (True by default)

num_channels_in = 1
num_channels_out = 1
kernel_size = 3
my_conv = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=kernel_size, bias=False)

my_conv.weight = nn.Parameter(torch.ones_like(my_conv.weight))#Initialise the weight of the kernels, set to one here
print(f"Kernel : {my_conv.weight}")
output = my_conv(image_t)
print(f"Image shape after being passed through convolution : {output.shape}")
#myplot(output)

#################################Exercice##########################################
for size in [5, 10, 20]:
    conv_layer = nn.Conv2d(num_channels_in, num_channels_out,kernel_size=size, bias=False )
    conv_layer.weight=(nn.Parameter(torch.ones_like(conv_layer.weight)))
    output = conv_layer(image_t)
    print(f"Shape after convolution, kernel_size = {size} : {output.shape}")
    #myplot(output)
###################################################################################

###########################Get Edges of an image###################################
# For instance, we can define a filter to detect edges in an image
edge_filter = torch.tensor([[[[-0.5, 0., 0.5], [-1., 0., 1.], [-0.5, 0., 0.5]]]])

# Then, we define a convolution and set the weights (=the kernel parameters) to this filter
my_conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
my_conv.weight = nn.Parameter(edge_filter, requires_grad=False)

# And now we apply convolution to the input image and get the edges
print(image_t.shape)
output = my_conv(image_t)
#myplot(output)

#########################Padding###################################################
# Create and apply a convolution without or with padding
my_conv = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=3, bias=False)
output = my_conv(image_t)
my_conv_padd = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=3, padding=1, bias=False)
output_padd = my_conv_padd(image_t)

# Check the size
print('Input shape :', image_t.shape)
print('Output shape, no padding: ', output.shape)
print('Output shape, with padding: ', output_padd.shape)

###################################Exercice########################################
conv_padd_layer = nn.Conv2d(num_channels_in, num_channels_out, kernel_size=5, padding=2, bias=False)
out = conv_padd_layer(image_t)
print('Input shape :', image_t.shape)
print('Output shape, with padding: ', out.shape)

#################################POOLING#############################################
avg_pooling_layer = nn.AvgPool2d(kernel_size=10)
max_pooling_layer = nn.MaxPool2d(kernel_size = 10)

avg_out = avg_pooling_layer(image_t)
myplot(avg_out)

max_out = max_pooling_layer(image_t)
myplot(max_out)

