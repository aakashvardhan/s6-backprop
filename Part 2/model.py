# Importing PyTorch Library and its modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Defining the CNN Model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(F.relu(self.batchnorm(self.conv(x))))
    
class MaxPoolingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPoolingBlock, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv1d(self.pool(x))
    
# Defining the CNN Model leveraging the ConvBlock and MaxPoolingBlock (less than 20k parameters)

class Net(nn.Module):
    def __init__(self, in_channels=1, n_channels=32):
        super(Net, self).__init__()
        # Conv Block 1
        # r_in: 1, n_in:28, j_in:1, s:1, p:1, r_out:3, n_out:28, j_out:1
        self.conv1 = ConvBlock(in_channels, n_channels // 8, dropout=0, kernel_size=3, padding=1)
        # r_in:3, n_in:28, j_in:1, s:1, p:1, r_out:5, n_out:28, j_out:1
        self.conv2 = ConvBlock(n_channels // 8, n_channels // 4, dropout=0, kernel_size=3, padding=1)
        # r_in:5, n_in:28, j_in:1, s:1, p:1, r_out:7, n_out:28, j_out:1
        self.conv3 = ConvBlock(n_channels // 4, n_channels // 2, dropout=0, kernel_size=3, padding=1)
        # r_in:7, n_in:28, j_in:1, s:1, p:1, r_out:9, n_out:28, j_out:1
        self.conv4 = ConvBlock(n_channels // 2, n_channels, dropout=0.1, kernel_size=3, padding=1)
        
        # Transition Block 1
        '''
        MaxPooling(2,2):
        r_in:9, n_in:28, j_in:1, s:2, p:0, r_out:9, n_out:15, j_out:2
        
        with 1x1 Convolution:
        r_in:9, n_in:14, j_in:2, s:1, p:1, r_out:9, n_out:16, j_out:2
        '''
        self.mp1 = MaxPoolingBlock(n_channels, n_channels // 2)
        
        # Output Block
        
        # r_in:9, n_in:16, j_in:2, s:1, p:1, r_out:13, n_out:16, j_out:2
        self.conv5 = nn.Conv2d(n_channels // 2, n_channels * 2, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.1)
        
        # r_in:13, n_in:16, j_in:2, s:1, p:1, r_out:13, n_out:18, j_out:2
        self.conv6 = nn.Conv2d(n_channels * 2, 10, kernel_size=1, padding=1)
        
        self.gap = nn.AvgPool2d(kernel_size=16)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp1(x)
        x = F.relu(self.conv5(x))
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    
def model_summary(model, input_size=(1, 28, 28)):
    test_model_sanity()
    from torchsummary import summary
    summary(model, input_size=input_size)
        
# Test Model Sanity
def test_model_sanity():
    from tqdm import tqdm
    from utils import train, train_losses
    # Load MNIST dataset
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    # Use a small subset for testing to speed up the process
    train_subset = Subset(mnist_train, range(100))
    # Check if CUDA is available on the system and set `use_cuda` accordingly
    use_cuda = torch.cuda.is_available()

    # Set the device to "cuda" if CUDA is available, otherwise fall back to using the CPU
    device = torch.device("cuda" if use_cuda else "cpu")
    # Set the seed
    torch.manual_seed(1)
    # Create model
    model = Net().to(device)
    
    # Using NLLLoss as the loss function'
    criterion = F.nll_loss
    
    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create data loader
    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    
    # Train the model on small subset
    for epoch in range(1, 2):
        # Print the current epoch number
        print(f'Epoch {epoch}')
        train(model, device, train_loader, optimizer, criterion, epoch)
        
    # Perform sanity check: the loss should be decreasing after the first epoch
    assert train_losses[0] > train_losses[-1], "Loss is not decreasing"
    
    print("Sanity check passed: Model is capable of overfitting to a small subset of the data.")
        
        