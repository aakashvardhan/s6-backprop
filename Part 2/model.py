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
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.batchnorm(F.relu(self.conv(x))))
    
class MaxPoolingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.pool(self.conv1d(x))
    
# Defining the CNN Model leveraging the ConvBlock and MaxPoolingBlock (less than 20k parameters)

class Net(nn.Module):
    def __init__(self, in_channels=1, n_channels=32):
        super().__init__()
        # Conv Block 1
        # r_in: 1, n_in:28, j_in:1, s:1, p:1, r_out:3, n_out:28, j_out:1
        self.conv1 = ConvBlock(in_channels, n_channels // 2, dropout=0, kernel_size=3)
        # r_in:3, n_in:28, j_in:1, s:1, p:1, r_out:5, n_out:28, j_out:1
        self.conv2 = ConvBlock(n_channels // 2, n_channels // 2, dropout=0, kernel_size=3)
        # r_in:5, n_in:28, j_in:1, s:1, p:1, r_out:7, n_out:28, j_out:1
        self.conv3 = ConvBlock(n_channels // 2, n_channels // 2, dropout=0, kernel_size=3)
        # r_in:7, n_in:28, j_in:1, s:1, p:1, r_out:9, n_out:28, j_out:1
        self.conv4 = ConvBlock(n_channels // 2, n_channels, dropout=0, kernel_size=3)
        
        # Transition Block 1
        '''
        MaxPooling(2,2):
        r_in:9, n_in:28, j_in:1, s:2, p:0, r_out:9, n_out:15, j_out:2
        with 1x1 Convolution:
        r_in:9, n_in:14, j_in:2, s:1, p:1, r_out:9, n_out:16, j_out:2
        '''
        
        self.mp1 = MaxPoolingBlock(n_channels, n_channels // 2)
        
        # Conv Block 2
        self.conv5 = ConvBlock(n_channels // 2, n_channels // 2, dropout=0, kernel_size=3)
        self.conv6 = ConvBlock(n_channels // 2, n_channels // 2, dropout=0.1, kernel_size=3)
        self.conv7 = ConvBlock(n_channels // 2, n_channels, dropout=0.1, kernel_size=3)
        
        # Output Block

        self.gap = nn.AvgPool2d(kernel_size=4)
        self.fc1 = nn.Linear(n_channels, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    
    
def model_summary(model, input_size=(1, 28, 28)):
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
    # Set the seed
    torch.manual_seed(1)
    # Create model
    model = Net()
    loss_function = F.nll_loss
    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create data loader
    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    
    # Train the model on the small subset
    # Calculate initial loss
    model.eval()  # Set the model to evaluation mode for initial loss calculation
    with torch.no_grad():
        data, target = next(iter(train_loader))
        initial_loss = loss_function(model(data), target).item()
    
    # Train the model on the small subset
    model.train()  # Set the model back to train mode
    for epoch in range(1, 4):  # Running for 3 epochs just for testing
        print(f"Epoch {epoch}")
        pbar = tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
    
    # Perform a sanity check: the loss should decrease after training
    model.eval()  # Set the model to evaluation mode for final loss calculation
    with torch.no_grad():
        final_loss = loss_function(model(data), target,def test_model_sanity():
    from tqdm import tqdm
    from utils import train, train_losses
    # Load MNIST dataset
    mnist_train = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    # Use a small subset for testing to speed up the process
    train_subset = Subset(mnist_train, range(100))
    # Set the seed
    torch.manual_seed(1)
    # Create model
    model = Net()
    loss_function = F.nll_loss
    # Using SGD as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create data loader
    train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
    
    # Train the model on the small subset
    # Calculate initial loss
    model.eval()  # Set the model to evaluation mode for initial loss calculation
    with torch.no_grad():
        data, target = next(iter(train_loader))
        initial_loss = loss_function(model(data), target).item()
    
    # Train the model on the small subset
    model.train()  # Set the model back to train mode
    for epoch in range(1, 4):  # Running for 3 epochs just for testing
        print(f"Epoch {epoch}")
        pbar = tqdm.tqdm(train_loader)

        for batch_idx, (data, target) in enumerate(pbar):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
    
    # Perform a sanity check: the loss should decrease after training
    model.eval()  # Set the model to evaluation mode for final loss calculation
    with torch.no_grad():
        data, target = next(iter(train_loader))
        final_loss = loss_function(model(data), target).item()

    assert final_loss < initial_loss, "Sanity check failed: Loss did not decrease after training."
    
    print("Sanity check passed: Model is capable of overfitting to a small subset of the data.")
).item()

    assert final_loss < initial_loss, "Sanity check failed: Loss did not decrease after training."
    
    print("Sanity check passed: Model is capable of overfitting to a small subset of the data.")

if __name__ == '__main__':
    test_model_sanity()