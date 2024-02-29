import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

# Dictionary to keep track of incorrect predictions for analysis
test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

# Helper function to count the number of correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
    # Compares predicted labels with the true labels and sums up the number of correct predictions
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

# Training function
def train(model, device, train_loader, optimizer, epoch):
    # Set model to training mode
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    # Iterate over the training data
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to the appropriate device (CPU or GPU)
        data, target = data.to(device), target.to(device)
        # Zero the gradients carried over from previous steps
        optimizer.zero_grad()

        # Forward pass - compute the model output
        pred = model(data)

        # Compute loss
        loss = F.nll_loss(pred, target)
        train_loss += loss.item()

        # Backward pass - compute the gradient
        loss.backward()
        # Update the parameters
        optimizer.step()
        
        # Accumulate the correct predictions
        correct += GetCorrectPredCount(pred, target)
        # Count the total number of processed samples
        processed += len(data)

        # Update the progress bar with the current loss and accuracy
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    # Record the accuracy and loss for the epoch
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

# Testing function
def test(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()

    test_loss = 0
    correct = 0

    # Disable gradient calculation for efficiency and to prevent changes during testing
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data to the appropriate device
            data, target = data.to(device), target.to(device)

            # Forward pass - compute the model output
            output = model(data)
            # Accumulate the batch loss
            test_loss += F.nll_loss(output, target).item()

            # Accumulate the number of correct predictions
            correct += GetCorrectPredCount(output, target)

    # Calculate the average loss and total accuracy
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print the test set results
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    

# Function to plot the training and testing graphs for loss and accuracy
def plt_fig():
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    # Plot training loss
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    # Plot training accuracy
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    # Plot testing loss
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    # Plot testing accuracy
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    # Save the figure
    fig.savefig("model_performance.png")
