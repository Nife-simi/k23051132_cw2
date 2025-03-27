import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import random
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
from TorchSSL.models.flexmatch.flexmatch import FlexMatch
from TorchSSL.train_utils import AverageMeter
from TorchSSL.models.nets.wrn import WideResNet
import matplotlib.pyplot as plt
from pytorch_resnet_cifar10.resnet import resnet56

# Define a SimCLR model class for feature extraction using a pre-trained model
class SimCLR(nn.Module):
    def __init__(self,model_path="selflabel_cifar-10.pth"):
        """
        Initialise the SimCLR model by loading a pretrained ResNet18 as the backbone.
        
        Parameters:
            model_path (str): Path to the pre-trained model weights.
        """
        super(SimCLR,self).__init__()

        base_model = models.resnet18(pretrained=False)  
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  

        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.feature_extractor.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return x

model = SimCLR(model_path="selflabel_cifar-10.pth")
model.eval()

transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_typicality(embeddings,K=20):
    """
    Compute typicality scores for a set of embeddings based on nearest neighbor distances.
    
    Parameters:
        embeddings (np.array): The embeddings of the samples.
        K (int): Number of neighbors to consider when calculating typicality scores.
        
    Returns:
        np.array: The typicality scores for each sample.
    """
    neighbours = NearestNeighbors(n_neighbors=K, metric='euclidean').fit(embeddings)
    distances, _ = neighbours.kneighbors(embeddings)
    typicality_scores = 1 / np.mean(distances, axis=1)
    
    return typicality_scores

# Define Confidence Threshold (tune this based on your dataset)
confidence_threshold = 0.95

# Define the function to calculate the pseudo-labels
def generate_pseudo_labels(model, unlabeled_data_loader, device):
    model.model.eval()  # Set model to evaluation mode
    pseudo_labels = []
    
    # Iterate over the unlabeled data
    with torch.no_grad():
        for data in unlabeled_data_loader:
            inputs, _ = data  # Assuming unlabeled data is in the form (inputs, _)
            inputs = inputs.to(device)
            outputs = model.model(inputs)
            
            # Get the predicted class probabilities and labels
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_probs, pseudo_label = torch.max(probabilities, dim=1)
            
            # Assign pseudo-labels only for high-confidence predictions
            high_confidence_mask = max_probs > confidence_threshold
            pseudo_labels.append((pseudo_label[high_confidence_mask], max_probs[high_confidence_mask]))
    
    return pseudo_labels

def train_flexmatch(labeled_dataset,unlabeled_dataset,random,num_classes=10,num_epochs=20):
    # Set up data loaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Function to create a WideResNet architecture
    def net_builder(num_classes, depth=28, widen_factor=2, drop_rate=0.0, first_stride=1, is_remix=False):
        return WideResNet(
            first_stride=first_stride,
            num_classes=num_classes,
            depth=depth,
            widen_factor=widen_factor,
            drop_rate=drop_rate,
            is_remix=is_remix
    )

    # Initialise the FlexMatch model
    model = FlexMatch(
    net_builder=net_builder,
    num_classes=num_classes,
    ema_m=0.999,        # EMA momentum value
    T=0.5,              # Sharpening temperature
    p_cutoff=0.95,      # Confidence threshold for pseudo-labels
    lambda_u=1.0        # Weight for unsupervised loss
    )


    # Optimizer and loss function
    optimizer = optim.Adam(model.model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.model.train()  # Set the model to training mode
        
        labeled_loss_meter = AverageMeter()
        unlabeled_loss_meter = AverageMeter()

        correct = 0
        total = 0

        # Iterate over the data in batches
        for batch_idx, (labeled_data, unlabeled_data) in enumerate(zip(labeled_loader, unlabeled_loader)):
            # Process labeled data
            inputs_labeled, targets = labeled_data
            inputs_labeled, targets = inputs_labeled.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs_labeled = model.model(inputs_labeled)
            labeled_loss = criterion(outputs_labeled, targets)
            labeled_loss.backward()
            
            optimizer.step()

            labeled_loss_meter.update(labeled_loss.item(), inputs_labeled.size(0))

            # Compute training accuracy
            _, predicted = outputs_labeled.max(1)  # Get predicted labels
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Generate pseudo-labels for unlabeled data
            pseudo_labels = generate_pseudo_labels(model, unlabeled_loader, device)

            # Process unlabeled data (with pseudo-labels)
            for pseudo_label, pseudo_prob in pseudo_labels:
                if len(pseudo_label) == 0:  # If no high-confidence pseudo-labels, skip
                    continue

                inputs_unlabeled, _ = unlabeled_data
                inputs_unlabeled = inputs_unlabeled.to(device)

                pseudo_label = pseudo_label.to(device)
    
                # Filter inputs to match pseudo-label size
                if inputs_unlabeled.shape[0] != pseudo_label.shape[0]:
                    inputs_unlabeled = inputs_unlabeled[: pseudo_label.shape[0]]
                
                # Compute loss for unlabeled data with pseudo-labels
                outputs_unlabeled = model.model(inputs_unlabeled)
                unlabeled_loss = criterion(outputs_unlabeled, pseudo_label)

                unlabeled_loss_meter.update(unlabeled_loss.item(), inputs_unlabeled.size(0))
                
                # Add loss to total loss
                unlabeled_loss.backward()
                
                # Optimization step for both labeled and unlabeled losses
        optimizer.step()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
        print(f'Epoch {epoch}/{num_epochs} - '
            f'Labeled Loss: {labeled_loss_meter.avg:.4f} - '
            f'Unlabeled Loss: {unlabeled_loss_meter.avg:.4f}')
    
    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f"Mean Training Accuracy: {mean_accuracy:.2f}%")

    if random == False:
        with open("accuracies.txt", 'a') as f:
            f.write(f"Semi Supervised: {mean_accuracy:.2f}%\n")
    else:
        with open("accuracies_random.txt", 'a') as f:
            f.write(f"Semi Supervised: {mean_accuracy:.2f}%\n")

    # Save the model after training
    torch.save(model.model.state_dict(), 'flexmatch_model.pth')

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CIFAR-10 training and test datasets
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create a list of indices for the unlabeled dataset
    unlabeled_indices = list(range(len(dataset)))
    random.shuffle(unlabeled_indices)

    # Initialise Labeled and Unlabeled Sets
    L = []  # Initially empty
    U = unlabeled_indices  # Full dataset as unlabeled pool
    B = 10

    max_iterations = 5

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Create labeled and unlabeled subsets
        labeled_set = Subset(dataset, L) if L else None
        unlabeled_set = Subset(dataset, U)

        # Create DataLoader for labeled and unlabeled datasets
        train_loader = DataLoader(labeled_set, batch_size=64, shuffle=True) if L else None
        unlab_loader = DataLoader(unlabeled_set, batch_size=64, shuffle=False)

        # Set model to evaluation mode to freeze its weights
        model.eval()
        embeddings_list = []

        # Extract embeddings for the unlabeled samples
        with torch.no_grad():
            for images, _ in unlab_loader:
                images = images.to(device)
                embeddings = model(images)
                embeddings_list.append(embeddings.cpu().numpy())
        
        embeddings_list = np.concatenate(embeddings_list, axis=0)

        # Compute Typicality Scores using K-NN (K=20)
        typicality_scores = get_typicality(embeddings_list, K=20)

        # Select the B most typical samples based on their typicality scores
        selected_indices = np.argsort(typicality_scores)[-B:]  # Highest typicality (farthest from centroids)

        # Convert to original dataset indices
        selected_data_indices = [U[i] for i in selected_indices]

        # Move Selected Samples from U to L
        L.extend(selected_data_indices)
        U = [idx for idx in U if idx not in selected_data_indices]

    # Create final labeled and unlabeled subsets
    labeled_set = Subset(dataset, L)
    unlabeled_set = Subset(dataset,U)
    train_loader_AL = DataLoader(labeled_set, batch_size=64, shuffle=True, num_workers=2)

    random_indices_L = random.sample(range(len(dataset)),len(L))
    random_indices_U = random.sample(range(len(dataset)),len(U))

    random_L = Subset(dataset,random_indices_L)
    random_U = Subset(dataset,random_indices_U)

    train_flexmatch(labeled_set,unlabeled_set,False)
    train_flexmatch(random_L,random_U,True)