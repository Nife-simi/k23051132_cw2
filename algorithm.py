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


class SimCLR(nn.Module):
    def __init__(self,model_path="selflabel_cifar-10.pth"):
        """
        Initialise the SimCLR model by loading a pretrained ResNet18 as the backbone.
        
        Parameters:
            model_path (str): Path to the pre-trained model weights.
        """
        super(SimCLR,self).__init__()
 
        # Create a ResNet18 model without the final classification layer
        base_model = models.resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  

        # Load pre-trained weights into the feature extractor
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.feature_extractor.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        """
        Forward pass through the network to get embeddings from input images.

        Parameters:
            x (Tensor): Batch of images.

        Returns:
            Tensor: Flattened feature embeddings.
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return x

# Instantiate the SimCLR model and set it to evaluation mode
model = SimCLR(model_path="selflabel_cifar-10.pth")
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to compute the typicality score of samples based on their embeddings
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

        # Compute Typicality Scores using K-NN
        typicality_scores = get_typicality(embeddings_list, K=20)

        # Select the B most typical samples based on their typicality scores
        selected_indices = np.argsort(typicality_scores)[-B:]  

        # Convert to original dataset indices
        selected_data_indices = [U[i] for i in selected_indices]

        # Move Selected Samples from U to L
        L.extend(selected_data_indices)
        U = [idx for idx in U if idx not in selected_data_indices]

    # Create final labeled and unlabeled subsets
    labeled_set = Subset(dataset, L)
    unlabeled_set = Subset(dataset,U)
    train_loader_AL = DataLoader(labeled_set, batch_size=64, shuffle=True, num_workers=2)