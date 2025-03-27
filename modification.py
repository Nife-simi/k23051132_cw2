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

# Modified SimCLR model using a pretrained ResNet-56 as the feature extractor
class SimCLRModified(nn.Module):
    """
    A modified version of the SimCLR model that uses a ResNet56 backbone for feature extraction.
    
    Parameters:
        model_path (str): Path to the pre-trained model for loading weights.
    """
    def __init__(self,model_path="resnet56-4bfd9763.th"):
        super(SimCLRModified,self).__init__()

        # Initialise the ResNet56 model as the feature extractor (without the final layer)
        base_model = resnet56()
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  

        # Load pre-trained model weights
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.feature_extractor.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        """
        Forward pass to extract features from the input images.
        
        Parameters:
            x (torch.Tensor): Input tensor containing images.
        
        Returns:
            torch.Tensor: Flattened feature tensor after extraction.
        """
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return x

model = SimCLRModified(model_path="resnet56-4bfd9763.th")
model.eval()

class LinearClassifierModified(nn.Module):
    """
    A modified fully connected classifier that takes in feature embeddings and outputs class predictions.
    
    Parameters:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The number of neurons in the hidden layer.
        output_dim (int): The number of output classes.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, input_dim=4096, hidden_dim=1024, output_dim=10, dropout=0.5):
        super(LinearClassifierModified, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_linear_classifier(train_loader,dimensions=512,classes=10,num_epochs=20):
    """
    Train the linear classifier on the given training data.
    
    Parameters:
        train_loader (DataLoader): DataLoader object that provides the training data.
        dimensions (int): The feature dimensionality of the input.
        classes (int): The number of output classes.
        num_epochs (int): The number of epochs to train for.
        
    Returns:
        LinearClassifierModified: The trained linear classifier.
    """
    classifier = LinearClassifierModified(input_dim=4096, hidden_dim=1024, output_dim=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=10)

    model.eval()

    accuracies = []

    classifier.train()
    for epoch in range(num_epochs):
        
        correct,total,running_loss = 0,0,0.0

        for images,labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                embeddings = model(images)
            
            classifier.train()
            outputs = classifier(embeddings)
            loss = criterion(outputs,labels)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f"Mean Training Accuracy: {mean_accuracy:.2f}%")

    with open("accuracies_modified.txt", 'w') as f:
        f.write(f"Fully Supervised with SSE: {mean_accuracy:.2f}%\n")

    return classifier

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

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


    unlabeled_indices = list(range(len(dataset)))
    random.shuffle(unlabeled_indices)

    # Initialise Labeled and Unlabeled Sets
    L = []  
    U = unlabeled_indices
    B = 10

    max_iterations = 5

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Create labeled and unlabeled subsets
        labeled_set = Subset(dataset, L) if L else None
        unlabeled_set = Subset(dataset, U)

        train_loader = DataLoader(labeled_set, batch_size=64, shuffle=True) if L else None
        unlab_loader = DataLoader(unlabeled_set, batch_size=64, shuffle=False)

        model.eval()
        embeddings_list = []
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

    labeled_set = Subset(dataset, L)
    unlabeled_set = Subset(dataset,U)
    train_loader_AL = DataLoader(labeled_set, batch_size=64, shuffle=True, num_workers=2)

    # Train the linear classifier
    linear_classifier = LinearClassifierModified(input_dim=4096, hidden_dim=1024, output_dim=10)
    linear_classifier.to(device)

    train_linear_classifier(train_loader_AL)