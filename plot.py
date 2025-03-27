import matplotlib.pyplot as plt
import numpy as np

# Function to read accuracy values from a file
def read_accuracies(filename):
    accuracies = {}
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                method, value = parts
                accuracies[method] = float(value.strip("%"))  # Convert to float
    return accuracies

# Read both text files
accuracies = read_accuracies("accuracies.txt")
accuracies_random = read_accuracies("accuracies_random.txt")

# Ensure the order is consistent
labels = list(accuracies.keys())
values1 = [accuracies[label] for label in labels]
values2 = [accuracies_random[label] for label in labels]

# X-axis positions
x = np.arange(len(labels))  
width = 0.35  # Bar width

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, values1, width, label="TPC-RP")
bars2 = ax.bar(x + width/2, values2, width, label="Random")

# Labels & Title
ax.set_xlabel("Testing Framework",fontsize=20)
ax.set_ylabel("Mean Accuracy (%)",fontsize=20)
ax.set_title("Comparison of TPC-RP and Random",fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right",fontsize=15)
ax.legend()

# Display values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset
                    textcoords="offset points",
                    fontsize=20,
                    fontweight='bold',
                    ha='center', va='bottom')

# Show plot
plt.tight_layout()
plt.show()
