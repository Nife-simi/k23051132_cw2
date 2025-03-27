import matplotlib.pyplot as plt

# Data
plots = [19.4, 38.7, 41.2]
labels = ["Random", "TPC-RP", "Modified"]

# Create bar plot
plt.figure(figsize=(8, 6))
plt.bar(labels, plots, color=['blue', 'green', 'red'])

# Labels and title
plt.xlabel("Algorithm")
plt.ylabel("Mean Accuracy (%)")
plt.title("Evaluation of Modification on Fully Supervised with Self Supervised Embeddings")

# Display the plot
plt.show()
