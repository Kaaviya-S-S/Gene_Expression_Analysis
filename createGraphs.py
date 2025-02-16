import matplotlib.pyplot as plt

# Dictionary to store model names and their accuracies
model_accuracies = {}

# Read from the text file
with open('./models/classification.txt', 'r') as file:
    for line in file:
        model, accuracy = line.split(':')
        model_accuracies[model.strip()] = float(accuracy.strip())

# Prepare data for plotting
models = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())

# Create the bar plot
plt.figure(figsize=(7, 4))
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])

# Add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracies Comparison')

plt.savefig('./plots/accuracies.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()