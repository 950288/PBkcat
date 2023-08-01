import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('./model/output/Kcat-MAEs.csv')

# Extract the epochs and R2 values from the CSV data
epochs = data['epoch']
R2_train = data['R2_train']
R2_test = data['R2_test']
Lr_values = data['Lr']

# Plotting R2_train and R2_test values
plt.figure(figsize=(10, 6))
plt.plot(epochs, R2_train, label='R2_train', marker='o', linestyle='-', color='blue')
plt.plot(epochs, R2_test, label='R2_test', marker='o', linestyle='-', color='red')
plt.xlabel('Epochs')
plt.ylabel('R2 Value')
plt.title('R2_train and R2_test over Epochs')
plt.legend()
plt.grid(True)

# Create a new figure for Lr values
plt.figure(figsize=(10, 6))
plt.plot(epochs, Lr_values, label='Lr', marker='o', linestyle='-', color='yellow')
plt.xlabel('Epochs')
plt.ylabel('Lr Value')
plt.title('Lr over Epochs')
plt.legend()
plt.grid(True)

# Set the Y-axis limits for R2 and Lr plots
plt.ylim(-0.2, 1)

plt.show()
