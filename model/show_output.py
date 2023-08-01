import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('./model/output/Kcat-MAEs.csv')

# Extract the epochs and R2 values from the CSV data
epochs = data['epoch']
R2_train = data[' R2_train']
R2_test = data[' R2_test']

# Plotting the R2_train and R2_test values
plt.figure(figsize=(10, 6))
plt.plot(epochs, R2_train, label='R2_train', marker='o', linestyle='-', color='blue')
plt.plot(epochs, R2_test, label='R2_test', marker='o', linestyle='-', color='red')
plt.xlabel('Epochs')
plt.ylabel('R2 Value')
plt.title('R2_train and R2_test over Epochs')
plt.legend()
plt.grid(True)

# Set the Y-axis limits to 0 and 100
plt.ylim(0, 1)

plt.show()
