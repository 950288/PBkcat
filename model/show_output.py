import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('./model/output/Kcat-MAEs.csv')

# Extract the epochs and R2 values from the CSV data
epochs = data['epoch']
R2_train = data['R2_train']
R2_test = data['R2_test']
Lr = data['Lr']

# Plotting the R2_train and R2_test values
R2, Lr = plt.subplots()
plt.figure(figsize=(10, 6))
plt.plot(epochs, R2_train, label='R2_train', marker='o', linestyle='-', color='blue')
R2.plot(epochs, R2_test, label='R2_test', marker='o', linestyle='-', color='red')
R2.xlabel('Epochs')
R2.ylabel('R2 Value')
plt.title('R2_train and R2_test over Epochs')
plt.legend()
plt.grid(True)

# Lr.figure(figsize=(10, 6))
Lr.plot(epochs, Lr, label='Lr', marker='o', linestyle='-', color='yellow')
Lr.xlabel('Epochs')
Lr.ylabel('R2 Value')

# Set the Y-axis limits to 0 and 100
plt.ylim(-0.2, 1)

plt.show()
