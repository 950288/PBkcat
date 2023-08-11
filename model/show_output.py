import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('./model/output/Kcat_810-MAEs.csv')

# Extract the epochs and R2 values from the CSV data
epochs = data['epoch']
R2_train = data['R2_train']
R2_test = data['R2_dev']
Lr_values = data['Lr']
RMSE_train = data['RMSE_train']
RMSE_dev = data['RMSE_dev']


# Create a figure and axis for the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting R2_train and R2_test values on the first Y-axis (ax1)
ax1.plot(epochs, R2_train, label='R2_train', marker='o', linestyle='-', color='blue')
ax1.plot(epochs, R2_test, label='R2_test', marker='o', linestyle='-', color='red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('R2 Value')
ax1.set_title('R2_train and R2_test over Epochs')
ax1.legend()

# Create a second Y-axis for Lr values (ax2)
ax2 = ax1.twinx()
ax2.plot(epochs, Lr_values, label='Lr', linestyle='-', color='green')
ax2.set_ylabel('Lr Value')

# Adjust the Y-axis limits for R2 values
ax1.set_ylim(-0.2, 1)

# Show the plot
plt.grid(True)
plt.show()

# Create anotner plot for RMSE and LR
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting RMSE_train and RMSE_dev values on the first Y-axis (ax1)
ax1.plot(epochs, RMSE_train, label='RMSE_train', marker='o', linestyle='-', color='blue')
ax1.plot(epochs, RMSE_dev, label='RMSE_dev', marker='o', linestyle='-', color='red')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('RMSE Value')
ax1.set_title('RMSE_train and RMSE_dev over Epochs')
ax1.legend()

# Create a second Y-axis for Lr values (ax2)
ax2 = ax1.twinx()
ax2.plot(epochs, Lr_values, label='Lr', linestyle='-', color='green')
ax2.set_ylabel('Lr Value')

# Adjust the Y-axis limits for RMSE values
# ax1.set_ylim(0, 0.5)

# Show the plot
plt.grid(True)
plt.show()


