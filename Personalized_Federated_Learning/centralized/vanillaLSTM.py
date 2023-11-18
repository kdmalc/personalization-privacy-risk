import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Hyperparameters
input_size = 64  # Number of channels
hidden_size = 128  # Size of the hidden state in LSTM
output_size = 2 # x,y vel predictions
num_layers = 2  # Number of LSTM layers
learning_rate = 0.001
num_epochs = 10

lambdaD = 1e-3
lambdaE = 1e-4
lambdaF = 0.0
update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]
starting_update = 10
final_update = 18
cond_num = 1

# Load Data
input_data = None
target_data = None
data_path = r"C:\\Users\\kdmen\\Desktop\\Research\\Data\\Client_Specific_Files"
for i in range(14):
    datafile = "UserID" + str(i) + "_TrainData_8by20770by64.npy"
    full_data = np.load(data_path+"\\"+datafile)
    cond_data = full_data[cond_num-1, update_ix[starting_update]:update_ix[final_update], :]
    data = np.transpose(cond_data)
    if input_data is None:
        input_data = data
    else:
        input_data = np.vstack((input_data, data))

    labelfile = "UserID" + str(i) + "_Labels_8by20770by2.npy"
    full_data = np.load(data_path+"\\"+labelfile)
    cond_data = full_data[cond_num-1, update_ix[starting_update]:update_ix[final_update], :]
    data = np.transpose(cond_data)
    if target_data is None:
        target_data = data
    else:
        target_data = np.vstack((target_data, data))

test_split_idx = ceil(input_data.shape[1]*.8)
testing_inputs = torch.tensor(input_data[:, test_split_idx:])
testing_targets = torch.tensor(target_data[:, test_split_idx:])
training_inputs = torch.tensor(input_data[:, :test_split_idx])
training_targets = torch.tensor(target_data[:, :test_split_idx])
# Convert data to DataLoader
#################### THIS IS BROKEN CAUSE TENSORDATASET CANT ACCOMODATE DIFFERENT NUMBERS OF ROWS...
## Should shuffle be t/f for timeseries?
training_dataset = TensorDataset(training_inputs, training_targets)
trainloader = DataLoader(training_dataset, batch_size=32, shuffle=False) 
testing_dataset = TensorDataset(testing_inputs, testing_targets)
testloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)

# Create LSTM model
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

model = TimeSeriesLSTM(input_size, hidden_size, output_size, num_layers)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_log = []
test_log = []

# Training the model
for epoch in range(num_epochs):
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        #loss = criterion(outputs, targets)
        t1 = lambdaE*criterion(outputs, targets)
        t2 = lambdaD*(torch.linalg.matrix_norm((model.weight))**2)
        t3 = lambdaF*(torch.linalg.matrix_norm(inputs)**2)
        loss = t1 + t2 + t3
        train_log.append(loss.item())
        loss.backward()
        optimizer.step()

    # Test the current model each epoch
    with torch.no_grad():
        for test_inputs, test_targets in trainloader:
            model.eval()
            prediction = model(test_inputs)
            t1 = lambdaE*criterion(prediction, test_targets)
            t2 = lambdaD*(torch.linalg.matrix_norm((model.weight))**2)
            t3 = lambdaF*(torch.linalg.matrix_norm(test_inputs)**2)
            test_loss = t1 + t2 + t3
            test_log.append(test_loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

plt.plot(range(len(train_log)), train_log, label="Train")
plt.plot(range(len(test_log)), test_log, label="Test")
plt.legend()
plt.title("Loss Per Epoch")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.show()
