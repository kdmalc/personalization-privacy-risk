import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

mybatchsize = 32

lambdaD = 1e-3
lambdaE = 1e-4
lambdaF = 0.0
starting_update = 10
final_update = 18
cond_num = 1

update_ix = [0,  1200,  2402,  3604,  4806,  6008,  7210,  8412,  9614, 10816, 12018, 13220, 14422, 15624, 16826, 18028, 19230, 20432, 20769]

# Load Data
print("Loading Data")
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
testing_inputs = torch.tensor(input_data[:, test_split_idx:], dtype=torch.float)
testing_targets = torch.tensor(target_data[:, test_split_idx:], dtype=torch.float)
training_inputs = torch.tensor(input_data[:, :test_split_idx], dtype=torch.float)
training_targets = torch.tensor(target_data[:, :test_split_idx], dtype=torch.float)
print("Data loaded!")

# Convert data to DataLoader
print("Create custom datasets")
class CustomTimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        #return (self.data.shape[-1])
        return (self.data.shape[1]) # This only works for 2D inputs I think...

    def __getitem__(self, idx):
        # Assuming 'data' and 'labels' are lists of numpy arrays
        sample_data = self.data[:,idx]
        sample_labels = self.labels[:,idx]
        # You can apply any custom logic here based on the specific requirements of your task
        # Convert to PyTorch tensors
        sample_data = torch.Tensor(sample_data)
        sample_labels = torch.Tensor(sample_labels)
        return sample_data, sample_labels

train_dataset = CustomTimeSeriesDataset(training_inputs, training_targets)
trainloader = DataLoader(train_dataset, batch_size=mybatchsize, shuffle=False)
test_dataset = CustomTimeSeriesDataset(testing_inputs, testing_targets)
testloader = DataLoader(test_dataset, batch_size=mybatchsize, shuffle=False)
print("Datasets and dataloaders created!")

# Create LSTM model
print("Create LSTM Model")
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        #out = self.fc(out[:, -1, :])  # Take the output of the last time step
        out = self.fc(out)  # Predictions for all time steps
        return out

model = TimeSeriesLSTM(input_size, hidden_size, output_size, num_layers)
print("Model instantiated!")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_log = []
test_log = []

print_boolean = True
# Training the model
print("Train model")
for epoch in range(num_epochs):
    for inputs, targets in trainloader:
        
        total_elements = inputs.numel()
        new_first_dim = total_elements // (mybatchsize * input_size)
        # Calculate the remainder
        remainder = total_elements % (mybatchsize * input_size)
        # Optionally, you can pad or trim the tensor to make it evenly divisible
        if remainder != 0:
            # Pad or trim the tensor to make it evenly divisible
            inputs = inputs.view(-1)[:new_first_dim * mybatchsize * input_size]
        # Reshape the tensor
        input_reshaped = inputs.view(new_first_dim, mybatchsize, input_size)

        total_elements = targets.numel()
        new_first_dim = total_elements // (mybatchsize * input_size)
        # Calculate the remainder
        remainder = total_elements % (mybatchsize * input_size)
        # Optionally, you can pad or trim the tensor to make it evenly divisible
        if remainder != 0:
            # Pad or trim the tensor to make it evenly divisible
            targets = targets.view(-1)[:new_first_dim * mybatchsize * input_size]
        # Reshape the tensor
        targets_reshaped = targets.view(new_first_dim, mybatchsize, input_size)

        #######################################################
        # Reshape to (N, batch_size, input_size) with batch_size = 32
        #input_reshaped = inputs.view(-1, mybatchsize, input_size)
        #targets_reshaped = targets.transpose(0, 1)  # Didnt fix it... need to have an output size of 2...
        #targets_reshaped = targets.view(-1, mybatchsize, 2)
        #######################################################

        if epoch==0 and print_boolean==True:
            print_boolean = False
            print("Original size of inputs:", inputs.size())
            print("Reshaped size of inputs:", input_reshaped.size())

        ##########################################################################################

        optimizer.zero_grad()

        outputs = model(input_reshaped)
        #loss = criterion(outputs, targets)
        t1 = lambdaE*criterion(outputs, targets_reshaped)
        # Initialize a variable to accumulate the norms
        running_weights_norm = 0.0
        # Iterate through the parameters and calculate the norm
        for param in model.parameters():
            # Uhh make sure this is only norming the weights and not other params (bias or something idk)
            running_weights_norm += torch.norm(param)
        t2 = lambdaD*(running_weights_norm**2)
        t3 = lambdaF*(torch.linalg.matrix_norm(inputs)**2)
        loss = t1 + t2 + t3
        train_log.append(loss.item())
        loss.backward()
        optimizer.step()

    # Test the current model each epoch
    with torch.no_grad():
        for test_inputs, test_targets in testloader:
            model.eval()

            test_inputs_reshaped = test_inputs.view(-1, mybatchsize, input_size)
            test_targets = test_targets.transpose(0, 1)

            prediction = model(test_inputs_reshaped)
            t1 = lambdaE*criterion(prediction, test_targets)
            t2 = lambdaD*(torch.linalg.matrix_norm((model.weight))**2)
            t3 = lambdaF*(torch.linalg.matrix_norm(test_inputs)**2)
            test_loss = t1 + t2 + t3
            test_log.append(test_loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
print("Training complete!")

print("Plot training and testing logs!")
plt.plot(range(len(train_log)), train_log, label="Train")
plt.plot(range(len(test_log)), test_log, label="Test")
plt.legend()
plt.title("Loss Per Epoch")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.show()
