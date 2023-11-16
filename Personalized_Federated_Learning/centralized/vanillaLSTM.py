import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Assuming your input data is a PyTorch tensor named 'input_data' of size (num_samples, num_channels, num_datapoints)
# And your target data is a tensor named 'target_data' of size (num_samples, num_datapoints)

# Hyperparameters
input_size = 64  # Number of channels
hidden_size = 128  # Size of the hidden state in LSTM
output_size = 1  # Predicting one value for each input instance
num_layers = 2  # Number of LSTM layers
learning_rate = 0.001
num_epochs = 10

lambdaD = 1e-3
lambdaE = 1e-4
lambdaF = 0.0


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

# Convert data to DataLoader
dataset = TensorDataset(input_data, target_data)
trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training the model
for epoch in range(num_epochs):
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        #loss = criterion(outputs, targets)
        t1 = lambdaE*criterion(outputs, targets)
        t2 = lambdaD*(torch.linalg.matrix_norm((model.weight))**2)
        t3 = lambdaF*(torch.linalg.matrix_norm(input_data)**2)
        loss = t1 + t2 + t3
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Prediction
# Assuming test_input is your input data of size (1, 64, N)
test_input = torch.randn(1, 64, 10)  # Example with N=10
with torch.no_grad():
    model.eval()
    prediction = model(test_input)
    print("Prediction:", prediction)
