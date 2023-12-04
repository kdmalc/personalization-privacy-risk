#import numpy as np
#import torch
#from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
#import torch.optim as optim


# Define a simple RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # RNN input: (batch_size, seq_len, input_size)
        # Output: (batch_size, seq_len, hidden_size)
        rnn_out, _ = self.rnn(x)

        # Fully connected layer
        output = self.fc(rnn_out)
        return output
    

# Define a simple LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM input: (batch_size, seq_len, input_size)
        # Output: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        last_output = lstm_out#[:, , :]
        # Fully connected layer
        output = self.fc(last_output)
        return output
    
    # Not sure which func to use, I've seen both...
    #def forward(self, x):
    #    _, (h_n, _) = self.lstm(x)
    #    output = self.fc(h_n[-1])
    #    return output


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_n = self.gru(x)
        output = self.fc(h_n[-1])
        return output
    

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, num_heads=8):
        super(TransformerModel, self).__init__()

        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Transformer input: (seq_len, batch_size, input_size)
        transformer_out = self.transformer(x, x)
        # Take the last time step's output
        last_output = transformer_out[-1, :, :]
        # Fully connected layer
        output = self.fc(last_output)
        return output