import torch.nn as nn

    
################################################################################################
# RNN

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
    
# This is a bit of a hard coded solution...
class DeepRNNModel(nn.Module):
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

# Hidden size as a list isn't supported yet...
class DynamicRNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[],
                 num_layers=1, rnn_type='RNN', batch_first=True):
        super(DynamicRNNModel, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        # Define the RNN layer(s)
        if num_layers == 1:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_sizes[0], batch_first=batch_first)
        else:
            rnn_layers = []
            rnn_layers.append(getattr(nn, rnn_type)(input_size, hidden_sizes[0], batch_first=batch_first))
            for i in range(1, num_layers):
                rnn_layers.append(getattr(nn, rnn_type)(hidden_sizes[i - 1], hidden_sizes[i], batch_first=batch_first))
            self.rnn = nn.Sequential(*rnn_layers)
        # Define the fully connected layer
        self.fc = nn.Linear(hidden_sizes[-1] if num_layers > 1 else hidden_sizes[0], output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)

        # Fully connected layer
        output = self.fc(rnn_out)
        return output
    
################################################################################################
# LSTM

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


# LSTM with dropout
class DropoutLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(DropoutLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, eval=False):
        # Eg use this via model(input, eval=...)
        # x must be of shape (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        if eval==False:
            lstm_out = self.dropout(lstm_out)
        # This only uses the output of the last time step for now...
        output = self.fc(lstm_out)
        return output


# Canonical LSTM model example
class CannonLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        # I think I actually want all of them right...
        return out

################################################################################################
# GRU

# Edited GRU to pass sequence length so that you get the full output...
class GRUModelEDITED(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, sequence_length, output_size)

    def forward(self, x):
        # Output contains all hidden states for each time step
        output, h_n = self.gru(x)

        # You can use the entire sequence for further processing
        return self.fc(output)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h_n = self.gru(x)
        output = self.fc(h_n[-1])
        return output
    
################################################################################################
# TRANSFORMER

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
    
################################################################################################
# FROM PFL-NONIID
# split an original model into a base and a head
# This isn't used/integrated yet...
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out