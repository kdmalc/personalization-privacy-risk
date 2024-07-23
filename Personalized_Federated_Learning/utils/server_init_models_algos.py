import torch

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverlocal import Local
from flcore.servers.serverapfl import APFL

from models.DNN_classes import *


def init_algo(args):
    if args.algorithm == "FedAvg":
        # FIX ARGS.HEAD --> LINEAR REGRESSION HAS NO FC
        #args.head = copy.deepcopy(args.model.fc)
        #args.model.fc = nn.Identity()
        #args.model = BaseHeadSplit(args.model, args.head)
        server = FedAvg(args)
    elif args.algorithm == "Local":
        server = Local(args)
    elif args.algorithm == "APFL":
        server = APFL(args)
    elif args.algorithm == "PerAvg":
        server = PerAvg(args)
    elif args.algorithm == "pFedMe":
        server = pFedMe(args)
    else:
        raise NotImplementedError
    return server


def init_model(args):
    # Do I need to return args here...

    # Generate args.model
    if args.model_str == "LinearRegression":
        fresh_model_obj = torch.nn.Linear(args.input_size, args.output_size, args.linear_model_bias)  #input_size, output_size, bias boolean
    elif args.model_str == "RNN":
        # Initialize the RNN model
        fresh_model_obj = RNNModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "LSTM":
        # Initialize the LSTM model
        fresh_model_obj.model = LSTMModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "GRU":
        fresh_model_obj = GRUModel(args.input_size, args.hidden_size, args.output_size)
    elif args.model_str == "Transformer":
        fresh_model_obj = TransformerModel(args.input_size, args.output_size)
    else:
        raise NotImplementedError
    return fresh_model_obj
