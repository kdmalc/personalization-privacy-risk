import os

os.chdir("flcore")
print(os.path.abspath(os.curdir))
from flcore.pflniid_utils.data_utils import read_client_data
#from flcore.pflniid_utils import *
os.chdir("..")
print(os.path.abspath(os.curdir))