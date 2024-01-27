import torch

from raug.loader import get_data_loader
from raug.train import fit_model
from raug.eval import test_model

def main():
    print(torch.__version__)

if __name__ == "__main__":
    main()