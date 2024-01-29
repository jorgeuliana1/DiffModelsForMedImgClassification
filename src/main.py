import os
import sys

# Including the path to the models folder
sys.path.insert(0, os.environ["MY_MODELS_PATH"])

import torch


def main():
    print(torch.__version__)


if __name__ == "__main__":
    main()
