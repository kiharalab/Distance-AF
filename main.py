import os
from utils.argpaser import argparser
from Train.train import train
if __name__ == "__main__":
    args = argparser()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    train(args)
