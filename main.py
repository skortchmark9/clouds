#! /usr/bin/env python
from model import build_and_train_model
from data_transforms import create_blocks_across_time



def main():
    create_blocks_across_time()


if __name__ == '__main__':
    main()
