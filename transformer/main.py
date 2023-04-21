#!/usr/bin/env python 

from dataset import read_nmt_dataset, split_dataset
from attention import *


def main():
    training_data, testing_data, validation_data = split_dataset(
        dataset=read_nmt_dataset()
    )

    # print(training_data.head())
    # print(testing_data.head())
    # print(validation_data.head())
    print(type(training_data["source"]))


if __name__ == "__main__":
    main()
