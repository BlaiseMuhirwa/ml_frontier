import os
import pandas as pd
import torch
from typing import Tuple, List
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import string


def read_nmt_dataset() -> pd.DataFrame:
    """
    Reads the input dataset for Neural Machine Translation and
    pre-processes it by adding a `[start]` and `[end]` token
    for each target word/sentence
    """
    base_path = os.getcwd()
    dataset_path = base_path + "/data/eng_-french.csv"

    dataframe = pd.read_csv(dataset_path)

    dataframe["source"] = dataframe["English words/sentences"]
    dataframe["target"] = dataframe["French words/sentences"]

    dataframe = dataframe.drop(
        ["English words/sentences", "French words/sentences"], axis=1
    )

    # dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    dataframe = dataframe.sample(frac=1, random_state=1000)
    return dataframe


def build_vocab(dataframe):
    english_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    french_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")

    def build_vocab():
        english_counter = Counter()
        french_counter = Counter()
        for index, row in dataframe.iterrows():
            english_counter.update(english_tokenizer(row["source"]))
            french_counter.update(french_tokenizer(row["target"]))

        english_vocab = torchtext.vocab.vocab(
            english_counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"]
        )
        french_vocab = torchtext.vocab.vocab(
            french_counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"]
        )

        return english_vocab, french_vocab

    english_vocab, french_vocab = build_vocab()

    return english_vocab, french_vocab, english_tokenizer, french_tokenizer


def split_dataset(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_size = len(dataset)
    training_set_size = int(dataset_size * 0.7)
    testing_set_size = int(dataset_size * 0.00001)
    validation_set_size = int(dataset_size * 0.2)

    training_set = dataset[:training_set_size]
    testing_set = dataset[training_set_size : training_set_size + testing_set_size]
    validation_set = dataset[training_set_size + testing_set_size :]

    return training_set, testing_set, validation_set


def process_dataset(
    dataset: pd.DataFrame,
    en_vocab: torchtext.vocab.Vocab,
    fr_vocab: torchtext.vocab.Vocab,
    en_tokenizer,
    fr_tokenizer,
    testing=False,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    data = []
    for _, row in dataset.iterrows():
        english_tensor = torch.tensor(
            [en_vocab[token] for token in en_tokenizer(row["source"])], dtype=torch.long
        )
        if testing:
            tokens = en_tokenizer(row["source"])
            ids = [en_vocab[token] for token in en_tokenizer(row["source"])]
            print(f"english_tokens = {tokens}")
            print(f"ids = {ids}")
        french_tensor = torch.tensor(
            [fr_vocab[token] for token in fr_tokenizer(row["target"])], dtype=torch.long
        )
        data.append((french_tensor, english_tensor))

    return data


def get_dataloader(
    dataset: List[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    fr_vocab: torchtext.vocab.Vocab,
) -> DataLoader:
    pass
    pad_index, bos_index, eos_index = (
        fr_vocab["<pad>"],
        fr_vocab["<bos>"],
        fr_vocab["<eos>"],
    )

    def generate_batch(dataset_batch):
        fr_batch, en_batch = [], []

        for fr_tensor, en_tensor in dataset_batch:
            fr_batch.append(
                torch.cat([torch.tensor(bos_index), fr_tensor, torch.tensor(eos_index)])
            )
            en_batch.append(
                torch.cat([torch.tensor(bos_index), en_tensor, torch.tensor(eos_index)])
            )

            fr_batch = pad_sequence(fr_batch, padding_value=pad_index)
            en_batch = pad_sequence(en_batch, padding_value=pad_index)

        return fr_batch, en_batch

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=generate_batch
    )

    return dataloader


if __name__ == "__main__":
    raw_dataset = read_nmt_dataset()
    en_vocab, fr_vocab, en_tokenizer, fr_tokenizer = build_vocab(dataframe=raw_dataset)

    training_data, testing_data, validation_data = split_dataset(dataset=raw_dataset)
    print(len(testing_data))

    training_data = process_dataset(
        dataset=training_data,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        en_tokenizer=en_tokenizer,
        fr_tokenizer=fr_tokenizer,
    )
    testing_data = process_dataset(
        dataset=testing_data,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        en_tokenizer=en_tokenizer,
        fr_tokenizer=fr_tokenizer,
        testing=True,
    )
    validation_data = process_dataset(
        dataset=validation_data,
        en_vocab=en_vocab,
        fr_vocab=fr_vocab,
        en_tokenizer=en_tokenizer,
        fr_tokenizer=fr_tokenizer,
    )
