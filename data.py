import datasets
from datasets import Dataset
import torch

#contains various data related functions

#loads the multi30k dataset (the one with 30k training examples)
def load_multi30K(split,**kwargs):
    return datasets.load_dataset("bentrevett/multi30k",split=split)

#loads the wmt14 datset (the one with 4 million training examples)
#count is how many of the training samples to actually get (starting from the first)
def load_wmt14(split,count=None):
    #if no count provided, gets all samples
    if count is None:
        data = datasets.load_dataset("wmt14", "de-en", split=split).flatten()
    elif type(count)==int:
        #with streaming on, the whole dataset doesnt have to be loaded to take a portion of it
        data = datasets.load_dataset("wmt14", "de-en", split=split, streaming=True)
        data = data.take(count)
        #turns the streaming dataset back into a regular one
        data = Dataset.from_list(list(data)).flatten()
    else:
        raise TypeError("Count must be an integer value or None (indicating all rows)")

    #renames the columns to work with the other functions
    return data.rename_columns({"translation.de": "de", "translation.en": "en"})

#list of valid datasets and their functions
valid_datasets = ["multi30k","wmt14"]
dataset_paths = {
    "multi30k":load_multi30K,
    "wmt14":load_wmt14,
}
#loads given dataset, split, and sample count
def load_data(dataset,split,count=None):
    if dataset not in valid_datasets:
        raise ValueError(f"{dataset} is not a valid dataset")
    # Do not load 4 million samples.
    if count is None and dataset=="wmt14" and split=="train":
        raise Exception("You do not want to try and load the entire wmt14 training dataset. Pass a count of samples to retrieve")
    #if the dataset is small (e.g. multi30k), the count will be ignored and the whole thing will be loaded just for convenience sake
    return dataset_paths[dataset](split,count=count)

#given an example (and the other arguments) returns the spacy tokenized example with a sos and eos token added
def tokenize_example_spacy(example, en_nlp, de_nlp, max_length):
    #tokenizes and lowercases every example
    en_tokens = [token.text.lower() for token in en_nlp.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text.lower() for token in de_nlp.tokenizer(example["de"])][:max_length]

    #adds sos and eos tokens to the beginning and end
    en_tokens = ["<sos>"] + en_tokens + ["<eos>"]
    de_tokens = ["<sos>"] + de_tokens + ["<eos>"]

    #creates new column in given datasets for this
    return {
        "en_tokens": en_tokens,
        "de_tokens": de_tokens
    }

#given an example (and the other arguments) returns the bpe tokenized example with a sos and eos token added
# this was made for bpe tokenizers but i guess it could work for any sentence piece based tokenizer
def tokenize_example_bpe(example, max_length,sp=None,sp_en=None,sp_de=None):
    #ensures either an sp tokenizer xor an sp_en and sp_de tokenizer is passed in
    assert ((sp is not None) ^ (sp_en is not None and sp_de is not None)), "You must either pass an individual sp_en and sp_de tokenizer for each language, or a single sp tokenizer for both"

    # uses the given sentencepiece tokenizer for both languages
    if (sp is not None):
        en_tokens = sp.encode(example["en"], out_type=str)
        de_tokens = sp.encode(example["de"], out_type=str)
    # uses the individual tokenizers for each language given
    else:
        en_tokens = sp_en.encode(example["en"], out_type=str)
        de_tokens = sp_de.encode(example["de"], out_type=str)

    #adds the sentence start and end tokens
    en_tokens = ["<sos>"] + en_tokens + ["<eos>"]
    de_tokens = ["<sos>"] + de_tokens + ["<eos>"]

    #clips the sentences to the maximum token length
    en_tokens = en_tokens[:max_length]
    de_tokens = de_tokens[:max_length]

    return {
        "en_tokens": en_tokens,
        "de_tokens": de_tokens
    }

# generates list of vocab indices for each token in an example
def numericalize_example(example,en_vocab,de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    #makes new column in dataset for this
    return {"en_ids":en_ids,"de_ids":de_ids}

#returns a data loader for the given dataset
def get_loader(data,batch_size,collate,shuffle=False):
    return torch.utils.data.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        #the preprocessing function
        collate_fn=collate,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )
