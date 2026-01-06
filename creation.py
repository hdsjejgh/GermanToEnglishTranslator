import random
from models import *
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import datasets
import torchtext
from datasets import Dataset

#to stop torchtext from complaining every time i run it
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator
import tqdm
import torch.multiprocessing as mp
mp.freeze_support()

pad_index=67

#returns an appropriately padded batch of sequences from given batch of english and german examples
#the preprocessing function for the loader
def collate(batch):
    #gets the english and german token ids for each example
    en = [ex["en_ids"] for ex in batch]
    de = [ex["de_ids"] for ex in batch]
    #pads them all to equal lengths
    en_pad = nn.utils.rnn.pad_sequence(en,padding_value=pad_index)
    de_pad = nn.utils.rnn.pad_sequence(de, padding_value=pad_index)
    return {
        "en_ids" : en_pad,
        "de_ids" : de_pad
    }


if __name__=="__main__":


    #small scale tokenizers for english and german
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')
    print("Tokenizers Loaded")

    #train,validation, and test sets of examples
    # train_data = datasets.load_dataset("wmt14","de-en", split="train",streaming=True)
    # train_data = train_data.take(50_000)
    # train_data = Dataset.from_list(list(train_data)).flatten()
    # valid_data = datasets.load_dataset("wmt14","de-en", split="validation").flatten()
    # test_data = datasets.load_dataset("wmt14","de-en", split="test").flatten()
    #
    # train_data = train_data.rename_columns({"translation.de":"de","translation.en":"en"})
    # valid_data = valid_data.rename_columns({"translation.de":"de","translation.en":"en"})
    # test_data = test_data.rename_columns({"translation.de":"de","translation.en":"en"})

    dataset = datasets.load_dataset("bentrevett/multi30k")
    print("Multi30k Dataset Loaded")

    # train,validation, and test sets of examples
    train_data = dataset["train"]
    valid_data = dataset["validation"]
    test_data = dataset["test"]


    print("Dataset Loaded")


    print(train_data.column_names)

    #given an example (and the other arguments) returns the tokenized example with a sos and eos token added
    def tokenize_example(example, en_nlp, de_nlp, max_length, sos_token, eos_token):
        #tokenizes and lowercases every example
        en_tokens = [token.text.lower() for token in en_nlp.tokenizer(example["en"])][:max_length]
        de_tokens = [token.text.lower() for token in de_nlp.tokenizer(example["de"])][:max_length]
        #adds sos and eos tokens to the beginning and end
        en_tokens = [sos_token] + en_tokens + [eos_token]
        de_tokens = [sos_token] + de_tokens + [eos_token]
        #creates new column in given datasets for this
        return {"en_tokens": en_tokens, "de_tokens": de_tokens}

    #the arguments for the above function
    fn_kwargs = {
        "en_nlp": spacy_en,
        "de_nlp": spacy_de,
        "max_length": 100,
        "sos_token": "<sos>",
        "eos_token": "<eos>",
    }

    #tokenizes the datasets
    train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

    #english vocabulary generated from training data.
    en_vocab = build_vocab_from_iterator(
        train_data["en_tokens"],
        #min_freq=2,
        specials=[
            "<unk>", #unknown token
            "<pad>", #padding token
            "<sos>", #start of sentence
            "<eos>", #end of sentence
        ],
        min_freq=2
    )

    #german vocabulary generated from training data. only includes words with multiple occurences
    de_vocab = build_vocab_from_iterator(
        train_data["de_tokens"],
        #min_freq=2,
        specials=[
            "<unk>", #unknown token
            "<pad>", #padding token
            "<sos>", #start of sentence
            "<eos>", #end of sentence
        ],
        min_freq=2
    )

    print(len(en_vocab))
    print(len(de_vocab))

    #indices for unknown tokens and padding tokens
    unk_index = en_vocab["<unk>"]
    pad_index = en_vocab["<pad>"]

    #standardizes the default (unknown token) index
    en_vocab.set_default_index(unk_index)
    de_vocab.set_default_index(unk_index)


    #generates list of vocab indices for each token in an example
    def numericalize_example(example):
        en_ids = en_vocab.lookup_indices(example["en_tokens"])
        de_ids = de_vocab.lookup_indices(example["de_tokens"])
        #makes new column in dataset for this
        return {"en_ids":en_ids,"de_ids":de_ids}

    #numericalizes all data
    train_data = train_data.map(numericalize_example)
    valid_data = valid_data.map(numericalize_example)
    test_data = test_data.map(numericalize_example)

    #converts all data to torch tensors
    train_data=train_data.with_format(type="torch", output_all_columns=True)
    valid_data=valid_data.with_format(type="torch", output_all_columns=True)
    test_data=test_data.with_format(type="torch", output_all_columns=True)

    print("Data Preprocessing Done")



    #returns a data loader for the given dataset
    def get_loader(data,batch_size,shuffle=False):
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

    BATCH_SIZE = 128
    #number of iterations per epoch in traing is roughly 32,000/BATCH_SIZE as the loader will go through all 30k-ish examples

    #creates loaders for all data sets
    #only shuffles the training dataset
    train_loader = get_loader(train_data,batch_size=BATCH_SIZE,shuffle=True)
    valid_loader = get_loader(valid_data,batch_size=BATCH_SIZE,shuffle=False)
    test_loader = get_loader(test_data,batch_size=BATCH_SIZE,shuffle=False)


    # one hot and prediction vector lengths
    input_dim = len(de_vocab)
    output_dim = len(en_vocab)

    # the embedding dimensions for the encoder and decoders
    enc_emb_dim = 256
    dec_emb_dim = 256

    # dimension of the hidden and cell vectors
    hidden_dim = 512

    # layers (rows) in the rnn
    layers = 2

    # dropout percentage during training for the encoder and decoder
    enc_dropout = .5
    dec_dropout = .5

    # device is the gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # to train or not to train
    TRAIN = False

    print("Model Created")


    # initializes the weights of the model to random ones
    def initialize(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)


    # trains the model for one epoch
    def train(model, loader, optimizer, criterion, clip, tf_ratio, device):
        # puts the model in train mode
        model.train()
        # total loss over all batches
        epoch_loss = 0

        for i, batch in enumerate(loader):
            print(i)
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)

            # clears gradient in optimizer
            optimizer.zero_grad()

            # calculates predictions based on source
            output = model(src, trg, tf_ratio)

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # calculates loss and gradients
            loss = criterion(output, trg)
            loss.backward()
            # clips gradient in order to stop exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # updates parameters
            optimizer.step()
            # adds loss this batch to total loss
            epoch_loss += loss.item()
        # returns the average loss per batch
        return epoch_loss / len(loader)


    def evaluate(model, loader, criterion, device):
        # puts the model in evaluation mode
        model.eval()

        # will sum loss over each batch
        epoch_loss = 0

        # turns off the gradient calculation for speed
        with torch.no_grad():
            for i, batch in enumerate(loader):
                src = batch["de_ids"].to(device)
                trg = batch["en_ids"].to(device)

                # calculates predictions without teacher forcing
                output = model(src, trg, 1.0)
                trg = trg[1:]

                loss = criterion(
                    output.reshape(-1, output_dim),
                    trg.reshape(-1)
                )
                # adds loss to the total
                epoch_loss += loss.item()

        # returns the average loss per batch
        return epoch_loss / len(loader)


    epochs = 10
    # max gradient to prevent exploding gradient
    clip = 1.0
    # change to teacher force
    tf_ratio = 0.5
    # the lowest validation loss. tracked in order to save best model
    best_loss = float("inf")

    if TRAIN:
        # the encoder
        encoder = BidirectionalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=enc_emb_dim,
            layers=layers,
            dropout=enc_dropout
        )

        # the decoder
        decoder = Decoder(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            embedding_dim=dec_emb_dim,
            layers=layers,
            dropout=dec_dropout
        )

        # the actual model sent to appropriate device
        model = Seq2Seq(encoder, decoder, device).to(device)

        model.apply(initialize)

        print("Model Weights Initialized")

        # optimizer for the training and criterion for evaluation (and training)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

        print("Beginning Training:")

        # tqdm displays a progress bar
        # it works its just really slow to train
        for epoch in tqdm.tqdm(range(epochs)):
            # runs through the training set and gets the loss (~30k examples)
            train_loss = train(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                clip=clip,
                tf_ratio=tf_ratio,
                device=device
            )
            # runs through the validation set and gets the loss (~1k examples)
            valid_loss = evaluate(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device
            )

            # if this epoch got the best validation loss so far, save this version of the model
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model, "de_to_en_bdrnn.pt")

            # displays the loss info
            print(f"Train Loss: {train_loss}")
            print(f"Validation Loss: {valid_loss}")

        exit()

    # loading the model if not training
    model = torch.load("de_to_en_bdrnn.pt")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}")


    def translate_sentence(
            sentence,
            model,
            en_tokenizer,
            de_tokenizer,
            en_vocab,
            de_vocab,
            device,
            max_output_length,
    ):
        model.eval()
        with torch.no_grad():
            tokens = [token.text for token in de_tokenizer.tokenizer(sentence)]
            tokens = [token.lower() for token in tokens]
            tokens = ["<sos>"] + tokens + ["<eos>"]
            ids = de_vocab.lookup_indices(tokens)
            tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
            hidden, cell = model.encoder(tensor)
            inputs = en_vocab.lookup_indices(["<sos>"])
            for i in range(max_output_length):
                inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
                output, (hidden, cell) = model.decoder(inputs_tensor, hidden, cell)
                predicted_token = output.argmax(-1).item()
                inputs.append(predicted_token)
                if predicted_token == en_vocab["<eos>"]:
                    break
            tokens = en_vocab.lookup_tokens(inputs)
        return tokens

    # A man in an orange hat starring at something.
    print(translate_sentence('Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.', model, spacy_en, spacy_de, en_vocab, de_vocab, device, 25))