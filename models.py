import torch
import torch.nn as nn
import random
import torchtext
#to stop torchtext from complaining every time i run it
torchtext.disable_torchtext_deprecation_warning()

# Encoder class for the first half of the rnn
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, layers, dropout):
        super().__init__()
        # length of the one hot vectors (ie the number of words in the vocab)
        self.input_dim = input_dim
        # dimensions of the hidden and cell states in rnn
        self.hidden_dim = hidden_dim
        # dimension of embeddings
        self.embedding_dim = embedding_dim
        # number of layers in the rnn
        self.layers = layers
        # percent dropout to use in training
        self.dropout = dropout

        # necessary layers
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, layers, dropout=self.dropout)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, source):
        # sequence of token indices ->
        # embedding layer (one dim for each token) ->
        # dropout applied to individual dimensions ->
        # tokens input into an rnn
        emb = self.dropout(self.embedding(source))
        o, (h, c) = self.rnn(emb)
        return h, c


# Decoder class for the second half of the rnn
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embedding_dim, layers, dropout):
        super().__init__()
        # output dimension of the prediction vectors (ie, the english vocab length)
        self.output_dim = output_dim
        # hidden and cell vector dimension
        self.hidden_dim = hidden_dim
        # embedding vector dimension
        self.embedding_dim = embedding_dim
        # number of layers for the rnn
        self.layers = layers

        # necessary layers
        self.embedding = nn.Embedding(self.output_dim, self.embedding_dim)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, layers, dropout=dropout)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hidden, cell):
        # token input ->
        # unsqueezed into new shape ->
        # embedded into vector ->
        # dropout applied to individual dimensions ->
        # the embed, hidden, and cell vectors (from previous rnn) get passed through new rnn ->
        # output is unsqueezed to correct shape ->
        # output is put through a dense linear layer to predict
        inp = inp.unsqueeze(0)
        emb = self.dropout(self.embedding(inp))
        o, (h, c) = self.rnn(emb, (hidden, cell))
        o = o.squeeze(0)
        pred = self.fc(o)
        return pred, (h, c)


# Seq2Seq class to connect the encoder and decoder classes
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # first half of the model
        self.encoder = encoder
        # second half of the model
        self.decoder = decoder
        # the device the model is running on
        self.device = device

    def forward(self, src, trg, tf_ratio=0.0):
        # tf_ratio is chance to use teacher forcing each time

        trg_length, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        if tf_ratio!=1:
            # zero vector which will store prediction vectors for each example
            outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
            # hidden and cells gotten from the encoder
            h, c = self.encoder(src)
            # input is sos tokens for the first column
            inp = trg[0, :]

            # first item in output (the sos token) is skipped (therefore just being 0), hence the 1
            # first item is ignored in evaluation to compensate
            for t in range(1, trg_length):
                # gets output, hidden, and cell from first column of decoder
                o, (h, c) = self.decoder(inp, h, c)
                # sets outputs vector appropriately
                outputs[t] = o
                # top = the chosen token index
                top = o.argmax(1)
                # the next input is either the correct input or the prediction depending on if teacher forcing or not
                inp = trg[t] if random.random() < tf_ratio else top
            return outputs
        else:
            h, c = self.encoder(src)

            #[trg_len - 1, batch]
            #for teacher forcing
            trg_input = trg[:-1]

            emb = self.decoder.dropout(
                self.decoder.embedding(trg_input)
            )  # [trg_len-1, batch, emb_dim]

            outputs, _ = self.decoder.rnn(emb, (h, c))

            outputs = self.decoder.fc(outputs)
            # [trg_len-1, batch, vocab]

            return outputs

#like the original encoder just bidirectional
class BidirectionalEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, layers, dropout):
        super().__init__()
        # length of the one hot vectors (ie the number of words in the vocab)
        self.input_dim = input_dim
        # dimension of embeddings
        self.embedding_dim = embedding_dim
        #number of layers
        self.layers = layers
        #hidden and cell state dimensions
        self.hidden_dim = hidden_dim

        #necessary layers
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=False,
            bidirectional=True
        )
        #to convert the output of the lstm into the shape of the input of the decoder
        self.fc_h = nn.Linear(self.hidden_dim*2, hidden_dim)
        self.fc_c = nn.Linear(self.hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,source):
        emb = self.embedding(source)

        out, (h,c) = self.lstm(emb)

        #reorders so that all the values for one direction are together in their respective halves
        #doesnt have to be done but just in case i need them like this in the future
        h = torch.cat((h[0::2], h[1::2]), dim=2)
        c = torch.cat((c[0::2], c[1::2]), dim=2)

        h = torch.tanh(self.fc_h(h))
        c = torch.tanh(self.fc_c(c))

        return h, c