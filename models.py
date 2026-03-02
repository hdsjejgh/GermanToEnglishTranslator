import torch
import torch.nn as nn
import random
import torchtext
from torch.nn.functional import softmax
import math

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
        return o, h, c


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

    def forward(self, inp, hidden, cell,**kwargs):
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

    def forward(self, src, trg, tf_ratio=0.0,**kwargs):
        # tf_ratio is chance to use teacher forcing each time

        trg_length, batch_size = trg.shape
        trg_vocab_size = self.decoder.output_dim

        if tf_ratio!=1 or isinstance(self.decoder,AttentionDecoder):
            # zero vector which will store prediction vectors for each example
            outputs = torch.zeros(trg_length-1, batch_size, trg_vocab_size).to(self.device)
            # hidden and cells gotten from the encoder
            o, h, c = self.encoder(src)
            # input is sos tokens for the first column
            inp = trg[0, :]

            # first item in output (the sos token) is skipped (therefore just being 0), hence the 1
            # first item is ignored in evaluation to compensate
            for t in range(1, trg_length):
                # gets output, hidden, and cell from first column of decoder
                pred, (h, c) = self.decoder(
                    inp, h, c, enc_states=o
                )
                # sets outputs vector appropriately
                outputs[t-1] = pred
                # top = the chosen token index
                top = pred.argmax(1)
                # the next input is either the correct input or the prediction depending on if teacher forcing or not
                inp = trg[t] if random.random() < tf_ratio else top
            return outputs
        else:
            o, h, c = self.encoder(src)

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

        return out,h, c

# like the original decoder just using attention
class AttentionDecoder(nn.Module):
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
        self.rnn = nn.LSTM(self.hidden_dim *2 + self.embedding_dim, self.hidden_dim, layers, dropout=dropout)

        self.energy = nn.Linear(self.hidden_dim*3,1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hidden, cell, enc_states):
        # token input ->
        # unsqueezed into new shape ->
        # embedded into vector ->
        # dropout applied to individual dimensions ->
        # the embed, hidden, and cell vectors (from previous rnn) get passed through new rnn ->
        # output is unsqueezed to correct shape ->
        # output is put through a dense linear layer to predict
        inp = inp.unsqueeze(0)
        emb = self.dropout(self.embedding(inp))

        seq_length = enc_states.shape[0]
        h_top = hidden[-1].unsqueeze(0)
        h_reshaped = h_top.repeat(seq_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped,enc_states),dim=2)))
        attention = softmax(energy)

        attention = attention.permute(1,2,0)
        enc_states = enc_states.permute(1,0,2)

        context = torch.bmm(attention,enc_states).permute(1,0,2)

        rnn_input = torch.cat((context,emb),dim=2)

        o, (h, c) = self.rnn(rnn_input, (hidden, cell))
        o = o.squeeze(0)
        pred = self.fc(o)
        return pred, (h, c)


#transformer stuff

#positional encoding from the pytorch docs
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
#####

class CustomTransformer(nn.Module):
    def __init__(self,input_dim,output_dim,d_model,nhead,layers,dropout,device):
        super().__init__()

        self.d_model = d_model

        self.device = device

        self.pos = PositionalEncoding(d_model, dropout)

        self.enc_embedding = nn.Embedding(input_dim, d_model)
        self.dec_embedding = nn.Embedding(output_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        #self.input_projection = nn.Linear(input_dim,d_model)
        self.output_projection = nn.Linear(d_model,output_dim)



        self.transformer= nn.Transformer(
                         d_model=d_model,
                         nhead=nhead,
                         num_encoder_layers=layers,
                         num_decoder_layers=layers,
                         dim_feedforward=2048,
                         dropout = dropout,
                         batch_first = False,
                         )

    def forward(self,src,target,pad_index=1,*args,**kwargs):

        emb = self.dropout(
            self.pos(
                self.enc_embedding(src) * math.sqrt(self.d_model)
            )
        )

        trg_emb = self.dropout(
            self.pos(
                self.dec_embedding(target[:-1]) * math.sqrt(self.d_model)
            )
        )

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target.size(0)-1,device=self.device)

        src_key_padding_mask = (src == pad_index).transpose(0, 1)
        tgt_key_padding_mask = (target[:-1] == pad_index).transpose(0, 1)

        y = self.transformer(emb,trg_emb,tgt_is_causal=True,src_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask,tgt_mask=tgt_mask)
        projected_y = self.output_projection(y)
        return projected_y