# GermanToEnglishTranslator

A German to English Translator which works using a Sequence to Sequence RNN.
Uses the multi30k or wmt14 datasets for training the models.

## Current Models

### de_to_en_bdrnn

#### Description: 
Uses a Bidirectional RNN as the encoder and a standard RNN to decode. Both multi-layer

#### Parameters:
Training set = Multi30k

Batch Size = 128

Encoder Embedding Dimension = 256

Decoder Embedding Dimension = 256

Hidden/Cell Dimension = 512

Layers = 2

Encoder Dropout = 50%

Decoder Dropout = 50%

Epochs = 10

Clip = 1.0

#### Other Info
De Vocab Size = 7853

En Vocab Size = 5893

Bleu Score on Testing Dataset = 0.16369

Loss on Testing Dataset (CCE) = 3.3620

