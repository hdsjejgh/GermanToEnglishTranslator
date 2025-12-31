import torch
import torch.nn as nn
import torch.optim as optim
import random
import spacy
import datasets
import torchtext
#to stop torchtext from complaining every time i run it
torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator
import tqdm

