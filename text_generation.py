import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

import tensorflow
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tqdm
import os


class generate_text:
    def __init__(self):
        with open('trump_tweets.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()

    def check_text(self):
        if self.text:
            return True

    def map_to_int(self, text):
        """
        Args:
            text:
               The corpus text
        Returns:
                X (3D matrix) that contains True / False at the index of each PREDICT char.
                y (2D matrix) that contains the label char
                char_indices dictionary with each unique char with an integer value
                indices_char inverse dictionary of char_indices
        """

        text = text.replace("\n", " ")  # We remove newlines chars for nicer display
        print("Corpus length:", len(text))

        chars = sorted(list(set(text)))
        print("Total chars:", len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        maxlen = 100
        step = 3
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print("Number of sequences:", len(sentences))

        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.float32)
        y = np.zeros((len(sentences), len(chars)), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        return x, y, char_indices, indices_char

    def create_model(self):
        """
        Returns:

        """
        model = Sequential()

        model.add(LSTM(len(chars) * 4, input_shape=(maxlen, len(chars)), return_sequences=True))
        model.add(LSTM(128))
        model.add(BatchNormalization())
        model.add(Activation('selu'))

        model.add(Dense(len(chars) * 4))
        model.add(Activation('selu'))

        model.add(Dense(len(chars) * 4))
        model.add(BatchNormalization())
        model.add(Activation('selu'))

        model.add(Dense(len(chars), activation='softmax'))

        optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)

        return model



if __name__ == '__main__':
    create_text = generate_text()
    print(create_text.check_text())
