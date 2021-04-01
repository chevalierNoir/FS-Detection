import math
import numpy as np

class Decoder(object):
    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.char_to_int = dict([(c, i) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def greedy_decode(self, prob, digit=False):
        indexes = np.argmax(prob, axis=1)
        string = []
        prev_index = -1
        for i in range(len(indexes)):
            if indexes[i] == self.blank_index:
                prev_index = -1
                continue
            elif indexes[i] == prev_index:
                continue
            else:
                if digit is False:
                    string.append(self.int_to_char[indexes[i]])
                else:
                    string.append(indexes[i])
                prev_index = indexes[i]
        return string
