import numpy as np
import collections

class DataHandler:
    def read_data(self, fname):
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        content = [content[i].split() for i in range(len(content))]
        content = np.array(content)
        content = np.reshape(content, [-1, ])
        content = np.hstack(content)
        return content
    
    def build_datasets(self, words):
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary
