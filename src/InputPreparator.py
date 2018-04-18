import numpy as np
from collections import defaultdict
import re
from functools import reduce



class EmbeddingsPreparator:

    def get_unique_tokens(self, files):
        vocab_tokens = list()

        for file in files:

            input_file = open(file, "r", encoding="utf8")
            AllWords = list()  # create new list

            for line in input_file:
                line.rstrip()  # strip white space
                words = line.split()  # split lines of words and make list
                AllWords.extend(words)  # make the list from 4 lists to 1 list

        for word in AllWords:  # for each word in line.split()
            word = re.sub(r'[^\w\s]', '', word)
            word = word.lower().strip()
            if not word.isdigit():
                if word not in vocab_tokens:  # if a word isn't in line.split
                    vocab_tokens.append(word)  # append it.
        vocab_tokens.append('.')
        vocab_tokens.append('?')
        return vocab_tokens


    def load_embedding_from_disks(self, glove_filename, words_to_keep, with_indexes=True):
        """
        Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionaries
        `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct
        `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
        """
        if with_indexes:
            word_to_index_dict = dict()
            index_to_embedding_array = []
        else:
            word_to_embedding_dict = dict()

        with open(glove_filename, 'r') as glove_file:
            count = 1
            for (i, line) in enumerate(glove_file):

                split = line.split(' ')

                word = split[0]

                representation = split[1:]
                representation = np.array(
                    [float(val) for val in representation]
                )
                if (word in words_to_keep):
                    if with_indexes:
                        word_to_index_dict[word] = count
                        index_to_embedding_array.append(representation)
                        count = count + 1
                    else:
                        word_to_embedding_dict[word] = representation
                        count = count + 1

        _WORD_NOT_FOUND = [0.0] * len(representation)  # Empty representation for unknown words.
        if with_indexes:
            _LAST_INDEX = count + 1
            word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
            index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
            return word_to_index_dict, index_to_embedding_array
        else:
            word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
            return word_to_embedding_dict


class StoryParser:

    def tokenize(self, sent):
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

    def parse_stories(self, lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            # line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                substory = None
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, file, only_supporting=False):
        '''Given a file name, read the file, retrieve the stories,
        and then convert the sentences into a single story.
        '''
        f=open(file, "r", encoding="utf8")
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        # print(data)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data]
        return data

    def vectorize_stories(self, data, word_to_index):
        xs = []
        xqs = []
        ys = []
        for story, query, answer in data:
            x = []
            xq = []
            for w in story:
                w = w.lower().strip()
                x.append(word_to_index[w])
            for w in query:
                w = w.lower().strip()
                xq.append(word_to_index[w])

                # The Answer is one-hot encoded in our vocabulary matrix
            y = np.zeros(len(word_to_index) + 1)
            answer = answer.lower().strip()
            y[word_to_index[answer]] = 1
            xs.append(x)
            xqs.append(xq)
            ys.append(y)
            # Idea: instead of padding here with the lengths of the whole datasets, make padding batch dependent!
        return np.array(xs), np.array(xqs), np.array(ys)

    def get_final_dataset(self, contexts, questions, answers):
        data_zipped = zip(contexts, questions, answers)
        final_data = np.array(list(data_zipped))

        return final_data


