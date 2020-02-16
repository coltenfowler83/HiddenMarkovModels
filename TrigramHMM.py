import json
import re
from collections import defaultdict
from math import log2


class TrigramHMM:
    def __init__(self, training_data):
        self.unk_threshold = 2
        self.k = 0.01
        self.training_data = self.tokenize_file(training_data)
        self.training_data = self.replace_word_classes(self.training_data)
        self.vocab = self._build_vocab()
        self.training_data = self.trim_low_freq(self.training_data)
        self.tag_counts = self._build_tag_counts()
        self.tags = list(self.tag_counts.keys() - ['<s2>', '<s>'])
        self.number_of_tags = len(self.tags)
        self.bigram_tag_counts = self._build_bigram_tag_counts()
        self.trigram_tag_counts = self._build_trigram_tag_counts()
        self.emission_counts = self._build_emission_counts()

    @staticmethod
    def tokenize_file(filename):
        with open(filename) as f:
            sequences = f.readlines()

        sequences = [json.loads(s) for s in sequences]
        # sequences will be a list (all tag sequences) of lists (individual sequences) of lists (word and tag)
        return sequences

    @staticmethod
    def replace_word_classes(sequences):
        # create closed vocab by replacing #, @, and urls with tokens. set words to lower case
        url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        for i in range(len(sequences)):
            for j in range(len(sequences[i])):
                sequences[i][j][0] = sequences[i][j][0].lower()
                word = sequences[i][j][0]
                if word.startswith('@'):
                    sequences[i][j][0] = '<@>'
                if word.startswith('#'):
                    sequences[i][j][0] = '<#>'
                if url_pattern.match(word):
                    sequences[i][j][0] = '<url>'

        return sequences

    def _build_vocab(self):
        word_counts = defaultdict(int)
        vocab = set()

        for seq in self.training_data:
            for word, tag in seq:
                word_counts[word] += 1

        for i in range(len(self.training_data)):
            for j in range(len(self.training_data[i])):
                word = self.training_data[i][j][0]
                if word_counts[word] > self.unk_threshold:
                    vocab.add(word)

        return vocab

    def trim_low_freq(self, sequences):
        for i in range(len(sequences)):
            for j in range(len(sequences[i])):
                word = sequences[i][j][0]
                if word not in self.vocab:
                    sequences[i][j][0] = '<unk>'

        return sequences

    def _build_tag_counts(self):
        tag_counts = defaultdict(int)
        # add 2 start tags for each sentence
        tag_counts['<s>'] = len(self.training_data)
        tag_counts['<s2>'] = len(self.training_data)
        for seq in self.training_data:
            for word, tag in seq:
                tag_counts[tag] += 1

        return tag_counts

    def _build_bigram_tag_counts(self):
        bigram_tag_counts = defaultdict(lambda: defaultdict(int))
        bigram_tag_counts['<s2>']['<s>'] = len(self.training_data)
        for seq in self.training_data:
            tags = list(zip(*seq))[1]
            # add values for (<start>, tag1)
            bigram_tag_counts['<s>'][tags[0]] += 1
            for tag1, tag2 in zip(tags, tags[1:]):
                bigram_tag_counts[tag1][tag2] += 1

        return bigram_tag_counts

    def _build_trigram_tag_counts(self):
        trigram_tag_counts = defaultdict(lambda: defaultdict(int))
        for seq in self.training_data:
            tags = list(zip(*seq))[1]
            trigram_tag_counts[('<s2>', '<s>')][tags[0]] += 1
            for tag1, tag2, tag3 in zip(tags, tags[1:], tags[2:]):
                trigram_tag_counts[(tag1, tag2)][tag3] += 1

        return trigram_tag_counts

    def _build_emission_counts(self):
        emission_counts = defaultdict(lambda: defaultdict(int))
        for seq in self.training_data:
            for word, tag in seq:
                emission_counts[tag][word] += 1

        return emission_counts

    def compute_transmission_mle(self, tag1, tag2, tag3):
        return (self.trigram_tag_counts[(tag1, tag2)][tag3] + self.k) / (self.bigram_tag_counts[tag1][tag2] + self.k * self.number_of_tags)

    def compute_emission_mle(self, tag, word):
        return (self.emission_counts[tag][word] + self.k) / (
                    self.tag_counts[tag] + self.k * len(self.vocab))

    def generate_tag_sequence(self, sentence):
        # sentence should be a list of words

        # first value refs tag column, second value refs word row
        v_matrix = [[0] * len(sentence) for _ in range(self.number_of_tags)]
        bp_matrix = [[0] * len(sentence) for _ in range(self.number_of_tags)]

        for i, tag in enumerate(self.tags):
            trans_prob = log2(self.compute_transmission_mle('<s2>', '<s>', tag))
            emi_prob = log2(self.compute_emission_mle(tag, sentence[0]))
            v_matrix[i][0] = trans_prob + emi_prob
            bp_matrix[i][0] = -1

        # start filling it the matrix at observation 1 since we initialized 0 above
        for i, word in enumerate(sentence[1:], 1):
            for j, current_tag in enumerate(self.tags):
                current_maxprob = float('-inf')
                current_maxbp = -1
                emi_prob = log2(self.compute_emission_mle(current_tag, word))
                for k, prev_tag in enumerate(self.tags):
                    for m, prev_prev_tag in enumerate(self.tags):
                        prev_v = v_matrix[k][i - 1]
                        prev_prev_v = v_matrix[m][i-2]
                        trans_prob = log2(self.compute_transmission_mle(prev_prev_tag, prev_tag, current_tag))
                        current_v = prev_prev_v + prev_v + emi_prob + trans_prob
                        if current_v > current_maxprob:
                            current_maxprob = current_v
                            current_maxbp = k
                    v_matrix[j][i] = current_maxprob
                    bp_matrix[j][i] = current_maxbp

        # need a tag-index map to decode
        index_tag_map = defaultdict(str)
        for i, tag in enumerate(self.tags):
            index_tag_map[i] = tag

        # the best path will start at max v_matrix tag for the last sentence
        bestpathprob = float('-inf')
        start_pointer = -1
        next_pointer = -1
        tag_sequence = []
        for i, tag in enumerate(self.tags):
            if v_matrix[i][len(sentence) - 1] > bestpathprob:
                bestpathprob = v_matrix[i][len(sentence) - 1]
                start_pointer = i
                next_pointer = bp_matrix[i][len(sentence) - 1]

        tag_sequence.append(index_tag_map[start_pointer])
        # bestpathpointer gives the tag of the last word in the sentence, follow backpointers to get tag seq
        for i in range(len(sentence) - 2, -1, -1):
            tag_sequence.append(index_tag_map[next_pointer])
            next_pointer = bp_matrix[next_pointer][i]

        return tag_sequence[::-1]

    def compute_accuracy(self, filename):
        error_count = 0
        observation_count = 0
        counter = 1
        data = self.tokenize_file(filename)
        data = self.replace_word_classes(data)
        data = self.trim_low_freq(data)

        for seq in data:
            sentence = list(zip(*seq))[0]
            actual_tags = list(zip(*seq))[1]
            predicted_tags = self.generate_tag_sequence(sentence)
            for actual, predicted in zip(actual_tags, predicted_tags):
                if actual != predicted:
                    if actual != predicted:
                        error_count += 1
                observation_count += 1

            acc = (observation_count - error_count) / observation_count
            print(counter, acc)
            counter += 1
        return (observation_count - error_count) / observation_count
