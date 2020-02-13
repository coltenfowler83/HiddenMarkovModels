import json
from collections import defaultdict


class BigramHMM:
    def __init__(self, training_data):
        self.training_data = self.tokenize_file(training_data)
        self.tag_counts = self._build_tag_counts()
        self.number_of_tags = len(self.tag_counts) - 1
        # don't need to consider the start token for number of tags
        self.bigram_tag_counts = self._build_bigram_tag_counts()
        self.emission_counts = self._build_emission_counts()

    @staticmethod
    def tokenize_file(filename):
        with open(filename) as f:
            sequences = f.readlines()

        sequences = [json.loads(s) for s in sequences]
        # sequences will be a list (all tag sequences) of lists (individual sequences) of lists (word and tag)
        return sequences

    def _build_tag_counts(self):
        tag_counts = defaultdict(int)
        # add a start tag for each sentence
        tag_counts['<s>'] = len(self.training_data)
        for seq in self.training_data:
            for word, tag in seq:
                tag_counts[tag] += 1

        return tag_counts

    def _build_bigram_tag_counts(self):
        bigram_tag_counts = defaultdict(lambda: defaultdict(int))
        for seq in self.training_data:
            tags = list(zip(*seq))[1]
            # add values for <start>, tag1
            bigram_tag_counts['<s>'][tags[0]] += 1
            for tag1, tag2 in zip(tags, tags[1:]):
                bigram_tag_counts[tag1][tag2] += 1

        return bigram_tag_counts

    def _build_emission_counts(self):
        emission_counts = defaultdict(lambda: defaultdict(int))
        for seq in self.training_data:
            for word, tag in seq:
                emission_counts[tag][word] += 1

        return emission_counts

    def compute_transmission_mle(self, tag1, tag2):
        return self.bigram_tag_counts[tag1][tag2] / self.tag_counts[tag1]

    def compute_emission_mle(self, tag, word):
        return self.emission_counts[tag][word] / self.tag_counts[tag]

    def compute_viterbi_sequence(self, sentence):
        pass
