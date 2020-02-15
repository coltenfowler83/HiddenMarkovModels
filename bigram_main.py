# import statements
from BigramHMM import BigramHMM

# constants
train_file = 'twt.train.json'
dev_file = 'twt.dev.json'

model = BigramHMM(train_file)


def compute_accuracy(model, filename):
    error_count = 0
    observation_count = 0
    data = model.tokenize_file(filename)
    data = model.replace_word_classes(data)
    data = model.trim_low_freq(data)

    for seq in data:
        sentence = list(zip(*seq))[0]
        actual_tags = list(zip(*seq))[1]
        predicted_tags = model.generate_tag_sequence(sentence)
        print(predicted_tags)
        for actual, predicted in zip(actual_tags, predicted_tags):
            if actual != predicted:
                error_count += 1
            observation_count += 1

    return error_count / observation_count


# train_error = compute_accuracy(model, train_file)
dev_error = compute_accuracy(model, dev_file)

# print('Training Error: ', train_error)
print('Dev Error: ', dev_error)
