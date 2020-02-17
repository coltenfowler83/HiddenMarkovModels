from TrigramHMM import TrigramHMM

train_file = 'twt.train.json'
dev_file = 'twt.dev.json'
test_file = 'twt.test.json'

model = TrigramHMM(train_file)

#dev_accuracy = model.compute_accuracy(dev_file)
#print('Dev Error: ', dev_accuracy)

test_accuracy = model.compute_accuracy(test_file)
print('Test Error: ', test_accuracy)