from BigramHMM import BigramHMM

train_file = 'twt.train.json'
dev_file = 'twt.dev.json'
test_file = 'twt.test.json'

model = BigramHMM(train_file, k=0.01)

dev_accuracy = model.compute_accuracy(dev_file)
test_accuracy = model.compute_accuracy(test_file)

print('Dev Error: ', dev_accuracy)
print('Test Error: ', test_accuracy)