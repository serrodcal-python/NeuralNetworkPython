from src.neural_network_class import NeuralNetwork

import tensorflow as tf

(train_data, train_targets), (test_data, test_targets) = tf.keras.datasets.boston_housing.load_data()

print(f'Training data : {train_data.shape}')
print(f'Test data : {test_data.shape}')
print(f'Training sample : {train_data[0]}')
print(f'Training target sample : {train_targets[0]}')

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

nn_class = NeuralNetwork(13, 4, 1)

nn_class.train(train_data, train_targets)

y = nn_class.predict(test_data)

print(f'Predicts : {y[0]}')
