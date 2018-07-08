import tensorflow as tf
from DataHandler import DataHandler
from RNNGenerator import RNNGenerator
from SessionRunner import SessionRunner

log_path = 'output/tensorflow/'
writer = tf.summary.FileWriter(log_path)

# Load and prepare data
data_handler = DataHandler()

training_data =  data_handler.read_data('Zulu.txt')

dictionary, reverse_dictionary = data_handler.build_datasets(training_data)

# TensorFlow Graph input
n_input = 3
n_units = 512

x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, len(dictionary)])

# RNN output weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_units, len(dictionary)]))
}
biases = {
    'out': tf.Variable(tf.random_normal([len(dictionary)]))
}

rnn_generator = RNNGenerator()
lstm = rnn_generator.create_LSTM(x, weights, biases, n_input, n_units)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lstm, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(lstm,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
initilizer = tf.global_variables_initializer()

session_runner = SessionRunner(optimizer, accuracy, cost, lstm, initilizer, writer)
session_runner.run_session(x, y, n_input, dictionary, reverse_dictionary, training_data)
