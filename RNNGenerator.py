import tensorflow as tf
from tensorflow.contrib import rnn

class RNNGenerator:
    def create_LSTM(self, inputs, weights, biases, seq_size, num_units):
        # Reshape input to [1, sequence_size] and split it into sequences
        inputs = tf.reshape(inputs, [-1, seq_size])
        inputs = tf.split(inputs, seq_size, 1)
    
        # LSTM with 2 layers
        rnn_model = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units),rnn.BasicLSTMCell(num_units)])
    
        # Generate prediction
        outputs, states = rnn.static_rnn(rnn_model, inputs, dtype=tf.float32)
    
        return tf.matmul(outputs[-1], weights['out']) + biases['out']
