import tensorflow as tf
import random
import numpy as np

class SessionRunner():
    training_iters = 50000
            
    def __init__(self, optimizer, accuracy, cost, lstm, initilizer, writer):
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.cost = cost
        self.lstm = lstm
        self.initilizer = initilizer
        self.writer = writer
    
    def run_session(self, x, y, n_input, dictionary, reverse_dictionary, training_data):
        
        with tf.Session() as session:
            session.run(self.initilizer)
            step = 0
            offset = random.randint(0, n_input + 1)
            acc_total = 0
        
            self.writer.add_graph(session.graph)
        
            while step < self.training_iters:
                if offset > (len(training_data) - n_input - 1):
                    offset = random.randint(0, n_input+1)
        
                sym_in_keys = [ [dictionary[ str(training_data[i])]] 
                                 for i in range(offset, offset+n_input) ]
                sym_in_keys = np.reshape(np.array(sym_in_keys), [-1, n_input, 1])
        
                sym_out_onehot = np.zeros([len(dictionary)], dtype=float)
                sym_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
                sym_out_onehot = np.reshape(sym_out_onehot,[1,-1])
        
                _, acc, loss, onehot_pred = session.run([self.optimizer, self.accuracy, 
                   self.cost, self.lstm], feed_dict={x: sym_in_keys, y: sym_out_onehot})
                acc_total += acc
                
                if (step + 1) % 1000 == 0:
                    print("Iteration = " + str(step + 1) + ", Average Accuracy= " + 
                                           "{:.2f}%".format(100*acc_total/1000))
                    acc_total = 0
                step += 1
                offset += (n_input+1)
