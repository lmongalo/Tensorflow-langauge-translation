import tensorflow as tf
import numpy as np

params = {
    'batch_size': 128,
    'text_iter_step': 25,
    'seq_len': 200,
    'kernel_sz': 5,
    'hidden_dim': 128,
    'n_hidden_layer': 4,
    'dropout_rate': 0.1,
    'display_step': 10,
    'generate_step': 100,
}

def parse_text(file_path):
    with open(file_path) as f:
        text = f.read()
    
    char2idx = {c: i+3 for i, c in enumerate(set(text))}
    char2idx['<pad>'] = 0
    char2idx['<start>'] = 1
    char2idx['<end>'] = 2
    
    ints = np.array([char2idx[char] for char in list(text)])
    return ints, char2idx

def next_batch(ints):
    len_win = params['seq_len'] * params['batch_size']
    for i in range(0, len(ints)-len_win, params['text_iter_step']):
        clip = ints[i: i+len_win]
        yield clip.reshape([params['batch_size'], params['seq_len']])
        
def input_fn(ints):
    dataset = tf.data.Dataset.from_generator(
        lambda: next_batch(ints), tf.int32, tf.TensorShape([None, params['seq_len']]))
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def start_sent(x):
    _x = tf.fill([tf.shape(x)[0], 1], params['char2idx']['<start>'])
    return tf.concat([_x, x], 1)

def end_sent(x):
    _x = tf.fill([tf.shape(x)[0], 1], params['char2idx']['<end>'])
    return tf.concat([x, _x], 1)

def embed_seq(x, vocab_sz, embed_dim, name, zero_pad=True):
    embedding = tf.get_variable(name, [vocab_sz, embed_dim])
    if zero_pad:
        embedding = tf.concat([tf.zeros([1, embed_dim]), embedding[1:, :]], 0)
    x = tf.nn.embedding_lookup(embedding, x)
    return x


def position_embedding(inputs):
    T = inputs.get_shape().as_list()[1]
    x = tf.range(T)                            # (T)
    x = tf.expand_dims(x, 0)                   # (1, T)
    x = tf.tile(x, [tf.shape(inputs)[0], 1])   # (N, T)
    return embed_seq(x, T, params['hidden_dim'], 'position_embedding')

def cnn_block(x, dilation_rate, pad_sz, is_training):
    pad = tf.zeros([tf.shape(x)[0], pad_sz, params['hidden_dim']])
    x =  tf.layers.conv1d(inputs = tf.concat([pad, x, pad], 1),
                          filters = params['hidden_dim'],
                          kernel_size = params['kernel_sz'],
                          dilation_rate = dilation_rate)
    x = x[:, :-pad_sz, :]
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, params['dropout_rate'], training=is_training)
    return x


def forward(inputs, reuse, is_training):
    inputs = start_sent(inputs)
    with tf.variable_scope('model', reuse=reuse):
        x = embed_seq(inputs, params['vocab_size'], params['hidden_dim'], 'word_embedding')
        x += position_embedding(x)
        
        for i in range(params['n_hidden_layer']):
            dilation_rate = 2 ** i
            pad_sz = (params['kernel_sz'] - 1) * dilation_rate
            x += cnn_block(x, dilation_rate, pad_sz, is_training)
        
        logits = tf.layers.dense(x, params['vocab_size'])
    return logits

def autoregressive():
    def cond(i, x, temp):
        return i < params['seq_len']

    def body(i, x, temp):
        logits = forward(x, reuse=True, is_training=False)
        ids = tf.argmax(logits, -1, output_type=tf.int32)[:, i]
        ids = tf.expand_dims(ids, -1)

        temp = tf.concat([temp[:, 1:], ids], -1)

        x = tf.concat([temp[:, -(i+1):], temp[:, :-(i+1)]], -1)
        x = tf.reshape(x, [1, params['seq_len']])
        i += 1
        return i, x, temp

    x = tf.zeros([1, params['seq_len']], tf.int32)
    _, res, _ = tf.while_loop(cond, body, [tf.constant(0), x, x])
    
    return res[0]


ints, params['char2idx'] = parse_text('Data/Zulu.txt')
params['vocab_size'] = len(params['char2idx'])
params['idx2char'] = {i: c for c, i in params['char2idx'].items()}
print('Vocabulary size:', params['vocab_size'])

X = input_fn(ints)
logits = forward(X, reuse=False, is_training=True)

ops = {}
ops['global_step'] = tf.Variable(0, trainable=False)

targets = end_sent(X)
ops['loss'] = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
    logits = logits,
    targets = targets,
    weights = tf.to_float(tf.ones_like(targets))))

ops['train'] = tf.train.AdamOptimizer().minimize(ops['loss'], global_step=ops['global_step'])

ops['generate'] = autoregressive()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
while True:
    try:
        _, step, loss = sess.run([ops['train'], ops['global_step'], ops['loss']])
    except tf.errors.OutOfRangeError:
        break
    else:
        if step % params['display_step'] == 0 or step == 1:
            print("Step %d | Loss %.3f" % (step, loss))
        if step % params['generate_step'] == 0 and step > 1:
            ints = sess.run(ops['generate'])
            print('\n'+''.join([params['idx2char'][i] for i in ints])+'\n')

