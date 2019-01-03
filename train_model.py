# coding: utf-8

import copy

import os

from Utils import save_obj

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
import re
from sklearn.model_selection import train_test_split


def load_text(path):
    input_file = os.path.join(path)
    with open(input_file, encoding="utf-8") as f:
        text = f.read()
    return text

def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = text.lower()
    text = re.sub(r'\n', ' ', text)


    return text


# In[3]:
def pre_data():
    global clean_text, sentence, training_sorted, testing_sorted
    path = './texts/'
    text_files = [f for f in listdir(path) if isfile(join(path, f))]

    texts = []
    for text in text_files:
        print(path)
        print(text)
        texts.append(load_text(path + text))
    # In[5]:
    # Compare the number of words in each text
    for i in range(len(texts)):
        print("There are {} words in {}.".format(len(texts[i].split()), text_files[i]))
    # In[9]:
    # Check to ensure the text looks alright
    texts[0][:500]

    # ## Preparing the Data
    # In[10]:

    # In[11]:
    # Clean the text of the texts
    clean_texts = []
    for text in texts:
        clean_texts.append(clean_text(text))
    # In[12]:
    # Check to ensure the text has been cleaned properly
    clean_texts[0][:500]
    # In[13]:
    # Create a dictionary to convert the vocabulary (characters) to integers
    vocab_to_int = {}
    count = 0
    for text in clean_texts:
        for character in text:
            if character not in vocab_to_int:
                vocab_to_int[character] = count
                count += 1
    # Add special tokens to vocab_to_int
    codes = ['<PAD>', '<EOS>', '<GO>']
    for code in codes:
        vocab_to_int[code] = count
        count += 1
    # In[14]:
    # Check the size of vocabulary and all of the values
    vocab_size = len(vocab_to_int)
    print("The vocabulary contains {} characters.".format(vocab_size))
    print(sorted(vocab_to_int))
    save_obj(vocab_to_int, "vocab_to_int")
    # *Note: We could have made this project a little easier by using only lower case words and fewer special characters ($,&,-...), but I want to make this spell checker as useful as possible.*
    # In[15]:
    # Create another dictionary to convert integers to their respective characters
    int_to_vocab = {}
    for character, value in vocab_to_int.items():
        int_to_vocab[value] = character
    save_obj(int_to_vocab, "int_to_vocab")
    # In[16]:
    # Split the text from the texts into sentences.
    sentences = []
    for text in clean_texts:
        for sentence in text.split('.'):
            sentences.append(sentence.strip() + '.')
    print("There are {} sentences.".format(len(sentences)))
    # In[17]:
    # Check to ensure the text has been split correctly.
    sentences[:5]
    # *Note: I expect that you have noticed the very ugly text in the first sentence. We do not need to worry about removing it from any of the texts because will be limiting our data to sentences that are shorter than it.*
    # In[18]:
    # Convert sentences to integers
    int_sentences = []
    for sentence in sentences:
        int_sentence = []
        for character in sentence:
            int_sentence.append(vocab_to_int[character])
        int_sentences.append(int_sentence)
    # In[19]:
    # Find the length of each sentence
    lengths = []
    for sentence in int_sentences:
        lengths.append(len(sentence))
    lengths = pd.DataFrame(lengths, columns=["counts"])
    # In[20]:
    lengths.describe()
    # In[21]:
    # Limit the data we will use to train our model
    max_length = 50
    min_length = 1
    good_sentences = []
    for sentence in int_sentences:
        if len(sentence) <= max_length and len(sentence) >= min_length:
            good_sentences.append(sentence)

    print("We will use {} to train and test our model.".format(len(good_sentences)))
    # *Note: I decided to not use very long or short sentences because they are not as useful for training our model. Shorter sentences are less likely to include an error and the text is more likely to be repetitive. Longer sentences are more difficult to learn due to their length and increase the training time quite a bit. If you are interested in using this model for more than just a personal project, it would be worth using these longer sentence, and much more training data to create a more accurate model.*
    # In[22]:
    # Split the data into training and testing sentences
    training, testing = train_test_split(good_sentences, test_size=0.15, random_state=2)
    print("Number of training sentences:", len(training))
    print("Number of testing sentences:", len(testing))
    # In[23]:
    # Sort the sentences by length to reduce padding, which will allow the model to train faster
    training_sorted = []
    testing_sorted = []
    training_sorted = training
    testing_sorted = testing

    sentences_len = map(len, sentences)
    from collections import Counter
    labels, values = zip(*Counter(sentences_len).items())
    print("labels: ", labels)
    print("values: ", values)

    for i in range(len(labels)):
        print(str(labels[i]) + ":" + str(values[i]))


    # In[24]:
    # Check to ensure the sentences have been selected and sorted correctly
    for i in range(2):
        print(training_sorted[i], len(training_sorted[i]))
    threshold = 0.95
    for sentence in training_sorted[:5]:
        print(sentence)
        print(noise_maker(sentence, threshold, vocab_to_int))
        print()
    return vocab_to_int

# In[36]:

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ø', 'å', ' ' ]


def noise_maker(sentence, threshold, vocab_to_int, depth = 3):
    '''Relocate, remove, or add characters to create spelling mistakes'''

    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0, 1, 1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0, 1, 1)
            # ~33% chance characters will swap locations
            if new_random > 0.75:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i + 1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random > 0.50:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # chance the wrong character is typed
            elif new_random > 0.25:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
            # chance a character will not be typed
            else:
                pass
        i += 1
    if depth <= 0:
        return noisy_sentence
    return noise_maker(noisy_sentence, threshold, vocab_to_int, depth - 1)

# *Note: The noise_maker function is used to create spelling mistakes that are similar to those we would make. Sometimes we forget to type a letter, type a letter in the wrong location, or add an extra letter.*

# In[38]:

# Check to ensure noise_maker is making mistakes correctly.



# # Building the Model

# In[63]:

def model_inputs():
    '''Create palceholders for inputs to the model'''

    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


# In[64]:

def process_encoding_input(targets, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# In[65]:

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    '''Create the encoding layer'''

    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm,
                                                         input_keep_prob=keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop,
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state

    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                            input_keep_prob=keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                            input_keep_prob=keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                            cell_bw,
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output, 2)
            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0]


def buil_model(lstm_sizes, sequence_length, inputs, batch_size, vocab_size, keep_prob_=1.0, num_layers=3):


    lstms = [tf.contrib.rnn.LSTMCell(size) for size in lstm_sizes]

    cells = tf.nn.rnn_cell.MultiRNNCell(lstms)

    init_state = cells.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cells, inputs, sequence_length, initial_state=init_state, swap_memory=False)

    return rnn_outputs, final_state

def build_lstm_layers(lstm_sizes, sequence_length, inputs, batch_size, vocab_size, keep_prob_=1.0):
    """
    Create the LSTM layers
    """


    lstms = [tf.contrib.rnn.LSTMCell(size) for size in lstm_sizes]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

    cell = tf.contrib.rnn.MultiRNNCell(drops)

    initial_state = cell.zero_state(batch_size, tf.float32)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs,  swap_memory=False)

    return lstm_outputs, final_state[-1][-1]


def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_target_length):
    '''Create the training logits'''

    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer)

        training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                                  swap_memory = False,
                                                                  maximum_iterations=max_target_length*2
                                                               )
        return training_logits


# In[67]:

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_target_length, batch_size):
    '''Create the inference logits'''

    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                   swap_memory=False,
                                                                   maximum_iterations=max_target_length*2)

        return inference_logits


# In[68]:

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length,
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    lstms = [tf.contrib.rnn.LSTMCell(size) for size in [rnn_size, rnn_size, rnn_size, rnn_size]]
    cells = tf.nn.rnn_cell.MultiRNNCell(lstms)

    output_layer = Dense(vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                     enc_output,
                                                     inputs_length,
                                                     normalize=False,
                                                     name='BahdanauAttention')

    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(cells,
                                                              attn_mech,
                                                              rnn_size)

    initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size= (batch_size))

    #initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state,
    #                                                                _zero_state_tensors(rnn_size,
    #                                                                                    batch_size,
    #                                                                                    tf.float32))

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  targets_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size)

    return training_logits, inference_logits


# In[69]:

def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):
    '''Use the previous functions to create the training and inference logits'''
    print("vocab size: ", vocab_size)
    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1), name="enc_embeddings")
    #enc_embeddings = tf.Print(enc_embeddings, [enc_embeddings, tf.shape(enc_embeddings)],
    #                             "enc_embeddings: ")
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    #enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers,enc_embed_input, keep_prob, direction)
    enc_output, enc_state = buil_model([rnn_size, rnn_size, rnn_size, rnn_size], inputs_length, enc_embed_input, batch_size, vocab_size, 1.0)

    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1), name="dec_embeddings")
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    training_logits, inference_logits = decoding_layer(dec_embed_input,
                                                       dec_embeddings,
                                                       enc_output,
                                                       enc_state,
                                                       vocab_size,
                                                       inputs_length,
                                                       targets_length,
                                                       max_target_length,
                                                       rnn_size,
                                                       vocab_to_int,
                                                       keep_prob,
                                                       batch_size,
                                                       num_layers,
                                                       direction)

    return training_logits, inference_logits


# In[70]:

def pad_sentence_batch(sentence_batch, vocab_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    #max_sentence = max([len(sentence) for sentence in sentence_batch])
    max_sentence = 100
    batch_ = [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    return batch_


# In[71]:

def get_batches(sentences, batch_size, threshold, vocab_to_int, count):
    """Batch sentences, noisy sentences, and the lengths of their sentences together.
       With each epoch, sentences will receive new mistakes"""

    #for batch_i in range(0, len(sentences) // batch_size):
    for batch_i in range(0, count):
        start_i = batch_i * batch_size
        sentences_batch = copy.deepcopy(sentences[start_i:start_i+batch_size])

        #sentences_batch = sentences[start_i:start_i + batch_size]
        #sentences_batch = sentences[randint(0, batch_size - 1)]
        pad_sentences_batch, pad_sentences_lengths, pad_sentences_noisy_batch, pad_sentences_noisy_lengths = get_batch(
            sentences_batch, threshold, vocab_to_int)

        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths
        #yield np.zeros(pad_sentences_noisy_batch.shape), np.zeros(pad_sentences_batch.shape), pad_sentences_noisy_lengths, pad_sentences_lengths


def get_batch(sentences_batch, threshold, vocab_to_int):
    sentences_batch_noisy = []
    for sentence in sentences_batch:
        sentences_batch_noisy.append(noise_maker(sentence, threshold, vocab_to_int))
    sentences_batch_eos = []
    for sentence in sentences_batch:
        sentence.append(vocab_to_int['<EOS>'])
        sentences_batch_eos.append(sentence)
    pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos, vocab_to_int))
    pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy, vocab_to_int))
    # Need the lengths for the _lengths parameters
    pad_sentences_lengths = []
    for sentence in pad_sentences_batch:
        pad_sentences_lengths.append(len(sentence))
    pad_sentences_noisy_lengths = []
    for sentence in pad_sentences_noisy_batch:
        pad_sentences_noisy_lengths.append(len(sentence))
    return pad_sentences_batch, pad_sentences_lengths, pad_sentences_noisy_batch, pad_sentences_noisy_lengths


# *Note: This set of values achieved the best results.*

# In[72]:

# The default parameters
epochs = 100000
batch_size = 1024
num_layers = 4
rnn_size = 70
embedding_size = 20
learning_rate = 0.000005
direction = 1
threshold = 0.97
keep_probability = 1
save_file_name="./kp=1,nl=4,th=0.999.ckpt"

# In[73]:

def build_graph(vocab_to_int, keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):
    #tf.reset_default_graph()

    # Load the model inputs
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_to_int) + 1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')

    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                targets,
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
        #optimizer = AdamWeightDecayOptimizer(
      #learning_rate=learning_rate,
      #weight_decay_rate=0,
      #beta_1=0.9,
      #beta_2=0.999,
      #epsilon=1e-6,
      #exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        grads = tf.gradients(cost, tf.trainable_variables())
        grads, _ = tf.clip_by_global_norm(grads, 1.0)  # gradient clipping
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        train_op = optimizer.apply_gradients(grads_and_vars)

    # Merge all of the summaries
    merged = tf.summary.merge_all()

    # Export the nodes
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op', 'optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


# ## Training the Model


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


def train(default_graph, session, model, epochs, log_string, vocab_to_int):

        saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())


        checkpoint = "./kp=1,nl=4,th=0.999.ckpt"
        saver.restore(session, checkpoint)

        # Used to determine when to stop the training early
        testing_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0

        stop_early = 0
        stop = 20000  # If the batch_loss_testing does not decrease in 3 consecutive checks, stop training
        epoch_size = 200
        epoch_size_test = 10

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), session.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))

        for epoch_i in range(1, epochs + 1):
                batch_loss = 0
                batch_time = 0

                np.random.shuffle(training_sorted)

                for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                        get_batches(training_sorted, batch_size, threshold, vocab_to_int, epoch_size)):
                    start_time = time.time()
                    #print("input_batch[0]: ", input_batch[0])
                    #print("input_length: ", input_length)
                    #print("target_length: ", target_length)
                    with default_graph.as_default():
                        #print(input_batch.shape)
                        #print(input_batch[0])
                        #print(target_batch[0])
                        loss, _ = session.run([
                                                     model.cost,
                                                     model.train_op],
                                                    {model.inputs: input_batch,
                                                     model.targets: target_batch,
                                                     model.inputs_length: input_length,
                                                     model.targets_length: target_length,
                                                     model.keep_prob: .99})

                    batch_loss += loss
                    end_time = time.time()
                    batch_time += end_time - start_time

                    # Record the progress of training
                    #train_writer.add_summary(summary, iteration)

                    iteration += 1

                print('Epoch {:>3} - Loss: {:>6.10f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              batch_loss / (epoch_size),
                              batch_time))

                batch_loss_testing = 0
                batch_time_testing = 0
                for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                        get_batches(testing_sorted, batch_size, threshold, vocab_to_int, epoch_size_test)):
                    start_time_testing = time.time()
                    #print(target_batch.shape)
                    #print(target_batch)
                    summary, loss = session.run([model.merged,
                                              model.cost],
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: 1})

                    batch_loss_testing += loss
                    end_time_testing = time.time()
                    batch_time_testing += end_time_testing - start_time_testing

                    # Record
                    # the progress of testing
                    test_writer.add_summary(summary, iteration)

                print('Testing Loss: {:>6.10f}, Seconds: {:>4.2f}'
                      .format(batch_loss_testing / (epoch_size),
                              batch_time_testing))


                # If the batch_loss_testing is at a new minimum, save the model
                testing_loss_summary.append(batch_loss_testing)
                if True or batch_loss_testing <= min(testing_loss_summary):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(session, save_file_name)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break

                if stop_early == stop:
                    print("Stopping Training.")
                    break


# In[ ]:


def main():

                vocab_to_int = pre_data()
                print(vocab_to_int["a"])
                print(vocab_to_int["b"])
                print(vocab_to_int["c"])
                print(vocab_to_int["d"])
                log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                        num_layers,
                                                        threshold)

                session = tf.Session()
                default_graph = tf.get_default_graph()

                model = build_graph(vocab_to_int, keep_probability, rnn_size, num_layers, batch_size,
                                    learning_rate, embedding_size, direction)

                train(default_graph, session, model, epochs, log_string, vocab_to_int)


if __name__ == "__main__":
    import time

    time.sleep(0)

    main()



"""
Training Model: kp=1,nl=4,th=0.999
Epoch   1 - Loss: 0.0000340639, Seconds: 353.48
Testing Loss: 0.0000011433, Seconds: 7.00
"""
