from Utils import load_obj
from train_model import build_graph, clean_text, get_batches, get_batch
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

epochs = 1000
batch_size = 500
num_layers = 4
rnn_size = 70
embedding_size = 20
learning_rate = 0.00001
direction = 2
threshold = 1.0
keep_probability = 1.0


vocab_to_int = load_obj("vocab_to_int")
int_to_vocab = load_obj("int_to_vocab")
def text_to_ints(text):
    '''Prepare the text for the model'''
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]


# In[176]:

# Create your own sentence or use one from the dataset
text = "Jeg liker å spise mit og drikke vann."
text = text_to_ints(text)

# random = np.random.randint(0,len(testing_sorted))
# text = testing_sorted[random]
# text = noise_maker(text, 0.95)

checkpoint = "./kp=1,nl=4,th=0.999.ckpt"

model = build_graph(vocab_to_int, keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)

with tf.Session() as sess:
    # Load saved model
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint)

    # Multiply by batch_size to match the model's input parameters
    test_sentences = [list(text) for _ in range(batch_size)]
    test_sentences = get_batch(test_sentences, threshold, vocab_to_int)[2]
    print("test_sentences[0]: ", test_sentences[0])
    answer_logits = sess.run(model.predictions, {model.inputs: test_sentences,
                                                 model.inputs_length: [100] * batch_size,
                                                 model.targets_length: [100] * batch_size,
                                                 model.keep_prob: [1.0]})[0]



# Remove the padding from the generated sentence
pad = vocab_to_int["<PAD>"]

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))
