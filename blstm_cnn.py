from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BLSTM_CNN(object):
    def __init__(self, embedding_size, lstm_hidden_state, vocab_size, num_classes):
        self.embedding_size = embedding_size
        self.lstm_hidden_state = lstm_hidden_state
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        tf.logging.info("Creating placeholders")
        self._create_placeholders()

        tf.logging.info("Building the BLSTM CNN model")
        self._build_model()

        tf.logging.info("Builing the loss functions")
        self._build_loss_function()

    def _create_placeholders(self):
        self.words = tf.placeholder(shape=[None, None], dtype=tf.int32, name="words")
        self.batch_size = tf.placeholder(shape = [], dtype = tf.int32, name = "batch_size")
        self.seq_lens = tf.placeholder(shape = [None], dtype = tf.int32,  name = "seq_lens") # sentence lengths
        self.labels = tf.placeholder(shape = [None, None], dtype = tf.int32, name = "labels")

    def _build_model(self):
        with tf.variable_scope("BLSTM_CNN"):
            self.word_embedding_matrix = tf.get_variable(shape=[self.vocab_size, self.embedding_size], dtype=tf.float32, name="word_embedding_matrix")

            embedded_words = tf.nn.embedding_lookup(self.word_embedding_matrix, self.words)
            print("embedded_words shape: {}".format(embedded_words.shape))

            self.cell = tf.contrib.rnn.LSTMCell(self.lstm_hidden_state)
            init_state = self.cell.zero_state(batch_size = self.batch_size, dtype = tf.float32)

            self.bi_lstm = tf.nn.bidirectional_dynamic_rnn(cell_fw = self.cell, cell_bw = self.cell, inputs = embedded_words, sequence_length = self.seq_lens, dtype = tf.float32)
            fwd_outputs, back_outputs = self.bi_lstm[0]
            fwd_logits  = tf.layers.dense(fwd_outputs,  units = self.num_classes)
            back_logits = tf.layers.dense(back_outputs, units = self.num_classes)

            logits = fwd_logits + back_logits
            self.preds = tf.nn.softmax(logits)  # this will have probability distribution over classes
            
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels, logits = logits) # (BATCH_SIZE, SENTENCE_LENGTH)
            self.loss = tf.reduce_mean(self.loss) # (BATCH_SIZE)

    def _build_loss_function(self):
        self.opt = tf.train.AdamOptimizer()
        self.train_step = self.opt.minimize(self.loss)
