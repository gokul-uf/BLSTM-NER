from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import DataFeeder as df


if __name__ == "__main__":
	batchSize = 100
	data = df.DataFeeder("", batchSize)

	embeddingSize = 100
	numUnitsLSTM = 100
	vocabularySize = len(data.dataImporter.words)
	numClasses = len(data.dataImporter.nerLabelsDictionary)

	cnn = BLSTM_CNN(embeddingSize, numUnitsLSTM, vocabularySize, numClasses)
        num_steps = 100 # TODO Change
        eval_every = 10 # evaluates every 10 steps now

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(num_steps):
            batch = data.getBatch("ner")
	    feed_dict = {
	        cnn.words: batch["words"],
	    	cnn.batch_size: batch["batch_size"],
	    	cnn.seq_lens: batch["seq_lens"],
	    	cnn.labels: batch["labels"]
	    }

	    _, current_loss = sess.run([cnn.train_step, cnn.loss], feed_dict=feed_dict)

            if i % eval_every == 0:
                print("running evaluation")
                batch = data.get_eval_batch()  # TODO implement this, same as getBatch but on test data
                
	        feed_dict = {
	            cnn.words: batch["words"],
	    	    cnn.batch_size: batch["batch_size"],
	    	    cnn.seq_lens: batch["seq_lens"],
	    	    cnn.labels: batch["labels"]
	        }
                eval_loss = sess.run(cnn.loss, feed_dict = feed_dict)
                print("Eval Loss at {} steps is {}".format(i, eval_loss))
