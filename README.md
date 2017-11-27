## NER Tagger using TF

This repo uses a BLSTM network for doing NER Tagging in TensorFlow, based off https://www.aclweb.org/anthology/Q16-1026 with the following differences

1. Character-level CNN features are not used
2. Softmax is used instead of Log-Softmax
3. Loss to be reduce is average cross-entropy (averaged over a sentence and then averaged over a batch)
4. We use ADAM with the default parameters


