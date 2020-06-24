This is a VGG-LSTM based End-to-End Model with attention, for image caption

Python version support: 

Python 3.7


Python packages satisfied:

PyQt5, pickle, pycocotools, cython, nltk, gensim, PIL, numpy, tensorflow


To be noticed, version of tensorflow == 2.1.0


To run the system, ensure the folowing document are in the GUI package:

show3.7.py 
deal.py
Dic.txt
dictionary 
wordvector_dictionary 
background.jpeg 
Model_cat.hdf5 
Model_decoder.hdf5 
Model_decoder.hdf5 


How to run the system:

step 1: open terminal
step 2: type cd ../GUI on terminal
step 3: type python show3.7.py on terminal


Python files' content:

deal.py : Model building modules

show3.7.py :  Interation board modules

To satisfy the above running, you should download the relative code with path below, and put them in the same load with the code files.

https://drive.google.com/file/d/137IEP5DxMa6-AkY9bh2ydQCm4NT06DGc/view?usp=sharing

If you prefer to get a new model with your own setting, you could get the dataset from the path below:

dataset resourse:

https://cocodataset.org/#overview

References:

LSTM Decoder by Keras: https://keras.io/examples/lstm_seq2seq/
Attention layers by Keras: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention
