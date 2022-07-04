# BiCapsHate Model

This repository contains the implementation of the deep learning model called BiCapsHate model to detect hate speech in online social media posts. The model consists of five layers of deep neural networks. It starts with an input layer to process the input text and follows on to an embedding layer to embed the text into a numeric representation. A BiCaps layer then learns the sequential and contextual representations, a dense layer prepares the model for final classification, and lastly the output layer produces the resulting class as either hate or non-hate speech. It is also aided by our rich set of hand-crafted shallow and deep auxiliary features including the Hatebase lexicon, making the model well-informed. The proposed model is evaluated over five benchmark datasets and demonstrated a significantly better performance than the existing three state-of-the-arts, six neural networks, four BERT, and three machine learning based baseline methods.

-------------------------
Pre-requisite:
-------------------------
1. Twitter REST API
2. Keras 2.2.4
3. Numpy
4. Pandas
5. Python 3.7
6. GloVe
7. Tensorflow 1.15


