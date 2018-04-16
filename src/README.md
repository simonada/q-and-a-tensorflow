# q-and-a-tensorflow
CS707: Q & A with TensorFlow project

Requirements:
1. TensorFlow 1.7 : https://www.tensorflow.org/install/
2. Python packages 
- tqdm : https://pypi.python.org/pypi/tqdm#installation

Running the model:
1. Run the "Q&A with TF - Data Import" notebook to download GloVe and Tasks 5, 6, 10.
2.  Run the "Q&A with TF - Low level API" notebook, note that one can change the task number from there

Visualising in TensorBoard:
1. In a terminal navigate to the folder where the project notebooks are. There should be a folder "log".
2. Type "tensorboard --logdir=log" in a terminal, and navigate to the pointer URL in a browser
3.  Relevant tabs:
- Graphs - giving an overview of the model architecture
- Scalars - for accuracy and loss evolution during train/ validation
- Projector for the word embeddings