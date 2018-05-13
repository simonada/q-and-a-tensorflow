# q-and-a-tensorflow
CS707: Q & A with TensorFlow Project

The project aims at exploring different alternatives to implement a neural network model in TensorFlow to solve three of babI's question answering tasks (https://research.fb.com/downloads/babi/). Preprocessing of the data is encapsulated in a Python module - InputPreparator.py. It is used in the three Notebooks: 
1. Q&A with TF- tf.layers and low-level API
2. Q&A with TF- tf.keras
3. Q&A with TF- TFRecords and Eager Execution : here the logic of the tf.keras model from 2. is encapsulated in a Class. The main difference to the previous solution is that TFRecords is used for the input pipeline. 

I. Prerequisites:
1. TensorFlow 1.7 : https://www.tensorflow.org/install/
2. Python packages 
- tqdm : https://pypi.python.org/pypi/tqdm#installation
3. Datasets
- GloVe pre-trained embeddings:  http://nlp.stanford.edu/data/glove.6B.zip
- The tasks data: https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz

II. Running the models:
1. Download the data if not yet done and store it in 'data' in the same folder where the notebooks are
2. Run "Q&A with TF- tf.layers and low-level API": the notebook contains the most explanations on the models logic, and also outputs the wrong predictions made at the end of the training and testing phase.
3.  Run any of the other notebooks.

III. Visualising in TensorBoard:
Name and variable scopes were used throughout in order to separate different conceptual contexts and aid visualization of the graph in TensorBoard. Tracking of the scalar outputs from the defined Loss function and a computed Accuracy score was implemented as well. Furthermore, the embeddings are visualised using the available Projector feature for the tool. 

1. In a terminal navigate to the folder where the project notebooks are. There should be a folder "log".
2. Type "tensorboard --logdir=log" or "tensorboard --logdir=keraslog" in a terminal, and navigate to the pointer URL in a browser. 
3.  Relevant tabs:
- Graphs - giving an overview of the model architecture
- Scalars - for accuracy and loss evolution during train/ validation
- Projector for the word embeddings
