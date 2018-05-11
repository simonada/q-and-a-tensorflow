# q-and-a-tensorflow
CS707: Q & A with TensorFlow project

The project aims at exploring different alternatives to implement a neural network model to solve three of babI's question answering tasks (https://research.fb.com/downloads/babi/). 

Requirements:
1. TensorFlow 1.7 : https://www.tensorflow.org/install/
2. Python packages 
- tqdm : https://pypi.python.org/pypi/tqdm#installation
3. Datasets
- GloVe pre-trained embeddings:  http://nlp.stanford.edu/data/glove.6B.zip
- The tasks data: https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz

Running the models:
1. Download the data if not yet done and store it in 'data' in the same folder where the notebooks are
2. Run "Q&A with TF- tf.layers and low-level API": the notebook contains the most explanations on the models logic, and also outputs the wrong predictions made at the end of the training and testing phase.
3.  Run any of the other notebooks.

Visualising in TensorBoard:
Name and variable scopes were used throughout in order to separate different conceptual contexts and aid visualization of the graph in TensorBoard. Tracking of the scalar outputs from the defined Loss function and a computed Accuracy score was implemented as well. Furthermore, the embeddings are visualised using the available Projector feature for the tool. 

1. In a terminal navigate to the folder where the project notebooks are. There should be a folder "log".
2. Type "tensorboard --logdir=log" in a terminal, and navigate to the pointer URL in a browser
3.  Relevant tabs:
- Graphs - giving an overview of the model architecture
- Scalars - for accuracy and loss evolution during train/ validation
- Projector for the word embeddings

The implemented solutions are described below (names correspond to the Notebook names):
1. Q&A with TF- tf.layers and low-level API: 

2. Q&A with TF- tf.keras: a tf.keras.Model is created with summary:
\includegraphics[]{../../../../keras.png}

3. Q&A with TF- TFRecords and Eager Execution: the functionalities of the model are encapsulated in a class, which inherits from tf.keras.Model and the building blogs for the model come from the tf.keras. The main difference is in how the gradient are computed, i.e. see function below. Also TFRecords was utilised to generate a tf.data dataset.

def grad(model, sent, quest, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, sent, quest, targets)
    return tape.gradient(loss_value, model.variables), loss_value