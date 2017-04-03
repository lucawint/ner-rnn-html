# Named Entity Recognition in HTML documents

A recurrent neural network implemented in Keras 
specially designed for
NER in HTML documents.

## Quickstart
1) Place datafile in /data folder.
2) Adjust configuration file to your needs.
3) Run neuralner.py to train (and evaluate)
 the network.
4) Run server.py to process requests with the
 newly trained network.

## Usage

### Input format
A textfile located in the data folder. It 
is expected to have a document per line, 
therefore the number of lines == number of documents.
These documents will be divided into a training, 
validation and evaluation set according to the 
ratios specified in the configuration file (config.ini).
The named entities should be tagged inline, just 
like an ordinary HTML tag, e.g. 

<code>\<body>My name is \<PER>Luca\</PER>, 
I live on the \<LOC>second floor\</LOC>.\</body>
</code>

which is why you have to specify which ones to
recognize in the 
[configuration file](./config.ini), separated
by commas, e.g. PER,LOC,ORG,MISC



### Configuration file
The [configuration file](./config.ini) allows you to 
set all of the parameters. You need to fill out the
following parameters, the rest is optional:
* ROOT_DIR - the root directory of the project
* DATA_FILENAME - the filename to be used for training
* ENTITIES - the entities you want to recognize

You can adjust many parameters of the network, like
its layer types & sizes, whether to work on a 
character level or word level, ...

### Training
Training is done by creating an instance of the 
NeuralNER class and calling the train function.

<code>params = parse_config(config_fp)</code>

<code>nn_wrapper = NeuralNER(params_dct=params)</code>

<code>nn_wrapper.train()</code>

After every epoch, the mean of all F1 scores 
on the training and validation set will be shown.

After the last epoch, the calculated metrics
(precision, recall, F1 score and support) as
well as the confusion matrix 
for the validation set will be shown.

### Evaluation
Evaluation is done by running

<code>results = nn_wrapper.evaluate()</code>

This will show the calculated metrics
(precision, recall, F1 score and support) as
well as the confusion matrix
for the evaluation set.

### Server
You can run the network as an API using 
the server.py file or in a Docker container
using the included Dockerfile.
For the tagging to work, you need to have
a model (.h5 file) and its corresponding 
encodings (located in [models/encodings](./models/encodings))!
If a Docker container is used, copy the trained
model with its encodings to the [Docker folder](./Docker).

The server runs on the port defined in the
[configuration file](./config.ini). The endpoint is at
address:port/ner and accepts data in json.
The json should contain either a document (string)
to process or a list of documents.

Example request (also see [example_request.py](./example_request.py)):

<code>ner_api = '{hostname}:{port}/ner'</code>

<code>resp = requests.post(ner_api, 
json=json.dumps('\<body>My name is Luca, 
I live on the second floor.\</body>'))</code>

The server returns a list of tagged documents:

<code>json.loads(resp.text)</code>

<code>\>>>['\<body>My name is \<name prob=0.1234>Luca\</name>, 
I live on the \<loc prob=0.5678>second floor\</loc>.\</body>']</code>
