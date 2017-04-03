# -*- coding: utf-8 -*-
"""
A neural network implemented in Keras for NER (named entity recognition)
in HTML documents.
"""

import os
import pickle
import tempfile
from configparser import ConfigParser, ExtendedInterpolation
from random import randint

import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.layers import Dense, Embedding, Activation, LSTM, GRU, SimpleRNN
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics.classification import precision_recall_fscore_support, \
    confusion_matrix
try:
    from tqdm import tqdm
except ImportError:
    tqdm=None
    print('\n[!] For progress logging during metrics calculation '
          'install tqdm.\n')

from html_tokenizer import html_to_tokens, tagged_html_to_tuples

UNKNOWN_TOKEN = '*** unknown token ***'
OTHER_ENTITY = 'O'


def generate_tuples_from_file(fpath: str, *,
                              encodings: dict,
                              first_layer: str,
                              batch_size: int=1,
                              inputs_only: bool=False
                              ):
    """
    Generator of tuples of inputs and outputs for the neural network
    to prevent using unnecessary amounts of memory.
    The generator is expected to loop over its data
    indefinitely as specified in the Keras functions
    using generators.

    :param fpath:       Path to file containing tab separated inputs &
                        possibly outputs.
    :param encodings:   Dict that contains encoding dictionaries used to
                        assign integers to strings and vice-versa.
                        For more info see the _build_encodings function.
    :param batch_size:  How many inputs & outputs should be
                        yielded at once.
    :param inputs_only: Whether to yield inputs only, used when outputs are
                        not known (predicting outputs).
    :param first_layer: If the first layer is an embedding one, pass inputs
                        as a vector of integers. If it is a dense layer,
                        pass the inputs as a one-hot encoded array.
    :return:
    """

    max_len = encodings['max_seq_len']
    vocab_size = encodings['vocab_size']
    no_of_labels = encodings['no_of_labels']

    while 1:
        with open(fpath) as _f:
            inputs = []
            outputs = []
            curr_inputs = []
            curr_outputs = []

            for orig_line in _f:
                line = orig_line.rstrip('\n')
                if inputs_only:
                    # Yield inputs only, for predicting
                    if orig_line == '\n':
                        # Newline character separating documents,
                        # add the current document encoded to the
                        # inputs and yield if there are enough inputs
                        inputs.append(curr_inputs)

                        if len(inputs) == batch_size:
                            x_enc = pad_sequences(inputs,
                                                  maxlen=max_len)
                            if first_layer == 'Embedding':
                                yield x_enc
                            elif first_layer == 'Dense':
                                x_cat = to_categorical(x_enc, vocab_size)
                                x_shaped = x_cat.reshape(batch_size,
                                                         max_len,
                                                         vocab_size)
                                yield x_shaped

                            inputs = []
                            curr_inputs = []
                    else:
                        try:
                            curr_inputs.append(encodings['token2ind'][line])
                        except KeyError:
                            # Failed to encode this line, because it is not
                            # in the encodings vocabulary
                            curr_inputs.append(encodings['token2ind']
                                               [UNKNOWN_TOKEN])

                else:
                    # Yield both inputs and outputs, for training
                    # and evaluation
                    token_lst = line.split('\t')
                    if orig_line != '\n' and len(token_lst) != 2:
                        print('Inputs & outputs not found on line: {},'
                              ' skipping'.format(line))
                    else:
                        if orig_line == '\n':
                            # Newline character separating documents,
                            # add the current document encoded to the
                            # inputs and outputs and yield if there are
                            # enough of them
                            inputs.append(curr_inputs)
                            outputs.append(curr_outputs)

                            if len(inputs) == batch_size:
                                x_enc = pad_sequences(inputs,
                                                      maxlen=max_len)

                                y_enc = pad_sequences(outputs,
                                                      maxlen=max_len)
                                y_cat = to_categorical(y_enc, no_of_labels)
                                y_shaped = y_cat.reshape(batch_size,
                                                         max_len,
                                                         no_of_labels)

                                if first_layer == 'Embedding':
                                    yield x_enc, y_shaped
                                elif first_layer == 'Dense':
                                    x_cat = to_categorical(x_enc, vocab_size)
                                    x_shaped = x_cat.reshape(batch_size,
                                                             max_len,
                                                             vocab_size)
                                    yield x_shaped, y_shaped

                                inputs = []
                                outputs = []
                                curr_inputs = []
                                curr_outputs = []
                        else:
                            token, label = token_lst
                            try:
                                curr_inputs.append(encodings['token2ind']
                                                   [token])
                                curr_outputs.append(encodings['label2ind']
                                                    [label])
                            except KeyError:
                                # Failed to encode this line, because it is not
                                # in the encodings vocabulary
                                curr_inputs.append(encodings['token2ind']
                                                   [UNKNOWN_TOKEN])
                                curr_outputs.append(encodings['label2ind']
                                                    [label])


class MetricsCalculator(Callback):
    """
    Class used to calculate metrics  
    (precision, recall, F score and support) 
    during and after training.
    """
    def __init__(self, fpath: str, encodings: dict, first_layer: str,
                 batch_size: int, steps: int, *,
                 model: Sequential=None):
        """
        
        :param fpath:       Path to file containing tab separated inputs &
                            possibly outputs.
        :param encodings:   Dict that contains encoding dictionaries used to
                            assign integers to strings and vice-versa.
                            For more info see the _build_encodings function.
        :param first_layer: The first layer of the neural network. The inputs
                            depend on whether the first layer is an Embedding
                            layer (inputs = integers) or a Dense layer
                            (inputs = one-hot encoded array).
        :param batch_size:  How many inputs & outputs should be
                            used to calculate scores at once.
        :param steps:       How many times it needs to go retrieve new inputs
                            to go over all documents.
                            Equal to the number of documents in fpath divided
                            by the batch size.
        :param model:       Model to use when using this class during
                            evaluation (not as a callback during training).
        """
        super(MetricsCalculator, self).__init__()
        self.fpath = fpath
        self.encodings = encodings
        self.first_layer = first_layer
        self.batch_size = batch_size
        self.steps = steps
        if model is not None:
            self.set_model(model)

        self.results = None

    @staticmethod
    def _score(yh, pr):
        """
        :param yh:  A numpy array of expected class predictions.
        :param pr:  A numpy array of class predictions.
        :return:    A bool tensor.
        """
        coords = [np.where(yhh > 0)[0][0] for yhh in yh]
        yh = [yhh[co:] for yhh, co in zip(yh, coords)]
        ypr = [prr[co:] for prr, co in zip(pr, coords)]
        fyh = [c for row in yh for c in row]
        fpr = [c for row in ypr for c in row]
        return fyh, fpr

    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        When an epoch of training is done, print the mean of all
        F1 scores.

        :param epoch:   The number of the epoch that just finished.
        :param logs:    Dictionary of logs.
        :return:
        """
        self.calculate()
        print('[i] F1 mean for {fpath}: {res}'.
              format(fpath=self.fpath,
                     res=self.results['f1mean']))

    def on_train_end(self, logs=None) -> None:
        """
        When training is done, print all results.

        :param logs:    Dictionary of logs.
        :return:
        """
        print(self.fpath)
        for key, value in self.results.items():
            print('{k} : {v}'.format(k=key, v=value))

    def all_results(self) -> dict:
        """
        Returns all of the metrics 
        (precision, recall, F score and support) 
        in a dictionary.

        :return:
        """
        if self.results is not None:
            return self.results
        else:
            self.calculate()
            return self.results

    def calculate(self) -> None:
        """
        Calculates all of the metrics
        (precision, recall, F score and support)
        and stores them
        in the results dictionary.
        Note: This function may eat up a lot of memory
        if it's used on a large file.

        :return:
        """
        print('\nCalculating metrics...')
        ftr_all = []
        fpr_all = []

        gen = generate_tuples_from_file(self.fpath,
                                        encodings=self.encodings,
                                        first_layer=self.first_layer,
                                        batch_size=self.batch_size)

        if tqdm:
            for _ in tqdm(range(self.steps)):
                x, y = next(gen)

                y_pred = self.model.predict_classes(x, verbose=0)
                y_true = y.argmax(2)

                ftr, fpr = self._score(y_true, y_pred)
                ftr_all.extend(ftr)
                fpr_all.extend(fpr)
        else:
            print('[!] For progress logging during metrics calculation '
                  'install tqdm.')
            for _ in range(self.steps):
                x, y = next(gen)

                y_pred = self.model.predict_classes(x, verbose=0)
                y_true = y.argmax(2)

                ftr, fpr = self._score(y_true, y_pred)
                ftr_all.extend(ftr)
                fpr_all.extend(fpr)

        confusion = confusion_matrix(ftr_all, fpr_all)
        p, r, f, s = precision_recall_fscore_support(ftr_all, fpr_all)

        self.results = {
            'confusion_matrix': confusion,
            'precision': p,
            'recall': r,
            'fscore': f,
            'f1mean': np.mean(f),
            'support': s
        }


class NeuralNER:
    """
    Wrapper class for training, evaluating and using
    a neural network for named entity recognition
    using Keras.
    """
    def __init__(self, params: dict = None,
                 other_entity_str: str = OTHER_ENTITY):
        """
        :param params:              Dictionary containing parameters
                                    to be used, such as train_split,
                                    which defines the ratio of documents
                                    to be used for training the network.
                                    For more information, see the
                                    parse_config function.
        :param other_entity_str:    The string to be used to describe
                                    tokens not belonging to any entity
                                    class.
        """
        self.params = params or {}
        self._other_entity = other_entity_str

        self.total_docs = None
        self.model = None
        self.encodings = None

    def _build_encodings(self) -> None:
        """
        Builds dictionaries containing the encodings of strings
        to integers and vice-versa. It also counts the number of
        occurrences of the strings and can be set to omit strings
        that do not occur often by specifying the min_token_count
        in the configuration file.

        :return:
        """
        train_fp = self.params['train_fp']
        val_fp = self.params['val_fp']
        eval_fp = self.params['eval_fp']
        encodings_fp = self.params['encodings_fp']
        min_token_count = self.params['min_token_count']

        print('Building encodings for {}'.format(self.params['data_fp']))

        token_vocab = set()
        token_counts = {}
        label_vocab = set()
        seq_lengths = []
        seq_len_counter = 0

        for fp in (train_fp, val_fp, eval_fp):
            with open(fp) as tokenized:
                for line in tokenized:
                    seq_len_counter += 1
                    token_label_lst = line.split('\t')
                    if line == '\n':
                        # Newline character separating documents,
                        # store the length of the sequence
                        seq_lengths.append(seq_len_counter - 1)
                        seq_len_counter = 0
                    elif len(token_label_lst) != 2:
                        # Some kind of a bad line
                        print('Bad line: {}'.format(line))
                    else:
                        token, label = token_label_lst
                        try:
                            t_count = token_counts[token]
                            if t_count > min_token_count:
                                token_vocab.add(token)
                            else:
                                token_counts[token] += 1
                        except KeyError:
                            token_counts[token] = 1
                        label_vocab.add(label.rstrip())

        token2ind = {word: (index+1) for index, word in enumerate(token_vocab)}
        ind2token = {(index+1): word for index, word in enumerate(token_vocab)}
        token2ind[UNKNOWN_TOKEN] = 0
        ind2token[0] = UNKNOWN_TOKEN

        label2ind = {word: (index+1) for index, word in enumerate(label_vocab)}
        ind2label = {(index+1): word for index, word in enumerate(label_vocab)}

        self.encodings = {
            'token2ind': token2ind,
            'ind2token': ind2token,
            'label2ind': label2ind,
            'ind2label': ind2label,
            'vocab_size': len(token_vocab)+1,
            'no_of_labels': len(label_vocab)+1,
            'max_seq_len': max(seq_lengths)
        }

        print('Saving encodings to {}'.format(encodings_fp))
        os.makedirs(os.path.dirname(encodings_fp), exist_ok=True)
        with open(encodings_fp, 'wb') as pklfile:
            pickle.dump(self.encodings, pklfile)

    def _load_encodings(self, build_new: bool = True, *,
                        enc_fp: str = None)-> None:
        """
        Helper function that loads existing encodings 
        from a pickle file.
        If existing ones do not exist and build_new=True,
        it will build new encodings.
        Pass build_new=False when using this function and
        new encodings should definitely not be built, like
        during evaluation or while predicting.

        :param build_new:   Whether to build new encodings 
                            if none exist.
        :param enc_fp:      Path to the encodings pickle file.
        :return:
        """
        enc_fp = enc_fp or self.params['encodings_fp']
        if os.path.isfile(enc_fp):
            print('Loading encodings from {}'.format(enc_fp))
            with open(enc_fp, 'rb') as pklfile:
                self.encodings = pickle.load(pklfile)
        elif build_new:
            self._build_encodings()
        else:
            raise FileNotFoundError('Encodings not found at {}'.
                                    format(enc_fp))

    def _load_model(self, *, model_fp: str = None) -> None:
        """
        Helper function that loads a model into memory.

        :param model_fp: Path to the model file.
        :return:
        """
        model_fp = model_fp or self.params['model_fp']
        if os.path.isfile(model_fp):
            print('Loading model from {}'.format(model_fp))
            self.model = load_model(model_fp)
        else:
            raise FileNotFoundError('Model not found at {}'.
                                    format(model_fp))

    def _compile_model(self) -> None:
        """
        Compiles a neural network model with the parameters
        specified in the config file. For more information,
        see the parse_config function.

        :return:
        """
        vocab_size = self.encodings['vocab_size']
        max_input_length = self.encodings['max_seq_len']
        output_size = self.encodings['no_of_labels']

        first_layer = self.params['first_layer']
        dense_size = self.params['dense_size']
        embed_size = self.params['embed_size']

        hidden_size = self.params['hidden_size']
        rec_layer_type = self.params['rec_layer_type']
        rec_layers = self.params['rec_layers']
        last_rec_size = self.params['last_rec_size']

        self.model = Sequential()

        if first_layer == 'Embedding':
            self.model.add(Embedding(vocab_size+1,
                                     embed_size,
                                     input_length=max_input_length,
                                     mask_zero=True,))
        elif first_layer == 'Dense':
            self.model.add(Dense(dense_size,
                                 input_shape=(max_input_length, vocab_size)))
        else:
            raise ValueError('Undefined first layer {}, expected'
                             'Embedding or Dense.'.format(first_layer))

        if rec_layer_type == 'LSTM':
            for _ in range(rec_layers):
                self.model.add(Bidirectional(LSTM(hidden_size,
                                                  return_sequences=True)))
            self.model.add(Bidirectional(LSTM(last_rec_size,
                                              return_sequences=True)))
        elif rec_layer_type == 'GRU':
            for _ in range(rec_layers):
                self.model.add(Bidirectional(GRU(hidden_size,
                                                 return_sequences=True)))
            self.model.add(Bidirectional(GRU(last_rec_size,
                                             return_sequences=True)))
        elif rec_layer_type == 'SimpleRNN':
            for _ in range(rec_layers):
                self.model.add(Bidirectional(SimpleRNN(hidden_size,
                                                       return_sequences=True)))
            self.model.add(Bidirectional(SimpleRNN(last_rec_size,
                                                   return_sequences=True)))

        self.model.add(TimeDistributed(Dense(output_size)))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def _tuples_to_inline_format(self, tuples: list) -> str:
        """
        Converts a list of tagged tuples to a string containing the tags
        inline and the average of the predicted probability, ie.
        [('I', 'O'), (' ', 'O'), ('am', 'O'), (' ', 'O'), 
        ('from', 'O'), (' ', 'O'), ('New', 'LOC'), 
        (' ', 'LOC'), ('York', 'LOC')]
        ->
        'I am from <LOC prob=0.87654>New York<END>'

        :param tuples: List of tagged tuples.
        :return: String containing the tags inline.
        """
        inline_str = ''

        prev_ent_type = None
        entity_content = []
        entity_probs = []

        tuples_iter = iter(tuples)

        for token, ent_type, ent_prob in tuples_iter:
            if prev_ent_type is None:
                # First word of the document
                if ent_type == self._other_entity:
                    inline_str += token
                else:
                    entity_content = [token]
                    entity_probs = [ent_prob]
                prev_ent_type = ent_type
            elif prev_ent_type != ent_type:
                # Different entity type than before

                # Handle possibly existing entity content
                if len(entity_content):
                    inline_str += '<{prev_ent_type} ' \
                                  'prob={prob}>{tokens}</{prev_ent_type}>'.\
                        format(prev_ent_type=prev_ent_type,
                               prob=np.average(entity_probs),
                               tokens=''.join(entity_content))

                entity_content = []
                entity_probs = []

                if ent_type == self._other_entity:
                    inline_str += token
                else:
                    entity_content = [token]
                    entity_probs = [ent_prob]

                prev_ent_type = ent_type
            elif prev_ent_type == ent_type:
                # Same entity as before
                if prev_ent_type == self._other_entity:
                    inline_str += token
                else:
                    entity_content.append(token)
                    entity_probs.append(ent_prob)

        # In case anything is left
        if len(entity_content):
            if prev_ent_type != self._other_entity:
                inline_str += '<{prev_ent_type} prob={prob}>' \
                              '{tokens}' \
                              '</{prev_ent_type}>'. \
                    format(prev_ent_type=prev_ent_type,
                           prob=np.average(entity_probs),
                           tokens=''.join(entity_content))
            else:
                inline_str += ''.join(entity_content)

        return inline_str

    @staticmethod
    def _count_docs_in_file(fp: str):
        with open(fp) as f:
            docs = sum(1 for _ in f)
        return docs

    def tokenize_split(self) -> None:
        """
        Tokenizes and splits the dataset into separate files 
        containing data for training, validation and evaluation 
        according to parameters specified in the configuration file.

        :return:
        """
        data_fp = self.params['data_fp']
        entities = self.params['entities']
        char_by_char = self.params['char_by_char']
        random = self.params['randomize_order']

        train_fp = self.params['train_fp']
        train_split = self.params['train_split']
        val_fp = self.params['val_fp']
        val_split = self.params['val_split']
        eval_fp = self.params['eval_fp']
        eval_split = self.params['eval_split']

        print('Splitting into train, validation and evaluation sets')

        if not sum([train_split, val_split, eval_split]) == 1:
            raise ValueError('Split ratios do not add up to 1.0')

        total_docs = self._count_docs_in_file(data_fp)

        train_docs = round(total_docs * train_split)
        val_docs = round(total_docs * val_split)
        eval_docs = round(total_docs * eval_split)

        with open(data_fp) as data_f,\
                open(train_fp, 'w') as train_f,\
                open(val_fp, 'w') as val_f,\
                open(eval_fp, 'w') as eval_f:
            train_complete = 0
            val_complete = 0
            eval_complete = 0

            def write_tuples_to_file(doc_tups: list, f):
                for token, ent_type in doc_tups:
                    f.write('{token}\t{ent_type}\n'.
                            format(token=token,
                                   ent_type=ent_type))
                f.write('\n')

            for row in data_f:
                doc_tuples = tagged_html_to_tuples(
                    doc=row.strip(),
                    other_ent_type=self._other_entity,
                    entities=entities,
                    char_by_char=char_by_char)

                if not random:
                    # Do not randomize == first add to train dataset,
                    # then validation, then evaluation
                    if not train_complete == train_docs:
                        write_tuples_to_file(doc_tuples, train_f)
                        train_complete += 1
                    elif not val_complete == val_docs:
                        write_tuples_to_file(doc_tuples, val_f)
                        val_complete += 1
                    elif not eval_complete == eval_docs:
                        write_tuples_to_file(doc_tuples, eval_f)
                        eval_complete += 1
                else:
                    # Randomize the order of the documents
                    while 1:
                        # 0 for train, 1 for validation, 2 for evaluation
                        rand = randint(0, 2)

                        if rand == 0 and train_docs > train_complete:
                            # Add it to the train file
                            write_tuples_to_file(doc_tuples, train_f)
                            train_complete += 1
                        elif rand == 1 and val_docs > val_complete:
                            # Add it to the validation file
                            write_tuples_to_file(doc_tuples, val_f)
                            val_complete += 1
                        elif rand == 2 and eval_docs > eval_complete:
                            # Add it to the train file
                            write_tuples_to_file(doc_tuples, eval_f)
                            eval_complete += 1
                        else:
                            continue
                        break

        self.total_docs = total_docs

    def train(self) -> None:
        """
        Trains a neural network with parameters specified in
        the configuration file. For more information, see the
        parse_config function.

        :return:
        """
        model_fp = self.params['model_fp']

        # Training params
        train_fp = self.params['train_fp']
        train_split = self.params['train_split']
        val_fp = self.params['val_fp']
        val_split = self.params['val_split']
        first_layer = self.params['first_layer']
        batch_size = self.params['batch_size']
        epochs = self.params['epochs']

        # Make sure the documents have been split up
        if not self.total_docs:
            self.tokenize_split()

        if self.encodings is None:
            self._load_encodings()

        # Compile the model
        self._compile_model()

        # Steps per epoch - the number of unique samples of the dataset
        # divided by the batch size
        train_docs = round(train_split * self.total_docs)
        train_steps = int(round(train_docs / batch_size))
        val_docs = round(val_split * self.total_docs)
        val_steps = int(round(val_docs / batch_size))

        # Set up callbacks - checkpointing the model
        # and logging to TensorBoard
        suffix = '{epoch: 02d}'
        callbacks = [ModelCheckpoint(filepath=
                                     '{name} - {suffix}.h5'.
                                     format(name=model_fp.replace('.h5', ''),
                                            suffix=suffix)),
                     MetricsCalculator(fpath=train_fp,
                                       encodings=self.encodings,
                                       first_layer=first_layer,
                                       batch_size=batch_size,
                                       steps=train_steps),
                     MetricsCalculator(fpath=val_fp,
                                       encodings=self.encodings,
                                       first_layer=first_layer,
                                       batch_size=batch_size,
                                       steps=val_steps)]

        if self.params['tb_logging']:
            log_dir = self.params['tb_logdir']
            tb_callback = TensorBoard(log_dir=log_dir,
                                      histogram_freq=1,
                                      write_graph=True,
                                      write_images=True)
            callbacks.append(tb_callback)

        # Train the model
        print('Starting training with {t} training samples and '
              '{v} validation samples...'.format(t=train_docs,
                                                 v=val_docs))

        gen = generate_tuples_from_file(train_fp,
                                        encodings=self.encodings,
                                        first_layer=first_layer,
                                        batch_size=batch_size)

        self.model.fit_generator(generator=gen,
                                 steps_per_epoch=train_steps,
                                 epochs=epochs,
                                 callbacks=callbacks)

        print('Saving final model to {}'.format(model_fp))
        self.model.save(model_fp)

    def eval(self) -> dict:
        """
        Evaluates the model using the data in the evaluation file.

        :return:    A dictionary containing the calculated metrics
                    (precision, recall, F score and support) on 
                    the evaluation set.
        """
        eval_fp = self.params['eval_fp']
        eval_split = self.params['eval_split']
        first_layer = self.params['first_layer']
        batch_size = self.params['batch_size']

        # Count the documents to evaluate
        if not self.total_docs:
            eval_docs = self._count_docs_in_file(fp=eval_fp)
        else:
            eval_docs = round(eval_split * self.total_docs)
        eval_steps = int(round(eval_docs / batch_size))

        if self.encodings is None:
            self._load_encodings(build_new=False)

        if self.model is None:
            self._load_model()

        score_calc = MetricsCalculator(fpath=eval_fp,
                                       encodings=self.encodings,
                                       first_layer=first_layer,
                                       batch_size=batch_size,
                                       steps=eval_steps,
                                       model=self.model)

        results = score_calc.all_results()
        for key, value in results.items():
            print('{k} : {v}'.format(k=key, v=value))

        return results

    def prepare_for_tagging(self) -> None:
        """
        Makes sure the model and encodings are loaded.
        
        :return: 
        """
        if self.encodings is None:
            self._load_encodings(build_new=False)

        if self.model is None:
            self._load_model()

    def tag_documents(self, docs: list) -> list:
        """
        Returns a list of inline tagged documents.

        :param docs: List containing documents as strings.
        :return: List of inline tagged documents as strings.
        """
        char_by_char = self.params['char_by_char']
        batch_size = self.params['batch_size']
        first_layer = self.params['first_layer']

        self.prepare_for_tagging()

        # Create a temporary file containing the documents tokenized
        _, _input_filepath = tempfile.mkstemp(text=True)

        # Write the actual documents to the temporary input file
        no_of_docs = 0
        with open(_input_filepath, 'w') as inputfile:
            for doc in docs:
                no_of_docs += 1
                tokens = html_to_tokens(doc, char_by_char=char_by_char)
                for t in tokens:
                    inputfile.write('{}\n'.format(t))
                # Newline after every document to separate them
                inputfile.write('\n')

        # Encode the inputs
        gen = generate_tuples_from_file(_input_filepath,
                                        encodings=self.encodings,
                                        first_layer=first_layer,
                                        batch_size=no_of_docs,
                                        inputs_only=True)
        x = next(gen)

        try:
            start_coords = [np.where(_x.argmax(1) > 0)[0][0] for _x in x]
        except ValueError:
            start_coords = [np.where(_x > 0)[0][0] for _x in x]

        # Predict entities
        y_probs = self.model.predict(x, batch_size=batch_size)
        y_pred = y_probs.argmax(axis=2)

        y_probs_from_start = [probs[co:]
                              for probs, co in zip(y_probs, start_coords)]
        y_pred_from_start = [pred[co:]
                             for pred, co in zip(y_pred, start_coords)]

        labels_encoded = [[label for label in y] for y in y_pred_from_start]
        all_probs = [[probs.max() for probs in y] for y in y_probs_from_start]

        # Decode the output
        ind2label = self.encodings['ind2label']
        ind2label[0] = self._other_entity
        labels_decoded = [[ind2label[label] for label in sent_labels]
                          for sent_labels in labels_encoded]

        sents_tuples = []

        for idx, doc in enumerate(docs):
            tokens = html_to_tokens(doc, char_by_char=char_by_char)
            tagged_tuples = []
            labels = iter(labels_decoded[idx])
            labels_probs = iter(all_probs[idx])
            for token in tokens:
                ent_type = next(labels)
                ent_prob = next(labels_probs)
                tagged_tuples.append((token, ent_type, ent_prob))
            sents_tuples.append(tagged_tuples)

        # Convert to 'inline' format instead of lists of tuples
        tagged_inline = [self._tuples_to_inline_format(tuples=tuples)
                         for tuples in sents_tuples]

        os.remove(_input_filepath)
        return tagged_inline


def parse_config(config_fp: str) -> dict:
    """
    Parses necessary parameters from the config file.
    All parameters are described in the config file.

    :param config_fp:   String containing the path to the config file.
    :return:            Dict of parameters.
    """

    if not os.path.isfile(config_fp):
        raise FileNotFoundError('Configuration file not found at {}'.
                                format(config_fp))

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(config_fp)

    # Config file sections
    nn_s = 'Neural Network'
    proj_s = 'Project'
    server_s = 'Server'

    # Load the parameters that are actually in the config file
    params = {
        'data_fp': config[proj_s]['DATA_FILEPATH'],
        'train_fp': config[proj_s]['DATA_FILEPATH'].replace('.txt',
                                                            '_train.txt'),
        'train_split': config.getfloat(proj_s, 'TRAIN_SPLIT'),
        'val_fp': config[proj_s]['DATA_FILEPATH'].replace('.txt',
                                                          '_val.txt'),
        'val_split': config.getfloat(proj_s, 'VAL_SPLIT'),
        'eval_fp': config[proj_s]['DATA_FILEPATH'].replace('.txt',
                                                           '_eval.txt'),
        'eval_split': config.getfloat(proj_s, 'EVAL_SPLIT'),

        'entities': [e.strip() for e in config[proj_s]['ENTITIES'].split(',')],

        'evaluate': config.getboolean(proj_s, 'EVALUATE'),
        'char_by_char': config.getboolean(proj_s, 'CHAR_BY_CHAR'),
        'min_token_count': config.getint(proj_s, 'MIN_TOKEN_COUNT'),
        'randomize_order': config.getboolean(proj_s, 'RANDOMIZE_ORDER'),

        'first_layer': config[nn_s]['FIRST_LAYER'],
        'dense_size': config.getint(nn_s, 'DENSE_SIZE'),
        'embed_size': config.getint(nn_s, 'EMBED_SIZE'),

        'hidden_size': config.getint(nn_s, 'HIDDEN_SIZE'),
        'rec_layer_type': config[nn_s]['REC_LAYER_TYPE'],
        'rec_layers': config.getint(nn_s, 'NO_OF_REC_LAYERS'),
        'last_rec_size': config.getint(nn_s, 'LAST_REC_SIZE'),

        'batch_size': config.getint(nn_s, 'BATCH_SIZE'),
        'epochs': config.getint(nn_s, 'EPOCHS'),

        'tb_logging': config.getboolean(nn_s, 'TB_LOGGING'),

        'server_host': config[server_s]['HOST'],
        'server_port': config[server_s]['PORT'],
    }

    params['model_name'] = \
        '{}'.format(os.path.basename(os.path.splitext(params['data_fp'])[0]))

    params['model_fp'] = \
        '{dir}/{name}_{type}x{layers}x{hidden_size}_' \
        'bsize_{b_size}_eps_{eps}_{c_by_c}{rand}.h5'\
        .format(dir=config[proj_s]['MODELS_DIR'],
                name=params['model_name'],
                type=params['rec_layer_type'],
                layers=params['rec_layers'],
                hidden_size=params['hidden_size'],
                b_size=params['batch_size'],
                eps=params['epochs'],
                c_by_c='c_by_c' if params['char_by_char'] else 'w_by_w',
                rand='_rand' if params['randomize_order'] else '')

    params['encodings_fp'] = \
        '{dir}/encodings/{name}_{c_by_c}_min_t_count_{min_t_count}.pkl'.\
            format(dir=config[proj_s]['MODELS_DIR'],
                   name=params['model_name'],
                   c_by_c='c_by_c' if params['char_by_char'] else 'w_by_w',
                   min_t_count=params['min_token_count'])

    params['tb_logdir'] = '{logdir}/{full_model_name}'.\
        format(logdir=config[nn_s]['TB_LOGDIR'],
               full_model_name=
               os.path.basename(os.path.splitext(params['model_fp'])[0]))

    print('Parameters:')
    for key, value in params.items():
        print('{key}: {value}'.format(key=key, value=value))

    return params

if __name__ == '__main__':
    from time import time
    start = time()

    config_fpath = './config.ini'
    params_dct = parse_config(config_fpath)

    nn_wrapper = NeuralNER(params=params_dct)
    nn_wrapper.train()
    if params_dct['evaluate']:
        nn_wrapper.eval()

    print('Finished in {} seconds.'.format(time() - start))
