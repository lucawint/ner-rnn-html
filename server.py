# -*- coding: utf-8 -*-
import json

from flask import Flask, request, Response, jsonify

from neuralner import NeuralNER, parse_config


class NERServer:
    def __init__(self, config_fp: str):
        """
        Initializes a new instance of the NER server
        using the parameters specified in the
        configuration file.
        
        :param config_fp: Path to the configuration file.
        """
        self.config_fp = config_fp
        self.app = Flask(__name__)
        self.app.add_url_rule('/ner', 'ner', self.ner, methods=['POST'])

        self.params = parse_config(config_fp)

        self.neuralNER = NeuralNER(params_dct=self.params)

        # Empty run to load the encodings and model
        self.neuralNER.prepare_for_tagging()

    def run(self):
        self.app.run(host=self.params['server_host'],
                     port=self.params['server_port'])

    def ner(self):
        """
        Processes the documents included in 
        the request in json format. Returns
        the documents with the entities tagged
        inline.
        
        :return: 
        """
        if not request.is_json:
            return Response(response='Failed to parse as json', status=400)

        docs = json.loads(request.json)

        # Create a list of documents if necessary
        if isinstance(docs, str):
            docs = [docs]
        elif not isinstance(docs, list):
            return Response(response='Did not a receive a string'
                                     'or list of strings', status=400)

        tagged = self.neuralNER.tag_documents(docs=docs)

        return jsonify(tagged)


if __name__ == '__main__':
    cfg_fp = './config.ini'
    server = NERServer(config_fp=cfg_fp)
    server.run()
