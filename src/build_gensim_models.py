"""This scripts takes all Gensim model configurations from the file gensim.json and 
creates diagnosis and intervention embedding models using the Gensim library.

It saves the models into the gensim/models folder. 

It saves the logs into the log/gensim.log file.

It retrieves the paths to log and models from paths.yaml which points to gensim.json.

"""
import gensim
import logging
import numpy as np
import json
import os
import sys
sys.path.append('..')

from utilities import health_data
from utilities import configuration
from utilities import logger


def build_intervention_models():
    config = configuration.get_config()
    logging = logger.init_logger(config['gensim_log'])

    # Opening JSON file 
    with open(config['gensim_config'])  as f:
        gensim_config = json.load(f) 

    for intervention_config_name, intervention_config in gensim_config['intervention_configs'].items():
        model_name = os.path.join(config['gensim_model_folder'], intervention_config_name )
        params = {
            "epoch_number": intervention_config['epochs'],
            "vector_size": intervention_config['vector_size']
        }
        if os.path.isfile(model_name):
            logging.debug(f'Skipping model with params: {params}')
        else:

            logging.debug(f'Running params: {params}')

            train, testing = health_data.Admission.get_training_testing_data()

            intervention_data = [gensim.models.doc2vec.TaggedDocument(admission.intervention_code, [ix]) 
                                                                for ix,admission in enumerate(train+testing)]

            model = gensim.models.doc2vec.Doc2Vec(vector_size=params["vector_size"], 
                                                min_count=2, 
                                                epochs=params["epoch_number"], 
                                                dm=params['dm'] if 'dm' in params else 1
                                                )
            model.build_vocab(intervention_data)

            logging.debug('Training ...')
            model.train(intervention_data, 
                        total_examples=model.corpus_count, 
                        epochs=params["epoch_number"],
                        )
            
            logging.debug(f'Saving at: {model_name}')
            model.save(model_name)

def build_diagnosis_models():
    config = configuration.get_config()
    logging = logger.init_logger(config['gensim_log'])

    # Opening JSON file 
    with open(config['gensim_config'])  as f:
        gensim_config = json.load(f) 
    print(gensim_config)
    for diagnosis_config_name, diagnosis_config in gensim_config['diagnosis_configs'].items():
        model_name = os.path.join(config['gensim_model_folder'], diagnosis_config_name )
        params = {
            "epoch_number": diagnosis_config['epochs'],
            "vector_size": diagnosis_config['vector_size']
        }
        if os.path.isfile(model_name):
            logging.debug(f'Skipping model with params: {params}')
        else:

            logging.debug(f'Running params: {params}')

            train, testing = health_data.Admission.get_training_testing_data()

            diagnosis_data = [gensim.models.doc2vec.TaggedDocument(admission.diagnosis.codes, [ix]) 
                                                                for ix,admission in enumerate(train+testing)]

            model = gensim.models.doc2vec.Doc2Vec(vector_size=params["vector_size"], 
                                                min_count=2, 
                                                epochs=params["epoch_number"],
                                                dm=params['dm'] if 'dm' in params else 1
                                                )
            model.build_vocab(diagnosis_data)

            logging.debug('Training ...')
            model.train(diagnosis_data, 
                        total_examples=model.corpus_count, 
                        epochs=params["epoch_number"],
                        )
            
            logging.debug(f'Saving at: {model_name}')
            model.save(model_name)

if __name__ == '__main__':
    build_diagnosis_models()
    build_intervention_models()