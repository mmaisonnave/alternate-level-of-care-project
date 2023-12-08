import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import os
import sys
sys.path.append('..')

from utilities import configuration
from utilities import logger
from utilities import health_data


if __name__ == '__main__':
    # Retrieving configuration (paths) and initializing logger
    config = configuration.get_config()
    logging = logger.init_logger(config['all_experiments_log'])
    logging.debug('Starting all experiments ...')

    # Retrieving model and experiment configurations 
    model_configurations = json.load(open(config['models_config'], encoding='utf-8'))
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    logging.debug(f'Using {len(model_configurations):4} different models.')
    logging.debug(f'Using {len(experiment_configurations):4} different configurations.')

    for configuration_id, configuration_dict in experiment_configurations.items():
        logging.debug(f'Running on configuration ID: {configuration_id}')
        params = configuration_dict
        
        # Computing training and testing matrices.
        X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(
               fix_missing_in_testing=params['fix_missing_in_testing'],
               normalize=params['normalize'],
               fix_skew=params['fix_skew'],
               numerical_features=params['numerical_features'],
               categorical_features=params['categorical_features'],
               diagnosis_features=params['diagnosis_features'],
               intervention_features=params['intervention_features'],
               use_idf=params['use_idf'],
               remove_outliers=params['remove_outliers'],
               )
        logging.debug(f'X_train.shape = {X_train.shape}')
        logging.debug(f'y_train.shape = {y_train.shape}')
        logging.debug(f'X_train.shape = {X_test.shape}')
        logging.debug(f'y_train.shape = {y_test.shape}')

        for model_id, model_dict in model_configurations.items():
            logging.debug(f'Working on model ID= {model_id}')

            if os.path.isfile(config['experiment_results']):
                auxdf = pd.read_csv(config['experiment_results'], sep=';')
                model_ids = set([model_id for model_id in auxdf['model_id']])
                configuration_ids = set([model_id for model_id in auxdf['config_id']])
                if model_id in model_ids and configuration_id in configuration_ids:
                    logging.debug(f'SKIPPING, configuration ({configuration_id}) and model ({model_id}) already found ...')
                    continue
                logging.debug('Results not found, running experiments ...')
            
            if 'skipping' in model_dict and model_dict['skipping']==True:
                logging.debug(f'SKIPPING model ({model_id}), as requested by configuration ...')
                continue

            # Creating classification model and training ...
            model = configuration.model_from_configuration(model_dict)
            logging.debug(f'Training model {str(model)}')
            model.fit(X_train, y_train)

            # Evaluating metrics on TRAINING
            y_true = y_train
            y_pred = model.predict(X_train)
            y_score= model.predict_proba(X_train)

            model_name = model_dict['model_name']
            columns = ['Model',
                       'split',
                       'TN',
                       'FP',
                       'FN',
                       'TP',
                       'Precision',
                       'Recall',
                       'F1-Score',
                       'AUC', 
                       'experiment_params', 
                       'model_params', 
                       'config_id', 
                       'model_id'
                       ]

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            vec1 = [model_name,
                    'TRAIN',
                    tn,
                    fp,
                    fn,
                    tp,
                    precision_score(y_true, y_pred,),
                    recall_score(y_true, y_pred,),
                    f1_score(y_true, y_pred,),
                    roc_auc_score(y_true=y_true, y_score=y_pred),
                    str(configuration_dict),
                    str(model_dict),
                    configuration_id,
                    model_id
                    ]
            
            # Evaluating metrics on TESTING
            y_true = y_test
            y_pred = model.predict(X_test)
            y_score= model.predict_proba(X_test)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()    
            vec2 = [model_name,
                    'TEST',
                    tn,
                    fp,
                    fn,
                    tp,
                    precision_score(y_true, y_pred,),
                    recall_score(y_true, y_pred,),
                    f1_score(y_true, y_pred,),
                    roc_auc_score(y_true=y_true, y_score=y_pred),
                    str(configuration_dict),
                    str(model_dict),
                    configuration_id,
                    model_id
                    ]
            m = np.vstack([vec1, vec2])
            new_df = pd.DataFrame(m, columns=columns)

            # Saving results
            if os.path.isfile(config['experiment_results']):
                old_df = pd.read_csv(config['experiment_results'], sep=';')
                logging.debug(f'Previous experiments found: {old_df.shape[0]} entries found')
                new_df = pd.concat([old_df,new_df])

            new_df.to_csv(config['experiment_results'], index=False, sep=';')
            logging.debug(f"Saving results in {config['experiment_results'].split('/')[-1]}")
            logging.debug(f'Saving {new_df.shape[0]} entries.')


