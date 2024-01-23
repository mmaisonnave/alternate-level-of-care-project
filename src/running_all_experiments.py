"""
This module contains *no* functions to import. It only contains a single main function that runs all 
Scikit-learn experiments (all models obtained from Scikit-learn that use the fit, predict, and 
predict_proba methods.)

You can pass a single argument (--simulation) that computes how many experiments are pending, but 
runs none and makes no change in disk (dry run).

The main function takes as INPUT:
(1). 'experiment_configuration.json' file and
(2). 'models.json' file
(the path for both files is obtained from paths.yaml)

In (1), I have the configuration on how to compute the training and testing matrices (in the 
config file I describe the pre-processing steps for the matrices).

In (2), I describe the parameters of the models (the kernel for SVM, number of estimators for 
random forests, etc.). 

Both files (1 and 2) have multiple configurations that describe ALL experiments to be run.

For example, the first model (model_0) in `models.json` is a SVC model that contains 
the following parameters:
    "model_0": {
        "model_name": "SVC",
        "C": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
        "shrinking": true,
        "probability": true,
        "tol": 0.001,
        "cache_size": 200,
        "class_weight": null,
        "verbose": false,
        "max_iter": 5000,
        "decision_function_shape": "ovr",
        "break_ties": false,
        "random_state": null,
        "configuration_ids": ["configuration_0"]
    }

Model_0 uses "configuration_0" to set up the matrices (see last line of model config). 
Configuration_0 looks like this:
    "configuration_0": {
        "fix_skew": false,
        "normalize": false,
        "fix_missing_in_testing": true,
        "numerical_features": true,
        "categorical_features": true,
        "diagnosis_features": true,
        "intervention_features": true,
        "use_idf": false,
        "class_balanced": false,
        "remove_outliers": true,
        "under_sample_majority_class": false,
        "over_sample_minority_class": false,
        "smote_and_undersampling":false
    }

In total, more than a hundred experiments are described in the config files, when this methods run, 
it computes all experiments unless they are already computed and we already have the results stored.


The OUTPUT of the experiments is stored in: 'experiments_results.csv' (path obtained from paths.yaml)


On a high level, the main functions does the following:
1. Recovers all experiment and model configurations from config files
2. Computes from all the described experiments how many are already ran.
3. for each experiment_config found in 'experiment_configuration.json':
4.     X_train, y_train, X_test, y_test <= get_matrices_from config(experiment_config)
5.     for each model_config in 'models.json':
6.         model <= get_model_from_config(model_config)
7.         model.fit(X_train,y_train)
8.         compute_training_metrics(model, X_train, y_train)
8.         compute_testing_metrics(model, X_test, y_test)
9.         append new results to existing result file ('experiments_results.csv')

** From step(2), we know which experiments were already ran, so in (3) and (4) we skip model_config and
   experiment_config that were already run.


"""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

import os
import argparse

import sys
sys.path.append('..')

from utilities import configuration
from utilities import logger
from utilities import health_data

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

if __name__ == '__main__':
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
    # Simulation parameter used to compute pending experiments without running them           #
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation',
                        dest='simulation', 
                        required=True,
                        help='Wether to save to disk or not',
                        type=str,
                        choices=['True', 'False']
                        )
    args = parser.parse_args()
    simulation = args.simulation=='True'

    # ---------- ---------- ---------- ---------- ---------- ---------- #
    # Retrieving configuration (paths.yaml) and initializing logger     #
    # ---------- ---------- ---------- ---------- ---------- ---------- #
    config = configuration.get_config()
    logging = logger.init_logger(config['all_experiments_log'])
    if simulation:
        logging.debug('Running only a SIMULATIOn run.')
    logging.debug('Starting all experiments ...')


    # ---------- ---------- ---------- ---------- ---------- ---------- #
    # Retrieving model (models.json) and experiment configurations      #
    # (experiment_configuration.json)                                   #
    # ---------- ---------- ---------- ---------- ---------- ---------- #
    model_configurations = json.load(open(config['models_config'], encoding='utf-8'))
    experiment_configurations = json.load(open(config['experiments_config'], encoding='utf-8'))
    logging.debug(f'Using {len(model_configurations):4} different models.')
    logging.debug(f'Using {len(experiment_configurations):4} different configurations.')


    # ---------- ---------- ---------- #
    # COMPUTING TO-DO EXPERIMENTS      #
    # ---------- ---------- ---------- #
    to_do = []
    for model_id, model_config in model_configurations.items():
        if not ('skipping' in model_config and model_config['skipping']):
            configurations_to_run = [config_id for config_id in model_config['configuration_ids']]
            to_do += [(config_id, model_id) for config_id in configurations_to_run]
    to_do = set(to_do)
    logging.debug(f'Number of experiments found (in total): {len(to_do)}')
    
    already_ran = {}
    if os.path.isfile(config['experiment_results']):
        already_run_df = pd.read_csv(config['experiment_results'], sep=';')

        already_ran = [(config_id, model_id) for config_id, model_id in zip(already_run_df['config_id'], already_run_df['model_id'])]
        already_ran = set(already_ran)
    logging.debug(f'Number of experiments already ran found (in total): {len(already_ran)}')

    pending = to_do.difference(already_ran)
    pending_conf = set([config_id for config_id, model_id in pending])


    logging.debug(f'Pending experiments ({len(pending)})={pending}')
    logging.debug(f'Number of configuration founds: {len(experiment_configurations)} ({experiment_configurations.keys()})')
    if simulation:
        logging.debug('Ending simulation without running any experiment ...')
        sys.exit(0)


    # Filtering configurations not needed for this run:
    experiment_configurations = {configuration_id:configuration_dict 
                                 for configuration_id, configuration_dict in experiment_configurations.items()
                                 if configuration_id in pending_conf
                                 }
    logging.debug(f'Number of config pending to-do: {len(experiment_configurations)} ({experiment_configurations.keys()})')

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    # RUNNING ALL CONFIGURATIONS
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
    for configuration_id, configuration_dict in experiment_configurations.items():
        logging.debug(f'Running on configuration ID: {configuration_id}')
        params = configuration_dict
         
        # Computing training and testing matrices.
        X_train, y_train, X_test, y_test, columns = health_data.Admission.get_train_test_matrices(params)
        logging.debug(f'X_train.shape = {X_train.shape}')
        logging.debug(f'y_train.shape = {y_train.shape}')
        logging.debug(f'X_train.shape = {X_test.shape}')
        logging.debug(f'y_train.shape = {y_test.shape}')


        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        # RUNNING ALL MODELS WITH CALCULATED MATRICES
        # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
        for model_id, model_dict in model_configurations.items():
            logging.debug(f'Working on model ID= {model_id}')

            # Skipping if already ran.
            if os.path.isfile(config['experiment_results']):
                auxdf = pd.read_csv(config['experiment_results'], sep=';')
                model_ids = set([model_id for model_id in auxdf['model_id']])
                configuration_ids = set([model_id for model_id in auxdf['config_id']])
                
                already_run_pairs = set([(model_id, config_id) 
                                     for model_id,config_id in zip(model_ids,configuration_ids)])
                if (model_id,configuration_id) in already_run_pairs:
                    logging.debug(f'SKIPPING, configuration ({configuration_id}) and model ({model_id}) already found ...')
                    continue
                logging.debug('Results not found, running experiments ...')

            # Skipping model if "skipping"==True
            if 'skipping' in model_dict and model_dict['skipping']:
                logging.debug(f'SKIPPING model ({model_id}), as requested by configuration ...')
                continue
            
            # Skipping model if not assigned to current configuration
            if not configuration_id in model_dict['configuration_ids']:
                logging.debug(f'SKIPPING model {model_id} because it is not in the configuration_ids list.')
                continue
            else:
                logging.debug('Configuration found in configuration_ids list, preparing to run ...')

            # Creating model and fitting
            MODEL_SEED = 1270833263
            model_random_state=np.random.RandomState(MODEL_SEED)
            model = configuration.model_from_configuration(model_dict, random_state=model_random_state)
            logging.debug(f'Training model {str(model)}')


            # Using X_train, y_train to evaluate TRAINING METRICS (might be undersampled)
            # Using X_test, y_test to evaluate TESTING METRICS
            logging.debug('Fitting ...')
            model.fit(X_train, y_train)

            # EVALUATING ON TRAINING
            y_true = y_train
            y_pred = model.predict(X_train)
            y_score= model.predict_proba(X_train)[:,1]

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
                    roc_auc_score(y_true=y_true, y_score=y_score),
                    str(configuration_dict),
                    str(model_dict),
                    configuration_id,
                    model_id
                    ]
            
            # EVALUATION ON TESTING
            y_true = y_test
            y_pred = model.predict(X_test)
            y_score= model.predict_proba(X_test)[:,1] #model.predict_proba(X_train)[:,1]

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
                    roc_auc_score(y_true=y_true, y_score=y_score),
                    str(configuration_dict),
                    str(model_dict),
                    configuration_id,
                    model_id
                    ]
            m = np.vstack([vec1, vec2])
            new_df = pd.DataFrame(m, columns=columns)

            # SAVING
            if os.path.isfile(config['experiment_results']):
                old_df = pd.read_csv(config['experiment_results'], sep=';')
                logging.debug(f'Previous experiments found: {old_df.shape[0]} entries found')
                new_df = pd.concat([old_df,new_df])

            new_df.to_csv(config['experiment_results'], index=False, sep=';')
            logging.debug(f"Saving results in {config['experiment_results'].split('/')[-1]}")
            logging.debug(f'Saving {new_df.shape[0]} entries.')
