"""
This script runs four DT with four different group of features and one ensemble.
"""

import pandas as pd
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score

import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utilities import configuration
from utilities import health_data
from utilities import metrics
import json
from collections import Counter

def _capitalize_feature_name(feature_name:str)->str:
    if feature_name=='cmg':
        return 'CMG'
    elif feature_name=='case_weight':
        return 'RIW'
    else:
        aux = feature_name.replace('_', ' ').replace('-', ' ').strip()
        if aux.split(' ')=='':
            print('Error')
            print(aux)
        return ' '.join([word[0].upper()+word[1:] for word in aux.split(' ')])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-configuration',
                        dest='model_configuration',
                        required=True,
                        help='Choose one particular model configuration to run (configuration_name:str)',
                        type=str,
                        )
    parser.add_argument('--experiment-configuration',
                        dest='experiment_configuration',
                        required=True,
                        help="Choose one particular configuration to run (configuration_name:str)",
                        type=str,
                        )
    args = parser.parse_args()

    EXPERIMENT_CONFIG_NAME=args.experiment_configuration # 'configuration_31'
    MODEL_CONFIG_NAME=args.model_configuration # 'model_345'
    # EXPERIMENT_CONFIG_NAME='configuration_93'
    # MODEL_CONFIG_NAME='model_345'
    NC_FEATURE_LIST_GROUP_I=['acute_days', 
                             'cmg', 
                             'New Acute Patient',
                             'Unplanned Readmit', 
                             'urgent admission']
    
    NC_FEATURE_LIST_GROUP_II=['alc_days', 
                              'Day Surgery Entry', 
                              'Emergency Entry',
                              'General Surgery', 
                              'level 1 comorbidity', 
                              'transfusion given']
    
    NC_FEATURE_LIST_GROUP_III=['age',
                               'case_weight',
                               'Direct Entry',
                               'elective admission',
                               'Family Practice',
                               'female',
                               'General Medicine',
                               'is alc',
                               'is central zone',
                              'level 4 comorbidity', 
                               'male',
                               'OBS Delivered',
                               'Oral Surgery',
                               'Orthopaedic Surgery',
                               'Palliative Care',
                               'Panned Readmit',
                               'Psychiatry',
                               'Urology'
                               ]
    DI_FEATURE_LIST_GROUP_I=['j441',
                             'i500',
                             'z515',
                             'Z515',
                             'z38000',
                             '5md50aa',
                             ]

    NC_FEATURE_LIST_GROUP_I_TO_III_DI_GROUP_I = NC_FEATURE_LIST_GROUP_I + \
                                                NC_FEATURE_LIST_GROUP_II + \
                                                NC_FEATURE_LIST_GROUP_III + \
                                                DI_FEATURE_LIST_GROUP_I

    config = configuration.get_config()

    PARAMS = configuration.configuration_from_configuration_name(EXPERIMENT_CONFIG_NAME)
    print(f"use_idf={PARAMS['use_idf']}")
    X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(PARAMS)

    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')
    print(f'X_test.shape= {X_test.shape}')
    print(f'y_test.shape= {y_test.shape}')
    print()

    ensemble=[]

    for FEATURE_LIST,DESCRIPTION in zip([NC_FEATURE_LIST_GROUP_I, 
                                        NC_FEATURE_LIST_GROUP_II, 
                                        NC_FEATURE_LIST_GROUP_III, 
                                        DI_FEATURE_LIST_GROUP_I,
                                        NC_FEATURE_LIST_GROUP_I_TO_III_DI_GROUP_I,
                                        ],
                                        ['NC_group_I',
                                         'NC_group_II',
                                         'NC_group_III',
                                         'DI_group_I',
                                         'NC_group_I_TO_III_AND_DI_group_I'
                                         ]
                                        ):
        DESCRIPTION = f'{DESCRIPTION}_{MODEL_CONFIG_NAME}_{EXPERIMENT_CONFIG_NAME}'
        print(f'Filtering columns (len(FEATURE_LIST)={len(FEATURE_LIST)})')
        print(f'FEATURE_LIST={FEATURE_LIST}')
        selected_columns_ix = [ix for ix,feature_name in enumerate(feature_names) if feature_name in FEATURE_LIST]

        temp_X_train = X_train[:,selected_columns_ix]
        temp_X_test = X_test[:,selected_columns_ix]
        temp_feature_names = feature_names[selected_columns_ix]


        print(f'temp_X_train.shape={temp_X_train.shape}')
        print(f'y_train.shape={y_train.shape}')
        print(f'temp_X_test.shape= {temp_X_test.shape}')
        print(f'y_test.shape= {y_test.shape}')
        print()

        print(f'temp_feature_names={list(temp_feature_names)}')
        dt_model = configuration.model_from_configuration_name(MODEL_CONFIG_NAME)

        dt_model.fit(temp_X_train, y_train)
        ensemble.append(dt_model.predict(temp_X_test))

        print('Computing model results ...')
        df_metrics = pd.concat([metrics.get_metric_evaluations(dt_model, temp_X_train, y_train, MODEL_CONFIG_NAME, EXPERIMENT_CONFIG_NAME, description=f'DT train(FEATURE_LIST={FEATURE_LIST})'),
                        metrics.get_metric_evaluations(dt_model, temp_X_test, y_test, MODEL_CONFIG_NAME, EXPERIMENT_CONFIG_NAME, description=f'DT test(FEATURE_LIST={FEATURE_LIST})')])

        metrics_csv_name = config['custom_explainable_dt_metrics'].replace('.csv', 
                                                                        f'_{DESCRIPTION}.csv')
        df_metrics.to_csv(metrics_csv_name, index=False)

        fig, ax = plt.subplots(figsize=(150,50))
        tree.plot_tree(dt_model,
                    feature_names=list(map(_capitalize_feature_name, temp_feature_names)),
                    class_names=['NR', 'R'],
                    fontsize=11,
                    impurity=False,
                    label='none',
                    filled=True,
                    node_ids=False,
                    )

        output_file = config['custon_explainable_dt_figures'].replace('.jpg', f'_{DESCRIPTION}.jpg')

        fig.savefig(output_file, bbox_inches='tight')
    
    y_pred = (np.sum(ensemble,axis=0)==len(ensemble)).astype('int')
    y_true = y_test


    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results = {'Description': 'Test ensemble',
                        'Accuracy': accuracy_score(y_true, y_pred),
                        'Precision': precision_score(y_true, y_pred),
                        'Recal': recall_score(y_true, y_pred),
                        'F1-Score': f1_score(y_true, y_pred),
                        'AUC': 'N/A',
                        'TN': tn,
                        'TP': tp,
                        'FN': fn,
                        'FP': fp,
                        'Experiment config':EXPERIMENT_CONFIG_NAME,
                        'Model config': MODEL_CONFIG_NAME
                        }
    results = {key: [value] for key, value in results.items()}
    df_metrics = pd.DataFrame(results)

    metrics_csv_name = config['custom_explainable_dt_metrics'].replace('.csv', 
                                                                f'_ensemble_{MODEL_CONFIG_NAME}_{EXPERIMENT_CONFIG_NAME}.csv')
    df_metrics.to_csv(metrics_csv_name, index=False)





# # if __name__ == '__main__':
# def individual():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model-configuration',
#                         dest='model_configuration',
#                         required=True,
#                         help='Choose one particular model configuration to run (configuration_name:str)',
#                         type=str,
#                         )
#     parser.add_argument('--experiment-configuration',
#                         dest='experiment_configuration',
#                         required=True,
#                         help="Choose one particular configuration to run (configuration_name:str)",
#                         type=str,
#                         )
#     parser.add_argument('--experiment-description',
#                         dest='experiment_description',
#                         required=True,
#                         help="Experiment Description to save results",
#                         type=str,
#                         )
#     parser.add_argument('-f',
#                     '--feature-list',
#                     dest='feature_list',
#                     required=True,
#                     action="append",
#                     nargs='+',
#                     help="Features to include in decision tree",
#                     )
#     args = parser.parse_args()
#     print('args=')
#     print(args)
#     print('---')


#     EXPERIMENT_CONFIG_NAME=args.experiment_configuration # 'configuration_31'
#     MODEL_CONFIG_NAME=args.model_configuration # 'model_345'
#     CUSTOM_CONFIGURATION_DESCRIPTION=args.experiment_description # 'N_and_C_Group_I_and_II'
#     FEATURE_LIST=set([ ' '.join(feature) if isinstance(feature,list) else feature for feature in args.feature_list])



#     CUSTOM_CONFIGURATION_DESCRIPTION=CUSTOM_CONFIGURATION_DESCRIPTION+f"_{MODEL_CONFIG_NAME}_{EXPERIMENT_CONFIG_NAME}"
    
#     config = configuration.get_config()

#     PARAMS = configuration.configuration_from_configuration_name(EXPERIMENT_CONFIG_NAME)
#     print(f"use_idf={PARAMS['use_idf']}")
#     X_train, y_train, X_test, y_test, feature_names = health_data.Admission.get_train_test_matrices(PARAMS)

#     print(f'X_train.shape={X_train.shape}')
#     print(f'y_train.shape={y_train.shape}')
#     print(f'X_test.shape= {X_test.shape}')
#     print(f'y_test.shape= {y_test.shape}')
#     print()

#     print(f'Filtering columns (len(FEATURE_LIST)={len(FEATURE_LIST)})')
#     print(f'FEATURE_LIST={FEATURE_LIST}')
#     selected_columns_ix = [ix for ix,feature_name in enumerate(feature_names) if feature_name in FEATURE_LIST]

#     X_train = X_train[:,selected_columns_ix]
#     X_test = X_test[:,selected_columns_ix]
#     feature_names = feature_names[selected_columns_ix]

#     for ix,feature_name in enumerate(feature_names):
#         if not feature_name in ['acute_days', 'case_weight', ]:
#             print(f'Feature_name={feature_name}')
#             data_from_var_train = X_train[:,ix]
#             data_from_var_test = X_test[:,ix]
#             print('training stats:')
#             print(pd.Series(data_from_var_train.toarray()[:,0]).describe())
#             print()
#             print('COUNTER:')
#             print('Training...')
#             print(Counter(data_from_var_train.toarray()[:,0]))
#             print('Testing...')
#             print(Counter(data_from_var_test.toarray()[:,0]))
#             print()


#     print(f'X_train.shape={X_train.shape}')
#     print(f'y_train.shape={y_train.shape}')
#     print(f'X_test.shape= {X_test.shape}')
#     print(f'y_test.shape= {y_test.shape}')
#     print()

#     print(f'feature_names={list(feature_names)}')
#     dt_model = configuration.model_from_configuration_name(MODEL_CONFIG_NAME)

#     dt_model.fit(X_train, y_train)

#     print('Computing model results ...')
#     metrics_df = pd.concat([metrics.get_metric_evaluations(dt_model, X_train, y_train, MODEL_CONFIG_NAME, EXPERIMENT_CONFIG_NAME, description=f'DT train(FEATURE_LIST={FEATURE_LIST})'),
#                     metrics.get_metric_evaluations(dt_model, X_test, y_test, MODEL_CONFIG_NAME, EXPERIMENT_CONFIG_NAME, description=f'DT test(FEATURE_LIST={FEATURE_LIST})')])

#     metrics_csv_name = config['custom_explainable_dt_metrics'].replace('.csv', 
#                                                                        f'_{CUSTOM_CONFIGURATION_DESCRIPTION}.csv')
#     metrics_df.to_csv(metrics_csv_name, index=False)

#     fig, ax = plt.subplots(figsize=(150,50))
#     tree.plot_tree(dt_model,
#                 feature_names=list(map(_capitalize_feature_name, feature_names)),
#                 class_names=['NR', 'R'],
#                 fontsize=11,
#                 impurity=False,
#                 label='none',
#                 filled=True,
#                 node_ids=False,
#                 )

#     output_file = config['custon_explainable_dt_figures'].replace('.jpg', f'_{CUSTOM_CONFIGURATION_DESCRIPTION}.jpg')

#     fig.savefig(output_file, bbox_inches='tight')


# if __name__ == '__main__':
#     individual()