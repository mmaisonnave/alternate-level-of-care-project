import numpy as np
import pandas as pd

import os

import sys

sys.path.append('..')

import matplotlib.pyplot as plt

from utilities import health_data
from utilities import configuration
from utilities import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def create_train_test_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    EXPERIMENT_CONFIGURATION='configuration_93'
    FEATURES=[
              
              # Group I
              'acute_days',
              'cmg',
              'New Acute Patient',
              'Unplanned Readmit',

              # Group II
              'alc_days',
              'Day Surgery Entry',
              'Emergency Entry',
              'General Surgery',
              'level 1 comorbidity',
              'transfusion given',
              'urgent admission',

              # Group III
              'age',
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
              'Urology',

              # Group I D+i
              'j441',
              'i500',
              'z515',
              'z38000',
              '5md50aa',

              # Yet another for individualize prediction (entry code)
              'Clinic Entry'
              ]
    print(f'Using EXPERIMENT_CONFIGURATION={EXPERIMENT_CONFIGURATION}, obtaining params ...')
    params = configuration.configuration_from_configuration_name(EXPERIMENT_CONFIGURATION)

    print('Retrieving train and test matrices ...')
    X_train, y_train, X_test, y_test, columns =  health_data.Admission.get_train_test_matrices(params)
    
    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')

    print(f'X_test.shape={X_test.shape}')
    print(f'y_test.shape={y_test.shape}')


    print()
    print(f'Removing irrelevant columns, only keeping the following features: {str(FEATURES)}')
    columns = list(columns)
    indexes = [ix for ix in [columns.index(feature) for feature in FEATURES]]

    
    X_train = X_train[:,indexes].toarray()
    X_test = X_test[:,indexes].toarray()

    print(f'X_train.shape={X_train.shape}')
    print(f'y_train.shape={y_train.shape}')

    print(f'X_test.shape={X_test.shape}')
    print(f'y_test.shape={y_test.shape}')

    print('Building DataFrames ...')

    train_df = pd.DataFrame(np.hstack([X_train, y_train.reshape(-1,1)]),
                            columns=FEATURES+['target'])
    
    test_df = pd.DataFrame(np.hstack([X_test, y_test.reshape(-1,1)]),
                           columns=FEATURES+['target'])
    
    return train_df, test_df
    # train_df.to_csv('train_df.csv', index=None)
    # test_df.to_csv('test_df.csv', index=None)




def individualize_prediction_1(train_df: pd.DataFrame, test_df: pd.DataFrame):
    MAX_DEPTH=3
    print(f'Using MAX_DEPTH={MAX_DEPTH}')

    config = configuration.get_config()
    # train_df = pd.read_csv('train_df.csv',)
    # test_df = pd.read_csv('test_df.csv', )

    print(f'train_df.shape={train_df.shape}')
    print(f'test_df.shape={test_df.shape}')

    print('Dropping Clinic Entry feature (as it is not part of Group I to III of N+C)')
    train_df.drop(columns=['Clinic Entry'])
    test_df.drop(columns=['Clinic Entry'])



    print('Preparting masks (non-urgent patients, urgent patients with unplanned readmit=0, etc...)')
    training_masks = [(train_df['urgent admission']==0),
                      (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==0),
                      (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==1)& (train_df['Day Surgery Entry']==0),
                      (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==1)& (train_df['Day Surgery Entry']==1),
                      ]
    
    testing_mask = [(test_df['urgent admission']==0),
                    (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==0),
                    (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==1)& (test_df['Day Surgery Entry']==0),
                    (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==1)& (test_df['Day Surgery Entry']==1),
                    ]
    
    experiment_names = ['urgent=0', 
                        'Urgent=1 -> Unplanned Readmit=0',
                        'Urgent=1 -> Unplanned Readmit=1 -> day surgery=0',
                        'Urgent=1 -> Unplanned Readmit=1 -> day surgery=1',
                        ]
    
    all_yhats = []
    all_ytrues = []
    results=None
    for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
        print(f'Experiment name={experiment_name}')
        train_data = train_df[train_mask]
        test_data = test_df[test_mask]


        print(f'train_data.shape={train_data.shape}')
        print(f'test_data.shape= {test_data.shape}')

        dt = DecisionTreeClassifier(max_depth=MAX_DEPTH, class_weight='balanced')

        dt.fit(train_data.drop(columns=['target']).values,
               train_data['target'].values,
               )

        print('Computing model results ...')

        yhat = dt.predict(test_data.drop(columns=['target']).values,)
        ytrue = test_data['target'].values

        all_yhats.append(yhat)
        all_ytrues.append(ytrue)
        print(f'yhat.shape={yhat.shape}')
        print(f'ytrue.shape={ytrue.shape}')
        if results is None:
            results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
                                                                        yhat, 
                                                                        description=experiment_name)
            print('results')
            print(results)
        else:
            tmp = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
                                                                     yhat, 
                                                                     description=experiment_name)
            results = pd.concat([results, tmp])
        
        
        fig, ax = plt.subplots(figsize=(25,15))
        tree.plot_tree(dt,
                    feature_names=list(train_data.columns),
                    class_names=['NR', 'R'],
                    fontsize=11,
                    impurity=False,
                    label='none',
                    filled=True,
                    node_ids=False,
                    )
        
        base_name = config['guidelines_dts_figures']
        figure_filename = f"{base_name}_{experiment_name.lower().replace(' ','_')}_dt.jpg"
        fig.savefig(figure_filename, bbox_inches='tight')

        
        print(pd.DataFrame(results))
        print()

    tmp = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
                                                             np.hstack(all_yhats),
                                                            description='both')
    results = pd.concat([results, tmp])

    performance_results_filename = config['guideline_dts_performances']

    pd.DataFrame(results).to_csv(performance_results_filename,index=None)
    print(pd.DataFrame(tmp))
    print()



def main():
    train_df, test_df = create_train_test_df()
    individualize_prediction_1(train_df, test_df)
    print('-'*80)


if __name__ == '__main__':
    main()
    # create_csv_files()

    # individualize_prediction_admit_category()
    # print('-'*80)

    # individualize_prediction_entry_code()
    # print('-'*80)

    # individualize_prediction_emergency_or_urgent()
    # print('-'*80)

    # individualize_prediction_urgent()
    # print('-'*80)

    # individualize_prediction_emergency_and_urgent()
    # print('-'*80)


    # individualize_prediction_unplanned_new_acute()
    # print('-'*80)


    # individualize_prediction_2()
    # print('-'*80)
    # individualize_prediction_3()
    # print('-'*80)








# def individualize_prediction_admit_category():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')



#     training_masks = [train_df['urgent admission']==1, 
#                       train_df['elective admission']==1,
#                       (train_df['elective admission']==0) & (train_df['urgent admission']==0),
#                       ]
    
#     testing_mask = [test_df['urgent admission']==1, 
#                     test_df['elective admission']==1,
#                     (test_df['elective admission']==0) & (test_df['urgent admission']==0),
#                     ]
    
#     experiment_names = ['Urgent Patients', 'Elective Patients', 'Newborn patients']
#     all_yhats = []
#     all_ytrues = []
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
        
#         fig, ax = plt.subplots(figsize=(25,15))
#         tree.plot_tree(dt,
#                     feature_names=list(train_data.columns),
#                     class_names=['NR', 'R'],
#                     fontsize=11,
#                     impurity=False,
#                     label='none',
#                     filled=True,
#                     node_ids=False,
#                     )
#         fig.savefig(f"{experiment_name.lower().replace(' ','_')}_dt.jpg", bbox_inches='tight')

#         print(pd.DataFrame(results))
#         print()

#     results = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
#                                                                  np.hstack(all_yhats),
#                                                                 description='both')
#     print(pd.DataFrame(results))
#     print()

#     # pd.DataFrame(results).to_csv('results_from_guidelines_dt.csv')



# def individualize_prediction_entry_code():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     # Clinic entry is not part of Group I to III of the consensus analysis for N+C
#     train_df.drop(columns=['Clinic Entry'])
#     test_df.drop(columns=['Clinic Entry'])

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')



#     training_masks = [train_df['Clinic Entry']==1, 
#                       train_df['Direct Entry']==1,
#                       train_df['Emergency Entry']==1,
#                       train_df['Day Surgery Entry']==1,

#                       (train_df['Clinic Entry']==0) & 
#                       (train_df['Direct Entry']==0) & 
#                       (train_df['Emergency Entry']==0) & 
#                       (train_df['Day Surgery Entry']==0),
#                       ]
    
#     testing_mask = [test_df['Clinic Entry']==1, 
#                     test_df['Direct Entry']==1,
#                     test_df['Emergency Entry']==1,
#                     test_df['Day Surgery Entry']==1,
                    
#                     (test_df['Clinic Entry']==0) & 
#                     (test_df['Direct Entry']==0) & 
#                     (test_df['Emergency Entry']==0) & 
#                     (test_df['Day Surgery Entry']==0),
#                     ]
    
#     experiment_names = ['Clinic Entry', 'Direct Entry', 'Emergency Entry', 'Day Surgery Entry', 'Newborn']
#     all_yhats = []
#     all_ytrues = []
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
#         print(pd.DataFrame(results))
#         print()

#     results = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
#                                                                  np.hstack(all_yhats),
#                                                                 description='both')
#     print(pd.DataFrame(results))
#     print()






# def individualize_prediction_emergency_or_urgent():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')

#     train_df.drop(columns=['Clinic Entry'])
#     test_df.drop(columns=['Clinic Entry'])



#     training_masks = [(train_df['urgent admission']==1) & (train_df['Emergency Entry']==1),
#                       (train_df['urgent admission']==0) & (train_df['Emergency Entry']==1),
#                       (train_df['urgent admission']==1) & (train_df['Emergency Entry']==0),
#                       (train_df['urgent admission']==0) & (train_df['Emergency Entry']==0)
#                       ]
    
#     testing_mask = [(test_df['urgent admission']==1) & (test_df['Emergency Entry']==1),
#                     (test_df['urgent admission']==0) & (test_df['Emergency Entry']==1),
#                     (test_df['urgent admission']==1) & (test_df['Emergency Entry']==0),
#                     (test_df['urgent admission']==0) & (test_df['Emergency Entry']==0)
#                     ]
    
#     experiment_names = ['Urgent and Emergency Patients', 
#                         'Only Emergency Patients', 
#                         'Only Urgent Patients',
#                         'Not Urgent nor Emergency Patients',
#                         ]
#     all_yhats = []
#     all_ytrues = []
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
        
#         fig, ax = plt.subplots(figsize=(25,15))
#         tree.plot_tree(dt,
#                     feature_names=list(train_data.columns),
#                     class_names=['NR', 'R'],
#                     fontsize=11,
#                     impurity=False,
#                     label='none',
#                     filled=True,
#                     node_ids=False,
#                     )
#         fig.savefig(f"{experiment_name.lower().replace(' ','_')}_dt.jpg", bbox_inches='tight')

#         print(pd.DataFrame(results))
#         print()

#     results = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
#                                                                  np.hstack(all_yhats),
#                                                                 description='both')
#     print(pd.DataFrame(results))
#     print()

#     # pd.DataFrame(results).to_csv('results_from_guidelines_dt.csv')




# def individualize_prediction_urgent():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')

#     train_df.drop(columns=['Clinic Entry'])
#     test_df.drop(columns=['Clinic Entry'])



#     training_masks = [train_df['urgent admission']==1,
#                       train_df['urgent admission']==0
#                       ]
    
#     testing_mask = [test_df['urgent admission']==1,
#                     test_df['urgent admission']==0
#                     ]
    
#     experiment_names = ['Urgent Patients', 
#                         'Not Urgent Patients',
#                         ]
#     all_yhats = []
#     all_ytrues = []
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
        
#         fig, ax = plt.subplots(figsize=(25,15))
#         tree.plot_tree(dt,
#                     feature_names=list(train_data.columns),
#                     class_names=['NR', 'R'],
#                     fontsize=11,
#                     impurity=False,
#                     label='none',
#                     filled=True,
#                     node_ids=False,
#                     )
#         fig.savefig(f"{experiment_name.lower().replace(' ','_')}_dt.jpg", bbox_inches='tight')

#         print(pd.DataFrame(results))
#         print()

#     results = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
#                                                                  np.hstack(all_yhats),
#                                                                 description='both')
#     print(pd.DataFrame(results))
#     print()

#     # pd.DataFrame(results).to_csv('results_from_guidelines_dt.csv')


# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
# Unplanned Readmit vs New Acute Patient #
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #


    # pd.DataFrame(results).to_csv('results_from_guidelines_dt.csv')



# # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
# # Unplanned Readmit vs New Acute Patient #
# # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
# def individualize_prediction_2():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')

#     train_df.drop(columns=['Clinic Entry'])
#     test_df.drop(columns=['Clinic Entry'])



#     training_masks = [(train_df['urgent admission']==0),
#                       (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==0),
#                       (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==1),
#                       ]
    
#     testing_mask = [(test_df['urgent admission']==0),
#                     (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==0),
#                     (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==1),
#                     ]
    
#     experiment_names = ['Urgent=0', 
#                         'Urgent=1 -> Unplanned=0',
#                         'Urgent=1 -> Unplanned=1',
#                         ]
#     all_yhats = []
#     all_ytrues = []
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
        
#         fig, ax = plt.subplots(figsize=(25,15))
#         tree.plot_tree(dt,
#                     feature_names=list(train_data.columns),
#                     class_names=['NR', 'R'],
#                     fontsize=11,
#                     impurity=False,
#                     label='none',
#                     filled=True,
#                     node_ids=False,
#                     )
#         fig.savefig(f"{experiment_name.lower().replace(' ','_')}_dt.jpg", bbox_inches='tight')

#         print(pd.DataFrame(results))
#         print()

#     results = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
#                                                                  np.hstack(all_yhats),
#                                                                 description='both')
#     print(pd.DataFrame(results))
#     print()



# def individualize_prediction_3():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')

#     train_df.drop(columns=['Clinic Entry'])
#     test_df.drop(columns=['Clinic Entry'])



#     training_masks = [(train_df['urgent admission']==0),
#                       (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==0) & (train_df['New Acute Patient']==0),
#                       (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==0) & (train_df['New Acute Patient']==1),
#                       (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==1) & (train_df['Day Surgery Entry']==0),
#                       (train_df['urgent admission']==1) & (train_df['Unplanned Readmit']==1) & (train_df['Day Surgery Entry']==1),
#                       ]
    
#     testing_mask = [(test_df['urgent admission']==0),
#                     (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==0) & (test_df['New Acute Patient']==0),
#                     (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==0) & (test_df['New Acute Patient']==1),
#                     (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==1) & (test_df['Day Surgery Entry']==0),
#                     (test_df['urgent admission']==1) & (test_df['Unplanned Readmit']==1) & (test_df['Day Surgery Entry']==1),
#                     ]
    
#     experiment_names = ['Urgent=0',
#                         'Urgent=1 -> Unplanned=0 -> New Acute=0',
#                         'Urgent=1 -> Unplanned=0 -> New Acute=1',
#                         'Urgent=1 -> Unplanned=1 -> Day Surgery=0',
#                         'Urgent=1 -> Unplanned=1 -> DaySurgery=1',
#                         ]
#     all_yhats = []
#     all_ytrues = []
#     results = None
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         if results is None:
#             results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                         yhat, 
#                                                                         description=experiment_name)
#         else:
#             tmp = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
#             results = {key:value+[tmp[key]] for  key,value in results.items()}
        
#         fig, ax = plt.subplots(figsize=(25,15))
#         tree.plot_tree(dt,
#                     feature_names=list(train_data.columns),
#                     class_names=['NR', 'R'],
#                     fontsize=11,
#                     impurity=False,
#                     label='none',
#                     filled=True,
#                     node_ids=False,
#                     )
#         fig.savefig(f"{experiment_name.lower().replace(' ','_')}_dt.jpg", bbox_inches='tight')

#         print(pd.DataFrame(results))
#         print()

    
#     tmp = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                             yhat, 
#                                                             description='both')
#     results = {key:value+[tmp[key]] for  key,value in results.items()}

#     pd.DataFrame(results).to_csv('results.csv' ,index=None)
#     print('Done!')
#     print()

# def individualize_prediction_emergency_and_urgent():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')

#     train_df.drop(columns=['Clinic Entry'])
#     test_df.drop(columns=['Clinic Entry'])



#     training_masks = [(train_df['Unplanned Readmit']==1) & (train_df['Emergency Entry']==1),
#                       ~((train_df['urgent admission']==1) & (train_df['Emergency Entry']==1)),
#                       ]
    
#     testing_mask = [(test_df['urgent admission']==1) & (test_df['Emergency Entry']==1),
#                     ~((test_df['urgent admission']==1) & (test_df['Emergency Entry']==1)),
#                     ]
    
#     experiment_names = ['Urgent and Emergency Patients', 
#                         'All other Patients',
#                         ]
#     all_yhats = []
#     all_ytrues = []
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
        
#         fig, ax = plt.subplots(figsize=(25,15))
#         tree.plot_tree(dt,
#                     feature_names=list(train_data.columns),
#                     class_names=['NR', 'R'],
#                     fontsize=11,
#                     impurity=False,
#                     label='none',
#                     filled=True,
#                     node_ids=False,
#                     )
#         fig.savefig(f"{experiment_name.lower().replace(' ','_')}_dt.jpg", bbox_inches='tight')

#         print(pd.DataFrame(results))
#         print()

#     results = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
#                                                                  np.hstack(all_yhats),
#                                                                 description='both')
#     print(pd.DataFrame(results))
#     print()

#     # pd.DataFrame(results).to_csv('results_from_guidelines_dt.csv')




# # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
# # Urgent=0; ugent=1 & unpanned=0; ugent=1 & unpanned=1; 
# # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- #
# def individualize_prediction_unplanned_new_acute():
#     train_df = pd.read_csv('train_df.csv',)
#     test_df = pd.read_csv('test_df.csv', )

#     print(f'train_df.shape={train_df.shape}')
#     print(f'test_df.shape={test_df.shape}')

#     train_df.drop(columns=['Clinic Entry'])
#     test_df.drop(columns=['Clinic Entry'])



#     training_masks = [(train_df['New Acute Patient']==1),
#                       (train_df['Unplanned Readmit']==1),
#                       (train_df['New Acute Patient']==0) & (train_df['Unplanned Readmit']==0) ,
#                       ]
    
#     testing_mask = [(test_df['New Acute Patient']==1),
#                     (test_df['Unplanned Readmit']==1),
#                     (test_df['New Acute Patient']==0) & (test_df['Unplanned Readmit']==0) ,
#                       ]
    
#     experiment_names = ['New Acute Patient', 
#                         'Unplanned Readmit',
#                         'All other Patients',
#                         ]
#     all_yhats = []
#     all_ytrues = []
#     for train_mask, test_mask, experiment_name in zip(training_masks, testing_mask, experiment_names):
#         print(f'Experiment name={experiment_name}')
#         train_data = train_df[train_mask]
#         test_data = test_df[test_mask]

#         print(f'train_data.shape={train_data.shape}')
#         print(f'test_data.shape= {test_data.shape}')

#         dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced')

#         dt.fit(train_data.drop(columns=['target']).values,
#                train_data['target'].values,
#                )

#         print('Computing model results ...')

#         yhat = dt.predict(test_data.drop(columns=['target']).values,)
#         ytrue = test_data['target'].values

#         all_yhats.append(yhat)
#         all_ytrues.append(ytrue)
#         print(f'yhat.shape={yhat.shape}')
#         print(f'ytrue.shape={ytrue.shape}')

#         results = metrics.get_metric_evaluations_from_yhat_and_ypred(ytrue, 
#                                                                      yhat, 
#                                                                      description=experiment_name)
        
#         fig, ax = plt.subplots(figsize=(25,15))
#         tree.plot_tree(dt,
#                     feature_names=list(train_data.columns),
#                     class_names=['NR', 'R'],
#                     fontsize=11,
#                     impurity=False,
#                     label='none',
#                     filled=True,
#                     node_ids=False,
#                     )
#         fig.savefig(f"{experiment_name.lower().replace(' ','_')}_dt.jpg", bbox_inches='tight')

#         print(pd.DataFrame(results))
#         print()

#     results = metrics.get_metric_evaluations_from_yhat_and_ypred(np.hstack(all_ytrues),
#                                                                  np.hstack(all_yhats),
#                                                                 description='both')
#     print(pd.DataFrame(results))
#     print()

#     # pd.DataFrame(results).to_csv('results_from_guidelines_dt.csv')


