"""
This script creates:
 - A three crosstab table using categorical variables (output: config['latex_tables'])
 - A matix plot of the numerical variables            (output: three png at config['figures_folder'])
 - A categorical variable description table           (output: config['latex_tables'])
 - A numerical variable description table             (output: config['latex_tables'])

The input for all is the training and testing data:
 - training ,testing = health_data.Admission.get_training_testing_data(filtering=True)

 The script doesn't use the method "get_train_test_matrices", which means this 
 script is slightly outdated.

"""
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import sys
sys.path.append('..')
from utilities import logger
from utilities import configuration
from utilities import health_data
from utilities import formatting
import matplotlib.patches as mpatches
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import os

from scipy import stats


# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
def list_variable_description(training, testing, logging, config):
    field_names = ['Diagnosis Texts',
                   'Diagnosis Codes',
                   'Diagnosis Types',
                   'Intervention Codes',
                   'Intervention Texts',
                   ]
    data = {'Avg Length': [],
            'Min': [],
            'Max': [],
            'Unique': [],
            'Unique (in training)': [],
            'Mode': [],
            'Mode Count': [],
            'Less Frequent': [],
            'Less Frequent Count': [],
            r'\% Empty ': [],
            }

    for list_of_lists in [[admission.diagnosis.texts for admission in training + testing],
                          [admission.diagnosis.codes for admission in training + testing],
                          [admission.diagnosis.types for admission in training + testing],
                          [admission.intervention_code for admission in training + testing],
                          [admission.px_long_text for admission in training + testing],
                          ]:
        data['Avg Length'].append(np.average([len(list_) for list_ in list_of_lists]))
        data['Min'].append(np.min([len(list_) for list_ in list_of_lists]))
        data['Max'].append(np.max([len(list_) for list_ in list_of_lists]))
        data['Unique'].append(len(set([elem.lower() for list_ in list_of_lists for elem in list_])))
        data[r'\% Empty '].append(100*(len([list_ for list_ in list_of_lists if len(list_)==0])/len(list_of_lists)))

        all_elements = [elem.lower() for list_ in list_of_lists for elem in list_]
        data['Mode'].append(pd.Series(all_elements).mode()[0])
        data['Mode Count'].append(len([elem for elem in all_elements if elem==data['Mode'][-1]]))

        freq = defaultdict(int)
        for elem in [elem.lower() for list_ in list_of_lists for elem in list_]:
            freq[elem]+=1

        data['Less Frequent'].append(sorted(freq.items(), key=lambda key_value: key_value[1])[0][0])
        data['Less Frequent Count'].append(sorted(freq.items(), key=lambda key_value: key_value[1])[0][1])

    for list_of_lists in [[admission.diagnosis.texts for admission in training ],
                          [admission.diagnosis.codes for admission in training ],
                          [admission.diagnosis.types for admission in training ],
                          [admission.intervention_code for admission in training ],
                          [admission.px_long_text for admission in training ],
                          ]:
        data['Unique (in training)'].append(len(set([elem.lower() for list_ in list_of_lists for elem in list_])))

    df = pd.DataFrame(data, index=field_names)


    df['Mode'] = [elem if len(str(elem))<=25 else str(elem)[:22]+'...' for elem in df['Mode']]
    df['Less Frequent'] = [elem if len(str(elem))<=25 else str(elem)[:22]+'...' for elem in df['Less Frequent']]

    with open(config['latex_tables'], 'a') as writer:
        writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')
        writer.write('==========                                LIST VAR TABLE                     ==========\n')
        writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')
        writer.write('\n')
        writer.write(formatting.wrap_tabular(
                        df.to_latex(float_format=f"{{:0.3f}}".format).replace('_','\\_')
                        ))
        writer.write('\n')





# ########## ########## ########## ########## ########## ########## ########## ##########
# MISSING ENTRY CODE
# MISSING NURSE DISCHARGE UNIT
# MISSING READMISSION CODE
# ########## ########## ########## ########## ########## ########## ########## ##########
def crosstabs(training, testing, logging, config):

    categorical_df, _ = health_data.Admission.categorical_features(training)
    y = health_data.Admission.get_y(training)
    categorical_df['Target']=y

    range_pairs = [(0,4),(4,8),(8,12)]
    for begin,end in range_pairs:
        fields = categorical_df.columns[begin:end]

        df = pd.crosstab(categorical_df['Target'],
                    [categorical_df[field_name] for field_name in fields], 
                    colnames=[field_name for field_name in fields], 
                    )

        df.iloc[0,:] = 100*(df.iloc[0,:]/np.sum(df.iloc[0,:]))
        df.iloc[1,:] = 100*(df.iloc[1,:]/np.sum(df.iloc[1,:]))
        df['All']=['100\%','100\%',]


        with open(config['latex_tables'], 'a') as writer:
            writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')
            writer.write('==========                                   CROSSTABS                       ==========\n')
            writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')
            writer.write('\n')
            writer.write(formatting.wrap_tabular(
                         df.to_latex(float_format=f"{{:0.3f}}".format)
                         ))
            writer.write('\n')



# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
def numerical_variables_plots(training, testing, logging, config):

    for only_ALCs in [True, False]:
        params = {'fix_missing_in_testing': True,
                'alpha': 0.5,
                'remove_outliers': True,
                'only_ALCs': only_ALCs,
                'only_not_ALCs': False,
                'save_fig': True,
                }

        assert not (params['only_ALCs'] and params['only_not_ALCs'])


        if params['fix_missing_in_testing']:
            for admission in testing:
                admission.fix_missings(training)

        numerical_features = health_data.Admission.numerical_features(training+testing)
        y = health_data.Admission.get_y(training+testing)


        stds = np.std(numerical_features)
        mean = np.mean(numerical_features, axis=0)

        # Masks
        is_outlier=np.sum(numerical_features.values > (mean+4*stds).values, axis=1)>0
        is_alc = numerical_features['alc_days'] > 0


        positive_color='#ff7f0e' # ORANGE
        negative_color='#1f77b4' # BLUE

        if params['remove_outliers']:
            numerical_features = numerical_features[~is_outlier]
            y = y[~is_outlier]
            is_alc = is_alc[~is_outlier]

        if params['only_ALCs']:
            numerical_features = numerical_features[is_alc]
            y = y[is_alc]
        
        if params['only_not_ALCs']:
            numerical_features = numerical_features[~is_alc]
            y = y[~is_alc]
            numerical_features = numerical_features.drop(columns=['alc_days'])



        feature_count = numerical_features.shape[1]
        fig,ax = plt.subplots(feature_count,feature_count,figsize=(10,10))


        scatter_matrix(numerical_features[y==0], 
                        alpha = params['alpha'], 
                        ax=ax, 
                        diagonal = 'kde',)
        scatter_matrix(numerical_features[y==1], 
                        alpha = params['alpha'], 
                        ax=ax, 
                        color=positive_color,
                        diagonal = 'kde',)


        positive_patch = mpatches.Patch(color=positive_color, label='Readmitted')
        negative_patch = mpatches.Patch(color=negative_color, label='Not Readmitted')

        ax[1,1].legend(handles=[positive_patch, negative_patch], )


        for ix in range(feature_count):
            for ix2 in range(ix, feature_count):
                if ix!=ix2:
                    ax[ix][ix2].cla()


        ax[0,0].set_ylabel('Age')
        ax[1,0].set_ylabel('CMG')
        ax[2,0].set_ylabel('Case Weight')
        ax[3,0].set_ylabel('Acute Days')
        if not params['only_not_ALCs']:
            ax[4,0].set_ylabel('ALC Days')


        ax[-1,0].set_xlabel('Age')
        ax[-1,1].set_xlabel('CMG')
        ax[-1,2].set_xlabel('Case Weight')
        ax[-1,3].set_xlabel('Acute Days')
        if not params['only_not_ALCs']:
            ax[-1,4].set_xlabel('ALC Days')


        if params['save_fig']:
            if params['only_ALCs']:
                fig.savefig(os.path.join(config['figures_folder'],'scatter_matrix_numeric_ALCs.png'))
            elif params['only_not_ALCs']:
                fig.savefig(os.path.join(config['figures_folder'],'scatter_matrix_numeric_not_ALCs.png'))
            else:
                fig.savefig(os.path.join(config['figures_folder'],'scatter_matrix_numeric_all.png'))


# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
def categorical_variable_description(training, testing, logging, config):
    """
    This function creates a table with the description of the categorical variables (code, gender, postal code, etc)

    **OUTPUT**
    Resulting table is **appended** to the file:
    - latex_tables: /Users/marianomaisonnave/Repositories/alternate-level-of-care-project/results/tables.tex

    """
    fields = ['code',                    #1 
            'admit_date',                #2
            'discharge_date',            #3
            'gender',                    #4
            'is_central_zone',           #5
            'postal_code',               #6
            'institution_number',        #7
            'institution_to',            #8
            'institution_from',          #9
            'institution_type',         #10
            'admit_category',           #11
            'transfusion_given',        #12
            'comorbidity_level',        #13
            'main_pt_service',          #14
            'discharge_unit',           #15
            'entry_code',               #16
            'mrdx',                     #17
            'readmission_code',         #18
            ]
            
    
    fields_names = ['Encoded HCN',                 #1
                    'Admit Date',           #2
                    'Discharge Date',       #3
                    'Gender',               #4
                    'CZ Status',            #5
                    'Postal Code',          #6
                    'Institution Number',   #7
                    'Institution To',       #8
                    'Institution From',     #9
                    'Institution Type',     #10
                    'Admit Category',       #11
                    'Transfusion Given',    #12
                    'Comorbidity Level',    #13
                    'Main Pt Service',      #14
                    'Discharge Nurse Unit', #15
                    'Entry Code',           #16
                    'MRDx',                 #17
                    'Readmission Code',     #18
]

    data = {
            'Levels':[],
            'Levels (in training)':[],
            'Mode':[],
            'Mode Count':[],
            'Less Frequent':[],
            'Less Frequent Count':[],
            'Missings (\%)':[],
    }

    for field in fields:
        vector = [getattr(admission,field) for admission in training+testing]
        if field == 'gender':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.Gender.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.Gender.NONE  ,vector))
        elif field == 'admit_category':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.AdmitCategory.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.AdmitCategory.NONE  ,vector))
        elif field == 'readmission_code':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.ReadmissionCode.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.ReadmissionCode.NONE  ,vector))
        elif field == 'transfusion_given':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.TransfusionGiven.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.TransfusionGiven.NONE  ,vector))
        elif field == 'entry_code':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.EntryCode.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.EntryCode.NONE  ,vector))
        else:
            miss_count = len([elem for elem in vector if elem is None])
            vector = list(filter(lambda elem: not elem is None,vector))
        data['Missings (\%)'].append(100*(miss_count/len(training+testing)))
        data['Levels'].append(len(set(vector)))
        data['Mode'].append(pd.Series(vector).mode()[0])
        data['Mode Count'].append(len([elem for elem in vector if elem==data['Mode'][-1]]))
        # Computing Less freq
        freq = defaultdict(int)
        for elem in vector:
            freq[elem]+=1
        data['Less Frequent'].append((sorted(freq.items(), key=lambda key_value: key_value[1])[0][0]))
        data['Less Frequent Count'].append(sorted(freq.items(), key=lambda key_value: key_value[1])[0][1])


        # Count levels in training
        vector = [getattr(admission,field) for admission in training]
        if field == 'gender':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.Gender.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.Gender.NONE  ,vector))
        elif field == 'admit_category':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.AdmitCategory.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.AdmitCategory.NONE  ,vector))
        elif field == 'readmission_code':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.ReadmissionCode.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.ReadmissionCode.NONE  ,vector))
        elif field == 'transfusion_given':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.TransfusionGiven.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.TransfusionGiven.NONE  ,vector))
        elif field == 'entry_code':
            miss_count = len([elem for elem in vector if elem is None or elem==health_data.EntryCode.NONE])
            vector = list(filter(lambda elem: not elem is None and elem!=health_data.EntryCode.NONE  ,vector))
        else:
            miss_count = len([elem for elem in vector if elem is None])
            vector = list(filter(lambda elem: not elem is None,vector))
        data['Levels (in training)'].append(len(set(vector)))

    df = pd.DataFrame(data, index=fields_names)
    logging.debug(str(df))
    # with open(config['latex_tables'], 'a') as writer:
    #     writer.write(df.to_latex(float_format=f"{{:0.3f}}".format).replace('_','\\_') + '\n')


    df['Mode'] = [elem if len(str(elem))<=25 else str(elem)[:22]+'...' for elem in df['Mode']]
    df['Less Frequent'] = [elem if len(str(elem))<=25 else str(elem)[:22]+'...' for elem in df['Less Frequent']]
    with open(config['latex_tables'], 'a') as writer:
        writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')
        writer.write('==========                                 CAT VAR TABLE                     ==========\n')
        writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')
        writer.write('\n')
        writer.write(formatting.wrap_tabular(
                        df.to_latex(float_format=f"{{:0.3f}}".format).replace('_','\\_')
                        ))
        writer.write('\n')


# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
# ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- 
def numeric_variable_description(training, testing, logging, config):
    """
    This function creates a table with the description of the numerical variables (age, alc days, acute days, 
    cmg, case weight)

    **OUTPUT**
    Resulting table is **appended** to the file:
    - latex_tables: /Users/marianomaisonnave/Repositories/alternate-level-of-care-project/results/tables.tex

    """

    table_begin =  r'\begin{table}[]\centering\resizebox{\textwidth}{!}{'
    table_end = r'}\caption{Caption}\label{tab:my_label}\end{table}'    
    num_fields = [
            'age',
            'alc_days',
            'acute_days',
            'cmg',
            'case_weight',
    ]

    field_names = ['Age',
                'ALC Days',
                'Acute Days',
                'Case Mix Group',
                'Case Weight',
                ]

    data = {'Mean':[],
            'Std':[],
            'Min':[],
            'Q1':[],
            'Median':[],
            'Q3':[],
            'Max':[],
            'Kurtosis':[],
            'Skew':[],
            'Mode':[],
            'Mode Count':[],
            'Missing Count (\%)': [],
            }
    for field in num_fields:
        numbers = [getattr(admission, field) for admission in training+testing]
        missing_count = len([num for num in numbers if num is None or np.isnan(num)])
        data['Missing Count (\%)'].append(100*(missing_count/len(training+testing)))
        numbers = list(filter(lambda num: not num is None and not np.isnan(num), numbers))
        data['Mean'].append(np.average(numbers))
        data['Std'].append(np.std(numbers))
        data['Min'].append(np.min(numbers))
        data['Max'].append(np.max(numbers))
        data['Median'].append(np.median(numbers))
        data['Mode'].append(stats.mode(numbers)[0])
        data['Mode Count'].append(stats.mode(numbers)[1])
        data['Kurtosis'].append(stats.kurtosis(numbers))
        data['Skew'].append(stats.skew(numbers))
        data['Q1'].append(np.quantile(numbers,0.25))
        data['Q3'].append(np.quantile(numbers,0.75))
    df = pd.DataFrame(data, index=field_names)
    logging.debug(str(df))

    # with open(config['latex_tables'], 'a') as writer:
    #     writer.write(df.to_latex(float_format=f"{{:0.3f}}".format) + '\n')
    with open(config['latex_tables'], 'a') as writer:
        writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')
        writer.write('==========                                 NUM VAR TABLE                     ==========\n')
        writer.write('========== ========== ========== ========== ========== ========== ========== ==========\n')

        writer.write('\n')
        writer.write(formatting.wrap_tabular(
                        df.to_latex(float_format=f"{{:0.3f}}".format)
                        ))
        writer.write('\n')



if __name__=='__main__':
    params = {'fix_missing_in_testing': True}

    config = configuration.get_config()
    logging = logger.init_logger(config['system_log'])
    logging.debug('\n')
    logging.debug('Starting data_exploration.py script ...')

    training ,testing = health_data.Admission.get_training_testing_data(filtering=True)
    if params['fix_missing_in_testing']:
        for admission in testing:
            admission.fix_missings(training)


    categorical_variable_description(training, testing, logging, config)
    numeric_variable_description(training, testing, logging, config)
    # numerical_variables_plots(training, testing, logging, config)
    crosstabs(training, testing, logging, config)
    list_variable_description(training, testing, logging, config)