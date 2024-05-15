#!/bin/bash
#SBATCH --time=0-00:45:0
#SBATCH --account=def-erajabi
#SBATCH --ntasks=1 
#SBATCH --nodes=1 
#SBATCH --mem=16GB 
#SBATCH --cpus-per-task=1
#SBATCH --job-name=custom_explainable_tree
#SBATCH --output=/home/maiso/cbu/slurm/output/%x-%j.out

echo Running script at $(pwd)

ENV=alc
REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
PYTHON_SCRIPTS_FOLDER=src
PYTHON_SCRIPT_NAME=custom_explainable_tree.py

PYTHON_SCRIPT=$REPOSITORY_PATH/$PYTHON_SCRIPTS_FOLDER/$PYTHON_SCRIPT_NAME


echo $(date) - Running python file: $PYTHON_SCRIPT
echo $(date) - Using python: $(python --version)
echo $(date) - Which python: $(which python)

# IF PYTHON SCRIPT NOT FOUND, EXIT
if [ ! -f $PYTHON_SCRIPT ]; then
    echo "Python script not found ($PYTHON_SCRIPT_NAME)"
fi


# NO VIRTUAL ENV, EXIT
if [ -z "${VIRTUAL_ENV}" ];
then 
    echo Scripts expect conda environment set "$ENV"
    exit 1
fi

# DIFFERENT VRITUAL ENV FROM EXPECTED, EXIT
if [ $(echo $VIRTUAL_ENV | sed 's/.*\/\(.*\)$/\1/g' | sed 's/\n\n*//g')  != $ENV ]; 
then 
    echo Scripts expect conda environment set "$ENV"
    exit 1
fi



MODEL_CONFIGURATION=model_345
EXPERIMENT_CONFIGURATION=configuration_93 #  #(N)+(C)+(I)+ Combined D (CD)


# # EXPERIMENT #1 --- GROUP I AND II (NUMERIC AND CATEGORICAL)
# NC_GROUP_I_AND_II_DESCRIPTION=N_and_C_Group_I_and_II
# NC_GROUP_I_AND_II="-f urgent admission \
# -f New Acute Patient \
# -f acute_days \
# -f case_weight \
# -f Unplanned Readmit \
# -f Emergency Entry"

# # EXPERIMENT #2 --- GROUP I (DIAGNOSIS AND INTERVENTION)
# DI_GROUP_I_DESCRIPTION=D_and_I_Group_I
# DI_GROUP_I="-f j441 \
# -f z38000 \
# -f 5md50aa \
# -f z515 \
# -f i500 \
# -f z751"

# # EXPERIMENT #3 --- Num and Cat Group I & II ; D and I Group I
# DI_GROUP_I_AND_NC_GROUP_I_AND_II_DESCRIPTION=NC_group_I_and_II_DI_group_I
# DI_GROUP_I_AND_NC_GROUP_I_AND_II=$(echo $NC_GROUP_I_AND_II $DI_GROUP_I)



# # EXPERIMENT #4 --- Num and Cat Group I to II
# NC_GROUP_I_TO_III_DESCRIPTION=NC_Group_I_to_III
# NC_GROUP_III="-f alc_days \
# -f cmg \
# -f age \
# -f male \
# -f female \
# -f Direct Entry \
# -f Orthopaedic Surgery \
# -f transfusion given \
# -f Panned Readmit \
# -f General Surgery \
# -f elective admission \
# -f level 4 comorbidity"
# NC_GROUP_I_TO_III=$(echo $NC_GROUP_I_AND_II $NC_GROUP_III)


# # EXPERIMENT #5 --- Num and Cat Group I to III and DI group I
# NC_GROUP_I_TO_III_AND_DI_GROUP_I_DESCRIPTION=N_and_C_Group_I_to_III_and_DI
# NC_GROUP_I_TO_III_AND_DI_GROUP_I=$(echo $NC_GROUP_I_TO_III $DI_GROUP_I)


NC_GROUP_I_TO_III="-f acute_days \
-f cmg \
-f New Acute Patient \
-f Unplanned Readmit \
-f urgent admission \
-f alc_days \
-f Day Surgery Entry \
-f Emergency Entry
-f General Surgery \
-f level 1 comorbidity \
-f transfusion given \
-f age \
-f case_weight \
-f Direct Entry \
-f elective admission \
-f Family Practice \
-f female \
-f General Medicine \
-f is alc \
-f is central zone \
-f level 4 comorbidity
-f male \
-f OBS Delivered \
-f Oral Surgery \
-f Orthopaedic Surgery \
-f Palliative Care \
-f Panned Readmit \
-f Psychiatry \
-f Urology"

DC_GROUP_I="-f j441 \
# -f i500 \
# -f z515 \
# -f Z515 \
# -f z38000 \
# -f 5md50aa"

NC_GROUP_I_TO_III_AND_DI_GROUP_I=$(echo $NC_GROUP_I_TO_III $DC_GROUP_I)
NC_GROUP_I_TO_III_AND_DI_GROUP_I_DESCRIPTION=N_and_C_Group_I_to_III_and_DI

CUSTOM_COMMAND="$PYTHON_SCRIPT --experiment-configuration=$EXPERIMENT_CONFIGURATION \
--experiment-description=$NC_GROUP_I_TO_III_AND_DI_GROUP_I_DESCRIPTION \
--model-configuration=$MODEL_CONFIGURATION \
$NC_GROUP_I_TO_III_AND_DI_GROUP_I"


echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND

