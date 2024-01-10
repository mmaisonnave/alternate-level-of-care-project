ENV=alc
REPOSITORY_PATH=/Users/marianomaisonnave/Repositories/alternate-level-of-care-project/
PYTHON_SCRIPTS_FOLDER=src
PYTHON_SCRIPT_NAME=build_gensim_models.py

PYTHON_SCRIPT=$REPOSITORY_PATH/$PYTHON_SCRIPTS_FOLDER/$PYTHON_SCRIPT_NAME


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


CUSTOM_COMMAND="$PYTHON_SCRIPT"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND

