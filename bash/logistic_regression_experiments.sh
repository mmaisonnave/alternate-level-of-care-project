ENV=alc
REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
BASH_SCRIPTS_FOLDER=src
PYTHON_SCRIPT_NAME=logistic_regression.py


PYTHON_SCRIPT=$REPOSITORY_PATH/$BASH_SCRIPTS_FOLDER/$PYTHON_SCRIPT_NAME

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

# # Only numeric
# python $PYTHON_SCRIPT --numerical-features --fix-missings --class-balanced
# python $PYTHON_SCRIPT --numerical-features --fix-missings --class-balanced --normalize
# python $PYTHON_SCRIPT --numerical-features --fix-missings --class-balanced --normalize --fix-skew 
# python $PYTHON_SCRIPT --numerical-features --fix-missings --class-balanced --normalize --fix-skew --remove-outliers

# # Only categorical
# python $PYTHON_SCRIPT --categorical-features  --fix-missings --class-balanced

# # Only diagnosis
# python $PYTHON_SCRIPT --diagnosis-features    --fix-missings --class-balanced --use-idf

# # Only interventions
# python $PYTHON_SCRIPT --intervention-features --fix-missings --class-balanced --use-idf

# # ALL
# python $PYTHON_SCRIPT --categorical-features\
#                       --numerical-features\
#                       --diagnosis-features\
#                       --intervention-features\
#                       --fix-missings\
#                       --class-balanced\
#                       --normalize\
#                       --fix-skew\
#                       --use-idf\
#                       --remove-outliers

# # ALL Except IDF
# python $PYTHON_SCRIPT --categorical-features\
#                       --numerical-features\
#                       --diagnosis-features\
#                       --intervention-features\
#                       --fix-missings\
#                       --class-balanced\
#                       --normalize\
#                       --fix-skew\
#                       --remove-outliers

# # ALL Except fix skew
# python $PYTHON_SCRIPT --categorical-features\
#                       --numerical-features\
#                       --diagnosis-features\
#                       --intervention-features\
#                       --fix-missings\
#                       --class-balanced\
#                       --normalize\
#                       --use-idf\
#                       --remove-outliers

# # ALl except normalzie
# python $PYTHON_SCRIPT --categorical-features\
#                       --numerical-features\
#                       --diagnosis-features\
#                       --intervention-features\
#                       --fix-missings\
#                       --class-balanced\
#                       --fix-skew\
#                       --use-idf\
#                       --remove-outliers

# # ALL except remove outliers
# python $PYTHON_SCRIPT --categorical-features\
#                       --numerical-features\
#                       --diagnosis-features\
#                       --intervention-features\
#                       --fix-missings\
#                       --class-balanced\
#                       --normalize\
#                       --fix-skew\
#                       --use-idf\

# # All features, no transformation
# python $PYTHON_SCRIPT --categorical-features\
#                       --numerical-features\
#                       --diagnosis-features\
#                       --intervention-features\
#                       --fix-missings\
#                       --class-balanced

# All features, no transformation
python $PYTHON_SCRIPT --categorical-features\
                      --numerical-features\
                      --diagnosis-features\
                      --intervention-features\
                      --fix-missings\
                      --use-idf\
                      --class-balanced




