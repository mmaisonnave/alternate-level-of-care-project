#!/bin/bash
#SBATCH --time=2:00:0
#SBATCH --account=def-erajabi
#SBATCH --ntasks=1 
#SBATCH --nodes=1 
#SBATCH --mem=20GB 
#SBATCH --cpus-per-task=1 
#SBATCH --job-name=testing_env_difference
#SBATCH --output=/home/maiso/cbu/slurm/output/%x-%j.out

echo Running script at $(pwd)

REPOSITORY_PATH=$(cat ../config/paths.yaml | grep repository\_path: | grep -v ^# | sed 's/^repository\_path:\ //g')
PYTHON_SCRIPTS_FOLDER=src
PYTHON_SCRIPT_NAME=testing_env_difference.py

PYTHON_SCRIPT=$REPOSITORY_PATH/$PYTHON_SCRIPTS_FOLDER/$PYTHON_SCRIPT_NAME


echo $(date) - Running python file: $PYTHON_SCRIPT
echo $(date) - Using python: $(python --version)
echo $(date) - Which python: $(which python)

# IF PYTHON SCRIPT NOT FOUND, EXIT
if [ ! -f $PYTHON_SCRIPT ]; then
    echo "Python script not found ($PYTHON_SCRIPT_NAME)"
fi



CUSTOM_COMMAND="$PYTHON_SCRIPT"
echo [RUNNING] python $CUSTOM_COMMAND
python $CUSTOM_COMMAND

