# Unplanned Hospital Readmission Prediction Task

Our project addresses the task of unplanned hospital readmission prediction using machine learning on administrative healthcare data.

The structure of this repository is as follows:
- `bash` folder: stores the project bash scripts. Each bash script calls one or more Python scripts to perform specific tasks. For example, the 'creating_database.sh' script calls three Python scripts to do three distinct tasks: (1) Creating a central zone and not central zone CSV files, (2) unifying both CSV files and (3) transforming the data into the JSON format.
- `config` folder: contains all configuration scripts for all experiments. We define model hyperparameters and architectures in configuration files (for example, in the model.json file). Similarly, we define the preprocessing task applied to our data in each experiment in the experiment_configuration.json configuration file.
- `notebooks` folder: As part of this project, we use notebooks to prototype ideas. We move promising experiments from the notebook prototypes (notebook folder) to a Python script in the source folder (src folder).
- `src` (source) folder: Important parts of this project have their own Python script in this folder. We have Python scripts for running experiments, transforming the data, creating embedding models, and other tasks.
- `results` folder: Some Python scripts (from the src folder) store results in CSV files or latex tables. The Python scripts store those results in the results folder.
- `utilities` folder: As part of this project, we use an external repository to hold the additional functionality required for this project. For example, we have Python modules for (1) managing logging (logger.py), (2) handling the data (health_data.py), (3) handling encryption of the data (crypto.py), and other functionalities. This external repository is linked and stored in the `utilities` folder of this repository. 


