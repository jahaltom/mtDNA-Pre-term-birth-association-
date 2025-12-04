#!/bin/bash
#SBATCH --time=1:00:00

source /home/haltomj/miniconda3/etc/profile.d/conda.sh

# Define Categorical and Continuous features
columnsCat=CAT
columnsCont=CONT



# Convert the arrays to comma-separated strings
columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")


# Load your Python environment
conda activate ML

# Function 
function run_analysis {
    local script_name=$1
    local dir_name=$2
    cp Final_Model/Metadata.Final.tsv "$dir_name"
    cd "$dir_name"
    sbatch --ntasks=1 --cpus-per-task=4 --time=20:00:00  --wrap="python $script_name '$columnCat_string' '$columnCont_string' > out.$script_name.txt"
    
    cd ../../../../
}

# Testing different configurations
run_analysis "GB.GA.py" "Final_Model/Feature_Selection/Gradient_Boosting/GA"
run_analysis "GB.PTB.py" "Final_Model/Feature_Selection/Gradient_Boosting/PTB"
run_analysis "NN.GA.py" "Final_Model/Feature_Selection/NeuralNetworks/GA"
run_analysis "NN.PTB.py" "Final_Model/Feature_Selection/NeuralNetworks/PTB"
run_analysis "RF.GA.py" "Final_Model/Feature_Selection/Random_Forest/GA"
run_analysis "RF.PTB.py" "Final_Model/Feature_Selection/Random_Forest/PTB"
run_analysis "LinearRegGA.py" "Final_Model/Feature_Selection/Linear-Logistic_Regression/GA"
run_analysis "LogisticRegPTB.py" "Final_Model/Feature_Selection/Linear-Logistic_Regression/PTB"

# Wait for all SLURM jobs to finish
wait




