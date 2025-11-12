#!/bin/bash
#SBATCH --time=1:00:00

source /home/haltomj/miniconda3/etc/profile.d/conda.sh

# Define Categorical and Continuous features
columnsCat=CAT
columnsCont=CONT
columnsBiN=BIN

# Convert the arrays to comma-separated strings
columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")
columnBiN_string=$( echo "${columnsBiN[*]}")

# Load your Python environment
conda activate ML

# Function 
function run_analysis {
    local script_name=$1
    local dir_name=$2
    cp Metadata.Final.tsv "$dir_name"
    cd "$dir_name"
    sbatch --ntasks=1 --cpus-per-task=4 --time=120:00:00 --mail-user=haltomj@chop.edu --mail-type=ALL --wrap="python $script_name '$columnCat_string' '$columnCont_string' '$columnBiN_string'> out.$script_name.txt"
    
    cd ../../
}

# Testing different configurations
run_analysis "GB.GA.py" "Feature_Selection/Gradient_Boosting/GA"
run_analysis "GB.PTB.py" "Feature_Selection/Gradient_Boosting/PTB"
run_analysis "NN.GA.py" "Feature_Selection/NeuralNetworks"
run_analysis "NN.PTB.py" "Feature_Selection/NeuralNetworks"
run_analysis "RF.GA.py" "Feature_Selection/Random_Forest/GA"
run_analysis "RF.PTB.py" "Feature_Selection/Random_Forest/PTB"
run_analysis "LinearRegGA.py" "Feature_Selection/Linear-Logistic_Regression"
run_analysis "LogisticRegPTB.py" "Feature_Selection/Linear-Logistic_Regression"

# Wait for all SLURM jobs to finish
wait

















# #!/bin/bash
# #SBATCH --nodes=2
# #SBATCH --ntasks=8
# #SBATCH --cpus-per-task=4
# #SBATCH --time=120:00:00


# source /home/haltomj/miniconda3/etc/profile.d/conda.sh


# # Define Categorical and Continuous features
# columnsCat=CAT
# columnsCont=CONT
# columnsBiN=BIN

# # Convert the arrays to comma-separated strings
# columnCat_string=$( echo "${columnsCat[*]}")
# columnCont_string=$( echo "${columnsCont[*]}")
# columnBiN_string=$( echo "${columnsBiN[*]}")

# # Load your Python environment
# conda activate ML

# # Directory and script execution
# function run_analysis {
#     cp Metadata.Final.tsv $1
#     cd $1
#     srun --exclusive  --cpu-bind=none --time=120:00:00 python $2 "$columnCat_string" "$columnCont_string" "$columnBiN_string"> "out.$2.txt" &
#     cd ../../
# }

# run_analysis "Feature_Selection/Gradient_Boosting" "GB.GA.py"
# run_analysis "Feature_Selection/Gradient_Boosting" "GB.PTB.py"
# run_analysis "Feature_Selection/NeuralNetworks" "NN.GA.py"
# run_analysis "Feature_Selection/NeuralNetworks" "NN.PTB.py"
# run_analysis "Feature_Selection/Random_Forest" "RF.GA.py"
# run_analysis "Feature_Selection/Random_Forest" "RF.PTB.py"
# run_analysis "Feature_Selection/Linear-Logistic_Regression" "LinearRegGA.py"
# run_analysis "Feature_Selection/Linear-Logistic_Regression" "LogisticRegPTB.py"

# # Wait for all background processes to finish
# wait



