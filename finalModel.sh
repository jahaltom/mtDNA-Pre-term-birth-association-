#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 24:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL

source /home/haltomj/miniconda3/etc/profile.d/conda.sh



conda activate ML

#Input file
file="Metadata.M.tsv"
# Define Categorical features
columnsCat=('MainHap','DIABETES')
# Define Continuous  features
columnsCont=('BMI','PW_AGE')






# Convert the array to a comma-separated string

columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")



cp "$file" Final_Model/
cp WeibullFiltering.py Final_Model/
cp CombinePCA.py Final_Model/
cp outlierPCA.py Final_Model/
cd Final_Model/

# Call the Python script with the column string as an argument
python WeibullFiltering.py $file "$columnCat_string" "$columnCont_string" 




bcftools view --types snps -t ^26,24,23 -S IDs.txt --force-samples /scr1/users/haltomj/PTB/plink2.vcf   >  plink2.vcf

conda activate plink

mkdir PCA
#Run plink PCA
plink --vcf plink2.vcf --pca --double-id --out PCA/out

conda activate ML
python outlierPCA.py

conda activate plink
# Recompute PCA using filtered samples
plink --vcf plink2.vcf --keep keep_samples.txt --pca --double-id --out PCA/cleaned



conda activate ML
python  CombinePCA.py





#Final model 
python finalModel.py "$columnCat_string" "$columnCont_string" > detailedModelSummary.txt


