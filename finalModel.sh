#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node 24
#SBATCH -t 6:00:00
#SBATCH --mail-user=haltomj@chop.edu
#SBATCH --mail-type=ALL

source /home/haltomj/miniconda3/etc/profile.d/conda.sh


conda activate ML


#Input file
file="Metadata.M.tsv"

# Define Categorical features
columnsCat=CAT
# Define Continuous  features
columnsCont=CONT
target=TARGET


# Convert the array to a comma-separated string
columnCat_string=$( echo "${columnsCat[*]}")
columnCont_string=$( echo "${columnsCont[*]}")





cp -r Final_Model Final_Model_IDENT
cp $file Final_Model_IDENT
cd Final_Model_IDENT

#Removes samples where gestational age "GAGEBRTH" or  PTB (0 or 1) is na. Also removes samples with missing data in any of the input columns.
python ../scripts/removeMissingData.py $file "$columnCat_string" "$columnCont_string"





conda activate plink



plink2 \
  --bfile /scr1/users/haltomj/PTB/Data/qc/maternal_ptb_globalPCA \
  --keep IDs.txt \
  --chr 1-22 \
  --snps-only just-acgt \
  --max-alleles 2 \
  --geno 0.05 \
  --mind 0.05 \
  --maf 0.01 \
  --hwe 1e-6 midp \
  --threads 8 \
  --make-bed --out nDNA_final > qc.log 2>&1



mkdir PCA
# PCA for ancestry covariates
plink2 --bfile nDNA_final --pca 10 --out PCA/out

# QC summary stats
plink2 --bfile nDNA_final --missing --freq --out nDNA_stats




conda activate ML
python ../scripts/outlierPCA.py


# Call the Python script with the column string as an argument
python ../scripts/WeibullFiltering.py  "$target"> out.txt







conda activate plink



plink2 --bfile nDNA_final --keep IDs2.txt --make-bed --out nDNA_final2

mkdir -p PCA2
#Run plink PCA
plink2  --bfile nDNA_final2 --pca 10 --out PCA2/cleaned



conda activate ML
python  ../scripts/CombinePCA.py



python site_pc_structure_tests.py \
  --input Metadata.Final.tsv \
  --sep $'\t' \
  --site-col site \
  --pc-prefix PC \
  --n-pcs 5 \
  --permutations 999 \
  --out-prefix nDNA_PC_site


module load R
sed "s/MainHap/$target/g" finalModel_freq_vs_bayesian.r > finalModel_freq_vs_bayesian_IDENT.r
Rscript finalModel_freq_vs_bayesian_IDENT.r REF COVARIATES
#Base model
Rscript finalModel_freq_vs_bayesian_IDENT.r REF "BABY_SEX + PW_AGE + MAT_HEIGHT + PC1 + PC2 + PC3"



for dir in model_outputs/*/; do
    mv "$dir" "${dir%/}@IDENT"
done


mkdir -p ../Final_Model_Results

cp -r model_outputs/* ../Final_Model_Results
