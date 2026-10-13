





# Get plink2's most up2date version
```
mkdir -p ~/bin/plink2_latest
cd ~/bin/plink2_latest

wget -O plink2.zip 'PASTE_THE_DOWNLOAD_URL_HERE' https://s3.amazonaws.com/plink2-assets/alpha7/plink2_linux_avx2_20261001.zip
unzip plink2.zip
chmod +x plink2

./plink2 --version

export PATH="$HOME/bin/plink2_latest:$PATH"

which plink2
plink2 --version
```


# ============================================================
# 1. RAW COHORT MISSINGNESS
# Used by masterMetadata.py to choose the best replicate
# ============================================================
```
mkdir -p qc

/home/haltomj/bin/plink2_latest/plink2 \
  --bfile plink2 \
  --missing sample-only \
  --out qc/cohort_raw_missing
 ``` 
# ============================================================
# 2. BUILD FILTERED MATERNAL METADATA
#
# Includes:
# - replicate selection
# - MOMI matching
# - live births
# - Haplogrep3 Quality >= 0.90
#
# Produces:
# momi_mothers_merged.csv
# qc/maternal_ptb.keep
# ============================================================
```
python masterMetadata.py
```
# ============================================================
# 3. CREATE INITIAL MATERNAL PLINK COHORT
# ============================================================
```
/home/haltomj/bin/plink2_latest/plink2 \
  --bfile plink2 \
  --keep qc/maternal_ptb.keep \
  --make-bed \
  --out qc/maternal_ptb
#Those should match.
wc -l qc/maternal_ptb.keep
wc -l qc/maternal_ptb.fam
```

# ============================================================
# 4. INITIAL STUDY + 1000G PCA
#
# Uses:
# qc/maternal_ptb
#
# Produces:
# pca/joint_global.eigenvec
# pca/joint_global.eigenval
# ============================================================

```
snakemake -j 22 -s snakefile --use-conda  --config run=initial --latency-wait 60 --cluster "sbatch -t 05:00:00 -c 8 -N 1"
```

# ============================================================
# 5. GLOBAL ANCESTRY QC
#
# Uses first PCA.
# Flags broad ancestry mismatches.
#
# Produces:
# qc/maternal_ptb_globalPCA.keep
# ============================================================
```
python globalPCA_pipeline.py --stage initial
```
# ============================================================
# 6. CREATE ANCESTRY-CLEAN MATERNAL COHORT
# ============================================================
```
/home/haltomj/bin/plink2_latest/plink2 \
  --bfile qc/maternal_ptb \
  --keep qc/maternal_ptb_globalPCA.keep \
  --make-bed \
  --out qc/maternal_ptb_globalPCA

wc -l qc/maternal_ptb_globalPCA.fam
```

# ============================================================
# 7. FINAL CLEAN STUDY + 1000G PCA
#
# Uses:
# qc/maternal_ptb_globalPCA
#
# Produces:
# pca/joint_global_final.eigenvec
# pca/joint_global_final.eigenval
# ============================================================

```
snakemake -j 22 -s snakefile --use-conda  --config run=final --latency-wait 60 --cluster "sbatch -t 05:00:00 -c 8 -N 1"
```


# ============================================================
# 8. FINAL 1000G ANNOTATION + ULTIMATE METADATA
# ============================================================
```
python globalPCA_pipeline.py --stage final
```

