
# This workflow computes mitochondrial DNA copy number (mtCN) from whole-genome sequencing (WGS) BAM files.

## mtCN = 2 × mean_mtDNA_coverage / median_autosomal_coverage

Key Features:
- Slurm array-based parallelization (1 BAM per job)
- Autosomal-only normalization (chr1–22)
- Robust handling of BAM indexing and mtDNA contig detection
- Safe parallel output handling


Notes:
- Reference FASTA must match BAM alignment
- Autosomal coverage uses chromosomes 1–22 only

  
Output Columns:
- sample
- bam
- mt_contig
- mean_mt_cov
- median_autosomal_cov
- mtcn




# Prepare the environment 
```
conda env create -f mtcn_slurm_env.yml
conda activate mtcn-parallel
```


# Step A: build BAM list
```
ls path/to/bam/dir/* | cat | grep -v "bai" >  bam_list.txt
```
bam_list.txt:
```
output_dir/NA12718.final.sorted.bam
output_dir/test_sample.sorted.bam
```
# Step B: count BAMs
```
wc -l bam_list.txt
```
Suppose it says:
```
127 bam_list.txt
```
Then submit: (%25 means there is a max of 25 jobs running at once, adjust to your HPCs limits)
```
sbatch --array=1-127%25 mtcn_array.sh \
  bam_list.txt \
  /path/to/Homo_sapiens_assembly38.fasta \
  mtcn_results
```

That will create:
```
mtcn_results/rows/sample1.tsv
mtcn_results/rows/sample2.tsv
...
```
### CollectWgsMetrics explained

#### What CollectWgsMetrics Does in This Workflow

Picard `CollectWgsMetrics` is used to estimate **nuclear (autosomal) sequencing depth**, which is required to normalize mitochondrial coverage into copy number.

#### Purpose in this pipeline

- Computes coverage statistics across the genome
- Restricted to **autosomes (chromosomes 1–22)** using an interval list
- Applies quality filters to remove low-confidence reads and bases
- Produces a robust estimate of typical nuclear depth

#### Key output used

The workflow extracts:

MEDIAN_COVERAGE

This represents:
- The median per-base sequencing depth across autosomes
- A stable estimate of diploid nuclear coverage

#### Role in mtDNA copy number

This value is used as the denominator in the mtCN calculation.


#### Parameters:  
##### INTERVALS
- Interval list specifying genomic regions to analyze
- In this workflow: autosomes only (chromosomes 1–22)
- Prevents bias from sex chromosomes, mitochondrial DNA, and contigs

##### MINIMUM_MAPPING_QUALITY=20
- Excludes reads with mapping quality < 20
- Removes poorly aligned or ambiguous reads

##### MINIMUM_BASE_QUALITY=20
- Excludes bases with base quality < 20
- Ensures only high-confidence bases are used

##### COVERAGE_CAP=100000
- Caps extremely high coverage values
- Prevents repetitive regions or artifacts from skewing results



# Step C: after jobs finish, merge
```
bash merge_mtcn_rows.sh mtcn_results/rows mtcn_results/output.tsv
```
output.tsv:
```
sample  bam     mt_contig       mean_mt_cov     median_autosomal_cov    mtcn
NA12718.final.sorted    output_dir/NA12718.final.sorted.bam     chrM    15950.172974    31      1029.043418
test_sample.sorted      output_dir/test_sample.sorted.bam       chrM    70.709578       1       141.419156
```

Look at logs:
```
ls logs/
tail -f logs/mtcn_<jobid>_<taskid>.out
tail -f logs/mtcn_<jobid>_<taskid>.err
```
