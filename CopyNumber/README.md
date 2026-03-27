
```
conda env create -f mtcn_slurm_env.yml
conda activate mtcn-parallel
```


# Step A: build BAM list
```
bash make_bam_list.sh /path/to/full_bams bam_list.txt
```

# Step B: count BAMs
```
wc -l bam_list.txt
```
Suppose it says:
```
127 bam_list.txt
```
Then submit:
```
sbatch --array=1-127 mtcn_array.sh \
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
# Step C: after jobs finish, merge
```bash merge_mtcn_rows.sh mtcn_results/rows mtcn_results/output.tsv
```


Look at logs:
```
ls logs/
tail -f logs/mtcn_<jobid>_<taskid>.out
tail -f logs/mtcn_<jobid>_<taskid>.err
```
