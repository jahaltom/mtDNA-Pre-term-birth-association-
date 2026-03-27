
```
conda env create -f mtcn_slurm_env.yml
conda activate mtcn-parallel
```


# Step A: build BAM list
```
ls path/to/bam/dir/* | cat | grep -v "bai" >  bam_list.txt
```
bam_list.txt
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
Then submit: (%25 means there is a max 25 jobs running at once, adjust to your HPCs limits)
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
# Step C: after jobs finish, merge
```
bash merge_mtcn_rows.sh mtcn_results/rows mtcn_results/output.tsv
```


Look at logs:
```
ls logs/
tail -f logs/mtcn_<jobid>_<taskid>.out
tail -f logs/mtcn_<jobid>_<taskid>.err
```
