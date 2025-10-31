


#| Step                      | Filter              | Meaning                            |
#| ------------------------- | ------------------- | ---------------------------------- |
#| `--bfile nDNA_raw`        | input               | your starting `.bed/.bim/.fam`     |
#| `--chr 1-22`              | autosomes only      | drops chr M, X, Y                  |
#| `--snps-only just-acgt`   | variant type        | drop indels and non-ACGT calls     |
#| `--biallelic-only strict` | allele structure    | keeps only clean biallelic SNPs    |
#| `--geno 0.05`             | variant missingness | removes SNPs with > 5 % missing    |
#| `--mind 0.05`             | sample missingness  | removes samples with > 5 % missing |
#| `--maf 0.01`              | allele frequency    | keeps MAF ≥ 1 %                    |
#| `--hwe 1e-6 midp`         | Hardy–Weinberg      | removes SNPs failing HWE ≤ 1e-6    |
#| `--make-bed`              | output              | writes new binary dataset          |
#| `--out nDNA_final`        | prefix              | output name for filtered data      |



