
```
bcftools view -i 'ALT!="."' merged_chrM_22175.vcf -o filtered.vcf
plink --vcf filtered.vcf --max-maf 0.01 --recode vcf --out rare_variants
rm filtered.vcf
```

```python
import cyvcf2
import pandas as pd

# Load VCF
vcf_path = "rare_variants.vcf"
vcf = cyvcf2.VCF(vcf_path)

# Extract samples and variants
samples = vcf.samples
variant_data = []

for variant in vcf:
    pos = variant.POS
    ref = variant.REF
    alt = ", ".join(variant.ALT)
    genotypes = variant.genotypes  # List of genotypes for all samples
    binary_encoding = [1 if g[0] == 1   else 0 for g in genotypes]
    variant_data.append(binary_encoding)

# Convert to DataFrame
variant_df = pd.DataFrame(variant_data).T


vcf = cyvcf2.VCF(vcf_path)

variant_df.columns = [f"Variant_{v.POS}_{v.REF}>{", ".join(v.ALT)}" for v in vcf]
variant_df.index = samples

variant_df.index.name = 'SampleID'
variant_df.reset_index(inplace=True)

variant_df.to_csv('rare_variants.NN.tsv', index=None,sep='\t')


```
