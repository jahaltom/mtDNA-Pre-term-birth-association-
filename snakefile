# ============================================================
# 1000 Genomes download + PLINK2 conversion
# ============================================================

CHR = [str(x) for x in range(1, 23)]

VCF_DIR = "1000GP_Data"
PLINK_DIR = "1000GP_PLINK"
PLINK2 = "/home/haltomj/bin/plink2_latest/plink2"

rule all:
    input:
        "merged/joint_1000G_cohort.pgen",
        "merged/joint_1000G_cohort.pvar",
        "merged/joint_1000G_cohort.psam",

        "pca/joint_global.eigenvec",
        "pca/joint_global.eigenval",
        "qc/joint_postqc_missing.smiss"


# ============================================================
# Download 1000 Genomes GRCh38 VCFs
# ============================================================

rule Download1KGP:
    output:
        vcf=(
            f"{VCF_DIR}/"
            "ALL.chr{chr}.shapeit2_integrated_snvindels_v2a_27022019."
            "GRCh38.phased.vcf.gz"
        )
    shell:
        r"""
        mkdir -p {VCF_DIR}

        wget --continue -O {output.vcf} ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/ALL.chr{wildcards.chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz
        """


# ============================================================
# Convert 1000 Genomes VCF -> PLINK2
#
# Variant IDs are rewritten as:
# CHR:POS:REF:ALT
#
# Example:
# 22:10651162:C:A
#
# This matches your cohort variant-ID format.
# ============================================================

rule Convert1KGP:
    input:
        vcf=(
            f"{VCF_DIR}/"
            "ALL.chr{chr}.shapeit2_integrated_snvindels_v2a_27022019."
            "GRCh38.phased.vcf.gz"
        )
    output:
        pgen=f"{PLINK_DIR}/1000G_chr{{chr}}.pgen",
        pvar=f"{PLINK_DIR}/1000G_chr{{chr}}.pvar",
        psam=f"{PLINK_DIR}/1000G_chr{{chr}}.psam"
    shell:
        r"""
        mkdir -p {PLINK_DIR}

        {PLINK2} \
          --vcf {input.vcf} \
          --double-id \
          --max-alleles 2 \
          --snps-only just-acgt \
          --set-all-var-ids '@:#:$r:$a' \
          --make-pgen \
          --out {PLINK_DIR}/1000G_chr{wildcards.chr}
        """
            
rule SharedSNPs:
    input:
        pvar=f"{PLINK_DIR}/1000G_chr{{chr}}.pvar",
        bim="qc/maternal_ptb.bim"
    output:
        "shared/shared_chr{chr}.ids"
    shell:
        r"""
        mkdir -p shared

        # 1KGP SNP IDs
        awk '!/^#/ {{print $3}}' {input.pvar} \
            | sort -u \
            > shared/1kg_chr{wildcards.chr}.ids

        # Maternal cohort SNP IDs for this chromosome
        awk -v chr={wildcards.chr} '
            $1==chr &&
            length($5)==1 &&
            length($6)==1 &&
            $5 ~ /^[ACGT]$/ &&
            $6 ~ /^[ACGT]$/ {{
                print $2
            }}' {input.bim} \
            | sort -u \
            > shared/cohort_chr{wildcards.chr}.ids

        # Intersection
        comm -12 \
            shared/cohort_chr{wildcards.chr}.ids \
            shared/1kg_chr{wildcards.chr}.ids \
            > {output}
        """
rule Extract1KGPShared:
    input:
        pgen=f"{PLINK_DIR}/1000G_chr{{chr}}.pgen",
        pvar=f"{PLINK_DIR}/1000G_chr{{chr}}.pvar",
        psam=f"{PLINK_DIR}/1000G_chr{{chr}}.psam",
        shared="shared/shared_chr{chr}.ids"

    output:
        bed="shared_1kg/1000G_chr{chr}.bed",
        bim="shared_1kg/1000G_chr{chr}.bim",
        fam="shared_1kg/1000G_chr{chr}.fam"

    shell:
        r"""
        mkdir -p shared_1kg

        {PLINK2} \
          --pfile {PLINK_DIR}/1000G_chr{wildcards.chr} \
          --extract {input.shared} \
          --make-bed \
          --out shared_1kg/1000G_chr{wildcards.chr}
        """
rule ExtractCohortSharedChr:
    input:
        bed="qc/maternal_ptb.bed",
        bim="qc/maternal_ptb.bim",
        fam="qc/maternal_ptb.fam",
        shared="shared/shared_chr{chr}.ids"

    output:
        bed="shared_cohort/cohort_chr{chr}.bed",
        bim="shared_cohort/cohort_chr{chr}.bim",
        fam="shared_cohort/cohort_chr{chr}.fam"

    shell:
        r"""
        mkdir -p shared_cohort

        {PLINK2} \
          --bfile qc/maternal_ptb \
          --chr {wildcards.chr} \
          --extract {input.shared} \
          --make-bed \
          --out shared_cohort/cohort_chr{wildcards.chr}
        """
rule MergeJointChr:
    input:
        kg_bed="shared_1kg/1000G_chr{chr}.bed",
        kg_bim="shared_1kg/1000G_chr{chr}.bim",
        kg_fam="shared_1kg/1000G_chr{chr}.fam",

        cohort_bed="shared_cohort/cohort_chr{chr}.bed",
        cohort_bim="shared_cohort/cohort_chr{chr}.bim",
        cohort_fam="shared_cohort/cohort_chr{chr}.fam"

    output:
        bed="joint_chr/joint_chr{chr}.bed",
        bim="joint_chr/joint_chr{chr}.bim",
        fam="joint_chr/joint_chr{chr}.fam"

    resources:
        mem_mb=75000

    shell:
        r"""
        mkdir -p joint_chr

        plink \
          --bfile shared_1kg/1000G_chr{wildcards.chr} \
          --bmerge shared_cohort/cohort_chr{wildcards.chr} \
          --make-bed \
          --out joint_chr/joint_chr{wildcards.chr}
        """
rule JointChrToPGEN:
    input:
        bed="joint_chr/joint_chr{chr}.bed",
        bim="joint_chr/joint_chr{chr}.bim",
        fam="joint_chr/joint_chr{chr}.fam"

    output:
        pgen="joint_chr_pgen/joint_chr{chr}.pgen",
        pvar="joint_chr_pgen/joint_chr{chr}.pvar",
        psam="joint_chr_pgen/joint_chr{chr}.psam"

    shell:
        r"""
        mkdir -p joint_chr_pgen

        {PLINK2} \
          --bfile joint_chr/joint_chr{wildcards.chr} \
          --make-pgen \
          --out joint_chr_pgen/joint_chr{wildcards.chr}
        """
rule MergeJointChromosomes:
    input:
        pgen=expand("joint_chr_pgen/joint_chr{chr}.pgen", chr=CHR),
        pvar=expand("joint_chr_pgen/joint_chr{chr}.pvar", chr=CHR),
        psam=expand("joint_chr_pgen/joint_chr{chr}.psam", chr=CHR)

    output:
        pgen="merged/joint_1000G_cohort.pgen",
        pvar="merged/joint_1000G_cohort.pvar",
        psam="merged/joint_1000G_cohort.psam"

    shell:
        r"""
        mkdir -p merged

        > merged/joint_chr_merge_list.txt

        for CHR in {{2..22}}; do
            echo "joint_chr_pgen/joint_chr${{CHR}}" \
                >> merged/joint_chr_merge_list.txt
        done

        {PLINK2} \
          --pfile joint_chr_pgen/joint_chr1 \
          --pmerge-list merged/joint_chr_merge_list.txt \
          --make-pgen \
          --out merged/joint_1000G_cohort
        """


# ============================================================
# Initial missingness report on merged cohort + 1000G
# ============================================================

rule JointMissingness:
    input:
        pgen="merged/joint_1000G_cohort.pgen",
        pvar="merged/joint_1000G_cohort.pvar",
        psam="merged/joint_1000G_cohort.psam"
    output:
        smiss="qc/joint_missing.smiss",
        vmiss="qc/joint_missing.vmiss"
    shell:
        r"""
        mkdir -p qc

        {PLINK2} \
          --pfile merged/joint_1000G_cohort \
          --missing \
          --out qc/joint_missing
        """

# ============================================================
# Variant QC
#
# geno 0.02 = remove variants missing in >2% of samples
# maf 0.05  = retain common SNPs for ancestry PCA
# ============================================================

rule JointVariantQC:
    input:
        pgen="merged/joint_1000G_cohort.pgen",
        pvar="merged/joint_1000G_cohort.pvar",
        psam="merged/joint_1000G_cohort.psam",
        smiss="qc/joint_missing.smiss",
        vmiss="qc/joint_missing.vmiss"
    output:
        pgen="qc/joint_variant_qc.pgen",
        pvar="qc/joint_variant_qc.pvar",
        psam="qc/joint_variant_qc.psam"
    shell:
        r"""
        {PLINK2} \
          --pfile merged/joint_1000G_cohort \
          --geno 0.02 \
          --maf 0.05 \
          --make-pgen \
          --out qc/joint_variant_qc
        """

# ============================================================
# Sample missingness after variant QC
# ============================================================

rule PostVariantMissingness:
    input:
        pgen="qc/joint_variant_qc.pgen",
        pvar="qc/joint_variant_qc.pvar",
        psam="qc/joint_variant_qc.psam"
    output:
        smiss="qc/joint_variant_qc_missing.smiss"
    shell:
        r"""
        {PLINK2} \
          --pfile qc/joint_variant_qc \
          --missing sample-only \
          --out qc/joint_variant_qc_missing
        """

# ============================================================
# Sample QC
#
# mind 0.05 = remove samples missing >5% of retained variants
# ============================================================

rule JointSampleQC:
    input:
        pgen="qc/joint_variant_qc.pgen",
        pvar="qc/joint_variant_qc.pvar",
        psam="qc/joint_variant_qc.psam",
        smiss="qc/joint_variant_qc_missing.smiss"
    output:
        pgen="qc/joint_sample_qc.pgen",
        pvar="qc/joint_sample_qc.pvar",
        psam="qc/joint_sample_qc.psam"
    shell:
        r"""
        {PLINK2} \
          --pfile qc/joint_variant_qc \
          --mind 0.05 \
          --make-pgen \
          --out qc/joint_sample_qc
        """
# ============================================================
# Missingness report after sample + variant QC
# ============================================================

rule PostQCMissingness:
    input:
        pgen="qc/joint_sample_qc.pgen",
        pvar="qc/joint_sample_qc.pvar",
        psam="qc/joint_sample_qc.psam"
    output:
        smiss="qc/joint_postqc_missing.smiss"
    shell:
        r"""
        {PLINK2} \
          --pfile qc/joint_sample_qc \
          --missing sample-only \
          --out qc/joint_postqc_missing
        """
# ============================================================
# Long-range LD regions to exclude from PCA marker selection
# GRCh38
# ============================================================

rule LongRangeLDRegions:
    output:
        "qc/long_range_ld.txt"
    shell:
        r"""
        mkdir -p qc

        echo -e "6\t25000000\t35000000" \
          > {output}
        """

# ============================================================
# LD pruning for ancestry PCA
# ============================================================

rule LDPrune:
    input:
        pgen="qc/joint_sample_qc.pgen",
        pvar="qc/joint_sample_qc.pvar",
        psam="qc/joint_sample_qc.psam",
        regions="qc/long_range_ld.txt"
    output:
        prune_in="qc/joint_ld.prune.in",
        prune_out="qc/joint_ld.prune.out"
    shell:
        r"""
        {PLINK2} \
          --pfile qc/joint_sample_qc \
          --exclude range {input.regions} \
          --indep-pairwise 200 50 0.2 \
          --out qc/joint_ld
        """

# ============================================================
# Global joint PCA
#
# Study cohort + 1000 Genomes together
# No projection
# ============================================================

rule GlobalPCA:
    input:
        pgen="qc/joint_sample_qc.pgen",
        pvar="qc/joint_sample_qc.pvar",
        psam="qc/joint_sample_qc.psam",
        prune="qc/joint_ld.prune.in"
    output:
        eigenvec="pca/joint_global.eigenvec",
        eigenval="pca/joint_global.eigenval"
    threads:
        8
    shell:
        r"""
        mkdir -p pca

        {PLINK2} \
          --pfile qc/joint_sample_qc \
          --extract {input.prune} \
          --pca 20 approx \
          --threads {threads} \
          --out pca/joint_global
        """
