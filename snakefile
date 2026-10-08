# ============================================================
# SETTINGS
# ============================================================

CHR = [str(x) for x in range(1, 23)]

VCF_DIR = "1000GP_Data"
PLINK_DIR = "1000GP_PLINK"
PLINK2 = "/home/haltomj/bin/plink2_latest/plink2"

RUN = config.get("run", "initial")

if RUN == "initial":
    COHORT_PREFIX = "qc/maternal_ptb"
    SUFFIX = ""
elif RUN == "final":
    COHORT_PREFIX = "qc/maternal_ptb_globalPCA"
    SUFFIX = "_final"
else:
    raise ValueError("run must be 'initial' or 'final'")

SHARED_DIR = f"shared{SUFFIX}"
SHARED_1KG_DIR = f"shared_1kg{SUFFIX}"
SHARED_COHORT_DIR = f"shared_cohort{SUFFIX}"
JOINT_CHR_DIR = f"joint_chr{SUFFIX}"
JOINT_PGEN_DIR = f"joint_chr_pgen{SUFFIX}"

MERGED_PREFIX = f"merged/joint_1000G_cohort{SUFFIX}"
VARIANT_QC_PREFIX = f"qc/joint_variant_qc{SUFFIX}"
SAMPLE_QC_PREFIX = f"qc/joint_sample_qc{SUFFIX}"
LD_PREFIX = f"qc/joint_ld{SUFFIX}"
PCA_PREFIX = f"pca/joint_global{SUFFIX}"

JOINT_MISSING_PREFIX = f"qc/joint_missing{SUFFIX}"
POSTVAR_MISSING_PREFIX = f"qc/joint_variant_qc_missing{SUFFIX}"
POSTQC_MISSING_PREFIX = f"qc/joint_postqc_missing{SUFFIX}"


# ============================================================
# TARGETS
# ============================================================

rule all:
    input:
        f"{MERGED_PREFIX}.pgen",
        f"{MERGED_PREFIX}.pvar",
        f"{MERGED_PREFIX}.psam",
        f"{PCA_PREFIX}.eigenvec",
        f"{PCA_PREFIX}.eigenval",
        f"{POSTQC_MISSING_PREFIX}.smiss"


# ============================================================
# DOWNLOAD 1000 GENOMES
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

        wget --continue \
          -O {output.vcf} \
          ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/ALL.chr{wildcards.chr}.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz
        """


# ============================================================
# CONVERT 1000 GENOMES TO PLINK2
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


# ============================================================
# FIND SHARED SNPS
# ============================================================

rule SharedSNPs:
    input:
        pvar=f"{PLINK_DIR}/1000G_chr{{chr}}.pvar",
        bim=f"{COHORT_PREFIX}.bim"
    output:
        f"{SHARED_DIR}/shared_chr{{chr}}.ids"
    shell:
        r"""
        mkdir -p {SHARED_DIR}

        awk '!/^#/ {{print $3}}' {input.pvar} \
          | sort -u \
          > {SHARED_DIR}/1kg_chr{wildcards.chr}.ids

        awk -v chr={wildcards.chr} '
            $1==chr &&
            length($5)==1 &&
            length($6)==1 &&
            $5 ~ /^[ACGT]$/ &&
            $6 ~ /^[ACGT]$/ {{
                print $2
            }}' {input.bim} \
          | sort -u \
          > {SHARED_DIR}/cohort_chr{wildcards.chr}.ids

        comm -12 \
          {SHARED_DIR}/cohort_chr{wildcards.chr}.ids \
          {SHARED_DIR}/1kg_chr{wildcards.chr}.ids \
          > {output}
        """


# ============================================================
# EXTRACT SHARED SNPS FROM 1000 GENOMES
# ============================================================

rule Extract1KGPShared:
    input:
        pgen=f"{PLINK_DIR}/1000G_chr{{chr}}.pgen",
        pvar=f"{PLINK_DIR}/1000G_chr{{chr}}.pvar",
        psam=f"{PLINK_DIR}/1000G_chr{{chr}}.psam",
        shared=f"{SHARED_DIR}/shared_chr{{chr}}.ids"
    output:
        bed=f"{SHARED_1KG_DIR}/1000G_chr{{chr}}.bed",
        bim=f"{SHARED_1KG_DIR}/1000G_chr{{chr}}.bim",
        fam=f"{SHARED_1KG_DIR}/1000G_chr{{chr}}.fam"
    shell:
        r"""
        mkdir -p {SHARED_1KG_DIR}

        {PLINK2} \
          --pfile {PLINK_DIR}/1000G_chr{wildcards.chr} \
          --extract {input.shared} \
          --make-bed \
          --out {SHARED_1KG_DIR}/1000G_chr{wildcards.chr}
        """


# ============================================================
# EXTRACT SHARED SNPS FROM STUDY COHORT
# ============================================================

rule ExtractCohortSharedChr:
    input:
        bed=f"{COHORT_PREFIX}.bed",
        bim=f"{COHORT_PREFIX}.bim",
        fam=f"{COHORT_PREFIX}.fam",
        shared=f"{SHARED_DIR}/shared_chr{{chr}}.ids"
    output:
        bed=f"{SHARED_COHORT_DIR}/cohort_chr{{chr}}.bed",
        bim=f"{SHARED_COHORT_DIR}/cohort_chr{{chr}}.bim",
        fam=f"{SHARED_COHORT_DIR}/cohort_chr{{chr}}.fam"
    shell:
        r"""
        mkdir -p {SHARED_COHORT_DIR}

        {PLINK2} \
          --bfile {COHORT_PREFIX} \
          --chr {wildcards.chr} \
          --extract {input.shared} \
          --make-bed \
          --out {SHARED_COHORT_DIR}/cohort_chr{wildcards.chr}
        """


# ============================================================
# MERGE STUDY + 1000 GENOMES BY CHROMOSOME
# ============================================================

rule MergeJointChr:
    input:
        kg_bed=f"{SHARED_1KG_DIR}/1000G_chr{{chr}}.bed",
        kg_bim=f"{SHARED_1KG_DIR}/1000G_chr{{chr}}.bim",
        kg_fam=f"{SHARED_1KG_DIR}/1000G_chr{{chr}}.fam",
        cohort_bed=f"{SHARED_COHORT_DIR}/cohort_chr{{chr}}.bed",
        cohort_bim=f"{SHARED_COHORT_DIR}/cohort_chr{{chr}}.bim",
        cohort_fam=f"{SHARED_COHORT_DIR}/cohort_chr{{chr}}.fam"
    output:
        bed=f"{JOINT_CHR_DIR}/joint_chr{{chr}}.bed",
        bim=f"{JOINT_CHR_DIR}/joint_chr{{chr}}.bim",
        fam=f"{JOINT_CHR_DIR}/joint_chr{{chr}}.fam"
    resources:
        mem_mb=75000
    shell:
        r"""
        mkdir -p {JOINT_CHR_DIR}

        plink \
          --bfile {SHARED_1KG_DIR}/1000G_chr{wildcards.chr} \
          --bmerge {SHARED_COHORT_DIR}/cohort_chr{wildcards.chr} \
          --make-bed \
          --out {JOINT_CHR_DIR}/joint_chr{wildcards.chr}
        """


# ============================================================
# CONVERT EACH JOINT CHROMOSOME TO PGEN
# ============================================================

rule JointChrToPGEN:
    input:
        bed=f"{JOINT_CHR_DIR}/joint_chr{{chr}}.bed",
        bim=f"{JOINT_CHR_DIR}/joint_chr{{chr}}.bim",
        fam=f"{JOINT_CHR_DIR}/joint_chr{{chr}}.fam"
    output:
        pgen=f"{JOINT_PGEN_DIR}/joint_chr{{chr}}.pgen",
        pvar=f"{JOINT_PGEN_DIR}/joint_chr{{chr}}.pvar",
        psam=f"{JOINT_PGEN_DIR}/joint_chr{{chr}}.psam"
    shell:
        r"""
        mkdir -p {JOINT_PGEN_DIR}

        {PLINK2} \
          --bfile {JOINT_CHR_DIR}/joint_chr{wildcards.chr} \
          --make-pgen \
          --out {JOINT_PGEN_DIR}/joint_chr{wildcards.chr}
        """


# ============================================================
# MERGE CHROMOSOMES
# ============================================================

rule MergeJointChromosomes:
    input:
        pgen=expand(f"{JOINT_PGEN_DIR}/joint_chr{{chr}}.pgen", chr=CHR),
        pvar=expand(f"{JOINT_PGEN_DIR}/joint_chr{{chr}}.pvar", chr=CHR),
        psam=expand(f"{JOINT_PGEN_DIR}/joint_chr{{chr}}.psam", chr=CHR)
    output:
        pgen=f"{MERGED_PREFIX}.pgen",
        pvar=f"{MERGED_PREFIX}.pvar",
        psam=f"{MERGED_PREFIX}.psam"
    shell:
        r"""
        mkdir -p merged

        > merged/joint_chr_merge_list{SUFFIX}.txt

        for CHR in {{2..22}}; do
            echo "{JOINT_PGEN_DIR}/joint_chr${{CHR}}" \
              >> merged/joint_chr_merge_list{SUFFIX}.txt
        done

        {PLINK2} \
          --pfile {JOINT_PGEN_DIR}/joint_chr1 \
          --pmerge-list merged/joint_chr_merge_list{SUFFIX}.txt \
          --make-pgen \
          --out {MERGED_PREFIX}
        """


# ============================================================
# INITIAL MISSINGNESS ON JOINT DATASET
# ============================================================

rule JointMissingness:
    input:
        pgen=f"{MERGED_PREFIX}.pgen",
        pvar=f"{MERGED_PREFIX}.pvar",
        psam=f"{MERGED_PREFIX}.psam"
    output:
        smiss=f"{JOINT_MISSING_PREFIX}.smiss",
        vmiss=f"{JOINT_MISSING_PREFIX}.vmiss"
    shell:
        r"""
        mkdir -p qc

        {PLINK2} \
          --pfile {MERGED_PREFIX} \
          --missing \
          --out {JOINT_MISSING_PREFIX}
        """


# ============================================================
# VARIANT QC
#
# geno 0.02 = remove variants missing >2%
# maf 0.05  = common markers for ancestry PCA
# ============================================================

rule JointVariantQC:
    input:
        pgen=f"{MERGED_PREFIX}.pgen",
        pvar=f"{MERGED_PREFIX}.pvar",
        psam=f"{MERGED_PREFIX}.psam",
        smiss=f"{JOINT_MISSING_PREFIX}.smiss",
        vmiss=f"{JOINT_MISSING_PREFIX}.vmiss"
    output:
        pgen=f"{VARIANT_QC_PREFIX}.pgen",
        pvar=f"{VARIANT_QC_PREFIX}.pvar",
        psam=f"{VARIANT_QC_PREFIX}.psam"
    shell:
        r"""
        {PLINK2} \
          --pfile {MERGED_PREFIX} \
          --geno 0.02 \
          --maf 0.05 \
          --make-pgen \
          --out {VARIANT_QC_PREFIX}
        """


# ============================================================
# SAMPLE MISSINGNESS AFTER VARIANT QC
# ============================================================

rule PostVariantMissingness:
    input:
        pgen=f"{VARIANT_QC_PREFIX}.pgen",
        pvar=f"{VARIANT_QC_PREFIX}.pvar",
        psam=f"{VARIANT_QC_PREFIX}.psam"
    output:
        smiss=f"{POSTVAR_MISSING_PREFIX}.smiss"
    shell:
        r"""
        {PLINK2} \
          --pfile {VARIANT_QC_PREFIX} \
          --missing sample-only \
          --out {POSTVAR_MISSING_PREFIX}
        """


# ============================================================
# SAMPLE QC
#
# mind 0.05 = remove samples missing >5%
# ============================================================

rule JointSampleQC:
    input:
        pgen=f"{VARIANT_QC_PREFIX}.pgen",
        pvar=f"{VARIANT_QC_PREFIX}.pvar",
        psam=f"{VARIANT_QC_PREFIX}.psam",
        smiss=f"{POSTVAR_MISSING_PREFIX}.smiss"
    output:
        pgen=f"{SAMPLE_QC_PREFIX}.pgen",
        pvar=f"{SAMPLE_QC_PREFIX}.pvar",
        psam=f"{SAMPLE_QC_PREFIX}.psam"
    shell:
        r"""
        {PLINK2} \
          --pfile {VARIANT_QC_PREFIX} \
          --mind 0.05 \
          --make-pgen \
          --out {SAMPLE_QC_PREFIX}
        """


# ============================================================
# POST-QC MISSINGNESS
# ============================================================

rule PostQCMissingness:
    input:
        pgen=f"{SAMPLE_QC_PREFIX}.pgen",
        pvar=f"{SAMPLE_QC_PREFIX}.pvar",
        psam=f"{SAMPLE_QC_PREFIX}.psam"
    output:
        smiss=f"{POSTQC_MISSING_PREFIX}.smiss"
    shell:
        r"""
        {PLINK2} \
          --pfile {SAMPLE_QC_PREFIX} \
          --missing sample-only \
          --out {POSTQC_MISSING_PREFIX}
        """


# ============================================================
# LONG-RANGE LD REGION
# GRCh38
# ============================================================

rule LongRangeLDRegions:
    output:
        "qc/long_range_ld.txt"
    shell:
        r"""
        mkdir -p qc
        echo -e "6\t25000000\t35000000" > {output}
        """


# ============================================================
# LD PRUNING
# ============================================================

rule LDPrune:
    input:
        pgen=f"{SAMPLE_QC_PREFIX}.pgen",
        pvar=f"{SAMPLE_QC_PREFIX}.pvar",
        psam=f"{SAMPLE_QC_PREFIX}.psam",
        regions="qc/long_range_ld.txt"
    output:
        prune_in=f"{LD_PREFIX}.prune.in",
        prune_out=f"{LD_PREFIX}.prune.out"
    shell:
        r"""
        {PLINK2} \
          --pfile {SAMPLE_QC_PREFIX} \
          --exclude range {input.regions} \
          --indep-pairwise 200 50 0.2 \
          --out {LD_PREFIX}
        """


# ============================================================
# GLOBAL JOINT PCA
# ============================================================

rule GlobalPCA:
    input:
        pgen=f"{SAMPLE_QC_PREFIX}.pgen",
        pvar=f"{SAMPLE_QC_PREFIX}.pvar",
        psam=f"{SAMPLE_QC_PREFIX}.psam",
        prune=f"{LD_PREFIX}.prune.in"
    output:
        eigenvec=f"{PCA_PREFIX}.eigenvec",
        eigenval=f"{PCA_PREFIX}.eigenval"
    threads:
        8
    shell:
        r"""
        mkdir -p pca

        {PLINK2} \
          --pfile {SAMPLE_QC_PREFIX} \
          --extract {input.prune} \
          --pca 20 approx \
          --threads {threads} \
          --out {PCA_PREFIX}
        """
