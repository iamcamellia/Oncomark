# Data Sources

Raw patient data is **not included** in this repository due to data sharing agreements and file size. All datasets used in this project are publicly available.

---

## Training Data

| Dataset | Description | Access |
|---|---|---|
| Weizmann 3CA | 3.1M single-cell transcriptomes, 941 patients, 14 tumor types | [weizmann.ac.il/sites/3CA](https://www.weizmann.ac.il/sites/3CA/) |
| Synthetic pseudo-bulk (Dryad) | Generated training data (hallmark-annotated bulk RNA-seq profiles) | [doi.org/10.5061/dryad.zw3r228jc](https://doi.org/10.5061/dryad.zw3r228jc) |

## Validation Datasets

| Dataset | n | Access |
|---|---|---|
| TCGA (The Cancer Genome Atlas) | 6,679 samples | [gdac.broadinstitute.org](https://gdac.broadinstitute.org/) |
| GTEx (normal tissue) | 8,228 samples | [gtexportal.org](https://www.gtexportal.org/home/) |
| CCLE (cell lines) | 1,019 samples | [sites.broadinstitute.org/ccle](https://sites.broadinstitute.org/ccle/datasets) |
| MET500 (metastatic) | 868 samples | [xenabrowser.net](https://xenabrowser.net/datapages/) |
| POG570 | 570 samples | [bcgsc.ca/downloads/POG570](https://www.bcgsc.ca/downloads/POG570/) |
| PCAWG (whole genomes) | 1,210 samples | [cbioportal.org](https://www.cbioportal.org/) |
| ENCODE (normal) | 329 samples | [encodeproject.org](https://www.encodeproject.org/) |
| TARGET (pediatric) | 734 samples | [xenabrowser.net](https://xenabrowser.net/datapages/) |

## Input Format

OncoMark expects:
- **Format:** CSV
- **Rows:** samples (patients or cell lines)
- **Columns:** gene names (HGNC symbols)
- **Values:** raw counts or normalized expression (non-negative)

```
          TP53   KRAS   EGFR  ...
sample_1   124    45    230   ...
sample_2    89    12    415   ...
```

## Sample Data

A small synthetic sample dataset is available in `sample_input.csv` (50 samples × 500 genes) for testing the preprocessing pipeline without downloading full datasets.
