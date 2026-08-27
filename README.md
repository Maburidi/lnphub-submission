# LNP Hub Submission

Initial repository setup.

## Oracle Training Layout

The Colab launcher notebook is `notebooks/in_vito_training_orachels.ipynb`.

It creates deterministic in-vitro expression splits under:

```text
oracle_splits/in_vitro_expression/
```

Key outputs:

```text
lnphub_oracle_in_vitro_expression_trainval_5fold.csv
lnphub_oracle_in_vitro_expression_untouched_test.csv
lnphub_oracle_in_vitro_expression_split_assignments.csv
encoded_trainval/
encoded_untouched_test/
```

Model artifacts are saved under:

```text
trained_oracles/in_vitro_expression/
```

Each trained oracle folder contains reusable model files, metadata, metrics, and predictions, so the artifacts can later be uploaded to Hugging Face and loaded by the LNPHub package.
