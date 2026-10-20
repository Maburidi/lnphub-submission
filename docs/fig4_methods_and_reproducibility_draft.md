# Fig. 4 Methods and Reproducibility Draft

Prepared from the checked-in notebook and training code on 19 October 2026.
This documents the implementation, not proof that every Colab result used this
exact revision. Resolve the verification items below before submitting.

## Suggested Replacement for Section 4.4

```latex
\subsection{Benchmarking multimodal delivery prediction and generalization}
\label{sec:methods_delivery_prediction}

\textbf{Prediction task and delivery-class definitions.}
The classification benchmark used records annotated as expression or discretized
expression, including the legacy spelling \texttt{expression\_descritized}.
Continuous outcomes were assigned to low (class 0), medium (class 1), and high
(class 2) delivery using the normalized delivery values supplied in the curated
dataset: low, $d<0$; medium, $0\leq d<1$; and high, $d\geq1$.
These classes were not defined by within-screen percentiles or equal-frequency
binning. Discretized outcomes were harmonized separately within each
study--metric combination. Three ordered levels were mapped directly to classes
0, 1, and 2. Four or five levels were reduced to three by iteratively merging
the closest adjacent level groups, preserving their order and assigning tied
outcomes to the same class. The exact study-specific mappings were recorded in
the harmonization table. Outcome normalization and the direction of benefit
must be documented in the data-curation methods.

\textbf{Input settings and molecular models.}
The main benchmark evaluated molecular structure alone; structure plus
formulation; structure, formulation, and experimental context; and these inputs
augmented with measured physicochemical properties. The first three describe
design inputs, whereas the fourth represents post-characterization prediction.
Twenty-one configurations comprised six fingerprint/descriptor classifiers,
three graph models, six frozen pretrained representations with classifiers, and
six fine-tuned pretrained encoders. The graph models comprised GCN, GIN, and an
edge-aware GINE implementation. Pretrained representations comprised AGILE,
ChemBERTa-MTR, LiON, MolCLR, MoLFormer, and Uni-Mol, with their corresponding
references. Feature definitions, preprocessing, model settings, and fusion
procedures are specified in the Supplementary Methods and accompanying code.

\textbf{Random cross-validation and repeated lipids.}
The main benchmark used stratified five-fold cross-validation at the record
level, with the same stored fold assignments across models and input settings
and no additional fixed test set. Each fold evaluated records withheld from
model fitting. Records sharing an ionizable-lipid structure were not grouped
into a single fold, and repeated lipids under different formulations or
experimental contexts could therefore occur in both training and evaluation
partitions. This benchmark measures prediction for withheld records rather
than transfer exclusively to unseen lipid identities. Performance was reported
as the arithmetic mean and sample standard deviation of fold-level metrics.

\textbf{Chemical and cross-study generalization.}
RDKit descriptors with XGBoost, GIN, and frozen ChemBERTa-MTR were evaluated
using scaffold holdout and separate leave-one-group-out experiments for reaction
family, lipid architecture, lipid class, head-group nitrogen class, head-group
architecture, and study identifier. In each leave-one-group-out experiment, all
records assigned to one specified group formed the test set. The remaining
records were divided into training and validation sets, with one ninth reserved
for validation. The withheld group was absent from both training and
validation. Category holdout and study holdout enforce separation by their
respective grouping variables; study holdout does not additionally exclude a
lipid structure when that structure also occurs in another study.

To compare information levels under these holdouts, the three representative
models were additionally evaluated using structure alone; structure plus
formulation; structure plus experimental context; structure, formulation, and
context; and these inputs plus physicochemical measurements. A further
experiment restricted physicochemical inputs to particle size, PDI, and zeta
potential, using training-partition medians for missing values and binary
missingness indicators. It reused the same record assignments as the
corresponding input-level leave-one-group experiments.

\textbf{Performance metrics and aggregation.}
The primary metric was macro-averaged one-versus-rest ROC--AUC over the three
delivery classes. Each held-out group was evaluated separately. Summary values
were the unweighted arithmetic mean and sample standard deviation of group-level
scores, so each evaluable group contributed equally regardless of its size.
Scores undefined for a held-out group, for example three-class ROC--AUC when a
required class was absent, were stored as missing and omitted from that metric's
summary. Group sizes, class counts, individual scores, and the number of groups
with defined scores should accompany the summaries.
```

## Supplementary Methods: Verified Implementation Details

### Cohort and Labels

Source: `notebooks/mulit_modal_lasttrail.ipynb`, code headed Cells 3--7.

- Endpoints retained: `expression`, `expression_discretized`, and
  `expression_descritized`. The classifier does not train on all delivery
  endpoint types in the master dataset.
- Eligibility requires numeric delivery, all four molar ratios
  (`il_mol_ratio`, `helper_lipid_mol_ratio`, `cholesterol_mol_ratio`,
  `peg_lipid_mol_ratio`), particle size, and PDI. The executable list does not
  require zeta potential, despite an outdated comment saying otherwise.
- Studies with fewer than 16 eligible records are excluded.
- Continuous class boundaries are 0 and 1 on the supplied `delivery` scale.
  The notebook does not compute the upstream normalization or reverse endpoint
  direction. Do not call these study z-scores without verifying their origin.
- For three discrete values, ascending values map to 0, 1, and 2. For four or
  five values, start with singleton groups in ascending order. Repeatedly merge
  adjacent groups with the smallest difference between their unweighted mean
  level values. Approximately equal minimum gaps use `np.isclose`; among tied
  pairs, choose the pair with the fewest combined records. Remaining ties use
  the first pair in ascending order. Recompute centers after each merge.
- Other discrete level counts fall back to the continuous thresholds. The
  saved `delivery_3class_harmonization_rules.csv` must establish whether that
  fallback occurred in the reported cohort.
- Identical delivery values within a study--metric mapping receive the same
  label. No random tie-breaking is used in class construction.
- The downsampling cell retains all class-2 records per study and caps class 1
  at `ceil(1.1 * n_class_2)` and class 0 at `ceil(1.4 * n_class_2)`, never
  oversampling. Studies without class 2 are retained unchanged. Sampling uses
  NumPy's generator seeded with 42 and generated pandas sampling seeds.
- Downsampling is performed before partitioning in the normal notebook run;
  evaluation therefore describes that selected cohort, not the original class
  prevalence. Fallback data-loading paths can load the earlier unbalanced CSV;
  verify saved cohort IDs and the balance audit from the actual run.

### Features and Missing Data

Eight formulation columns:

```python
[
    "helper_lipid_mol_ratio", "cholesterol_mol_ratio",
    "peg_lipid_mol_ratio", "il_mol_ratio",
    "il_to_nucleic_acid_mass_ratio", "il_to_nucleic_acid_charge_ratio",
    "total_lipid_to_cargo_ratio", "total_flow_rate_ml_min",
]
```

Experimental context uses one-hot encodings of `cargo_type`,
`experimental_setting`, `biological_model`, `biological_target`, and
`route_of_administration`. Grouping variables and target values are not added
as predictor columns by these input lists.

Original physicochemical inputs: `particle_size_nm`, `pdi`,
`zeta_potential_mv`, `encapsulation_efficiency_percent`, `measured_pka`.
Original notebook preparation fills missing numeric covariates with zero.
The selected-physicochemical follow-up uses only the first three measurements,
imputes their missing values from the training split median, and adds binary
indicators when missingness is detected. It raises an error if a selected
property has no observed training values. This preprocessing is different from
the original five-property experiment and must be disclosed as such.

Random-CV context categories are enumerated across the full selected cohort
to define a stable schema. This is not a training-fold-only categorical schema.
In category/study holdout experiments, context categories are fitted on training
records and unseen-category handling is recorded. Check the actual split
encoding map and unseen-category table; the original section and the later
input-level section use different encoding helper implementations.

### Models and Fusion

Sources: `src/lnphub_oracles/features.py`, `torch_models.py`,
`train_frozen_classification.py`, `train_gnn_classification.py`,
`train_finetuned_multimodal_classification.py`, and `encode_exact_frozen.py`.

| Model | Explicit implementation/settings |
| --- | --- |
| ECFP4 | Morgan radius 2; 2,048 binary features |
| RDKit descriptors | Functions in installed `Descriptors.descList`; export actual names and RDKit version because the descriptor count is version-dependent |
| MLP classifier | Median imputer, standard scaling, hidden layers 512/128, alpha 0.0001, batch size 128, max 500 iterations, internal early stopping; other options are installed scikit-learn defaults |
| Random forest | 500 trees, minimum leaf size 2, balanced-subsample class weighting, all CPU cores; other options are installed scikit-learn defaults |
| XGBoost | 700 estimators, depth 4, learning rate 0.035, row/column subsampling 0.85/0.85, multi:softprob, mlogloss, hist trees; validation logging does not itself enable early stopping |
| GCN / GIN / edge-aware graph model | Four layers, embedding width 128, batch normalization, ReLU, dropout 0.20, global mean pooling; pooled molecular embedding concatenated with tabular features before a 128-unit classification head |
| Graph training | AdamW, learning rate 0.001, weight decay 0.00001, batch 64, at most 120 epochs, patience 25; retain checkpoint selected by validation loss |
| Frozen representations | Molecular embeddings concatenated with selected tabular features and fitted using the MLP classifier above; pretrained encoder weights are unchanged |
| Fine-tuned HF models | Attention-mask mean pooling, tabular concatenation, hidden-width ReLU head with dropout 0.15; structure-only head uses dropout and linear classification |
| Fine-tuning command settings | 8 epochs, patience 3, batch 32, learning rate 0.00002, weight decay default 0.00001; encoder-specific implementations and effective settings must be taken from saved metadata |

The configuration named `D-MPNN` is implemented with `GINEConv`, not a canonical
directed-bond Chemprop D-MPNN. Describe it as an edge-aware GINE model, or
explicitly disclose the implementation behind the historical display name.

Frozen ChemBERTa uses `DeepChem/ChemBERTa-77M-MTR`; MoLFormer uses
`ibm-research/MoLFormer-XL-both-10pct`. The encoding helper uses token limits of
256. Uni-Mol is invoked through `UniMolRepr` on SMILES; no experimental 3D
coordinates are supplied by this notebook. Report the installed Uni-Mol tools
version and its coordinate-generation configuration from the run environment.

LiON is an exception to simple concatenation: structure-only inference uses a
zero auxiliary vector in its native 92-column schema, and formulation-aware
inference populates supported formulation ratios in that schema. Extra
context/physicochemical columns are appended at the prediction head where
requested. The exact native mapping and `--lion-append-tabular-features` flag
must be included with the run configuration.

### Splits, Repeated Lipids, and Aggregation

- Main random CV: shuffled `StratifiedKFold`, five folds, seed 42, using stored
  `cv_fold` IDs. No grouping by lipid, formulation ID, or study.
- Group holdout: test consists of every record matching the requested normalized
  grouping value; non-test records are split into train and validation with
  validation fraction 1/9. Stratification is attempted and falls back to a
  shuffled unstratified split if infeasible. The fallback prints a warning.
- Section 12 seed: 2027. Subsequent input-level leave-one section seed: 3031.
  They share group definitions but are not guaranteed identical train/validation
  memberships or numerical scores. The selected-physchem follow-up reuses the
  later input-level section's split record IDs.
- Held-out IL classes: biodegradable lipid; amino lipid; siloxane-based lipidoid;
  biodegradable dendron-like lipid; biodegradable amino lipid.
- Held-out studies: LNPhub_XH_2025, LNPhub_ZH_2023, LNPhub_LX_2024c,
  LNPhub_LX_2024b, LNPhub_YZ_2022, LNPhub_LX_2024, LNPhub_GN_2025,
  LNPhub_MP_2025, LNPhub_AP_2025, LNPhub_LZ_2025. Missing/ineligible requested
  groups are recorded as skipped, not given a score of zero.
- `record_id` uniqueness is enforced or duplicates removed by the relevant
  setup helper. This is not deduplication or grouping by lipid SMILES or LNP ID.
- IL-class holdout checks class separation. If identical lipid structures have
  inconsistent class annotations, this does not itself exclude them across
  partitions. Study holdout likewise allows a structure found in other studies.
- Consequently, the figure label "Lipid Held-Out Split" means IL-class holdout
  in the current figure, not an exact-identity lipid-disjoint experiment. Prefer
  "Lipid-Class Held-Out Split" or define the shortened label in the caption.
- Group summaries are unweighted means and sample SDs (`ddof=1`) of the test
  metrics. Undefined metrics become NaN and pandas omits NaNs from each summary.
  The stored number of held-out groups is not necessarily the number with
  defined AUC; report both where they differ.
- GNN and fine-tuned CV routines use the held-out fold for checkpoint selection
  as well as reported fold evaluation. These are not nested-CV estimates with
  a separate inner selection set. Describe this accurately when presenting CV.

### Within-Screen Percentiles and Top-10% Ties

Source: `notebooks/Fig3.ipynb`, cells headed "Define eligible screening
contexts", "Construct within-screen delivery ranks", and "Define top
performers". These analyses are separate from Fig. 4 class construction.

A screen is defined by study ID, experimental setting, cargo type, biological
model, biological target, administration route, and delivery metric where
available. Repeated records are collapsed to one `(screen_id, lnp_id)` row.
The current aggregation retains the first delivery value, uses medians for
numeric covariates, and retains available categorical metadata. It does not
average repeated delivery outcomes. Confirm that this retention rule matches
the intended handling of replicate outcomes before describing it as replicate
aggregation.

Percentiles use ascending average ranks: `p_i = (r_i - 1) / (n - 1)` for
`n > 1`; a singleton is assigned 1. Ties receive identical average ranks and
therefore identical percentiles. In tied screens, minimum/maximum percentiles
need not be exactly 0/1.

For the top-10% flag, set `k = max(1, ceil(0.10 * n))` and take the k-th largest
delivery value as the cutoff. All values greater than or equal to this cutoff
are selected. Boundary ties are retained in full, so the selected fraction may
exceed 10%, particularly for discretized outcomes. This is not equivalent to
requiring percentile >= 0.9 when ties are present. The saved top-performer QC
table records the realized selected fraction per screen.

Suggested manuscript text:

> Within each screen, formulations were ranked using average ranks for tied
> delivery outcomes, and ranks were rescaled as (rank - 1)/(n - 1). Top performers
> were defined using the delivery value at rank ceil(0.10 n) in descending order
> as the cutoff. All formulations tied at the cutoff were retained; consequently,
> the selected fraction could exceed 10% in screens with discretized outcomes.

### Items to Resolve Before Submission

1. Supply the upstream normalization equation, reference population, and an
   endpoint-to-direction mapping. Neither downstream notebook implements a
   lower-is-better reversal. They assume larger `delivery` means better delivery.
   A residual-expression endpoint cannot be treated like reporter expression
   unless its direction was already harmonized upstream. Do not claim this
   happened without the curation code or an audit table.
2. Verify the final figure's actual result folders and saved configurations.
   Original five-property zero-fill and selected-three-property median
   imputation are distinct experiments. Removing a bar from a plot does not
   change the experiment or validate its preprocessing.
3. Verify that the selected-physchem source preserves raw NaNs: its loader can
   reuse a live `source_df` or a previously exported source. Once missing values
   have been replaced by zero, median imputation cannot recover their identity.
   The current helper includes a missingness indicator if any partition has a
   missing value; for strict train-only schema selection, this requires review.
4. Release the harmonization mapping, balance audit, exact split IDs, context
   maps, feature schemas, imputation medians, per-group class counts/predictions,
   metric summaries including valid-score counts, and model metadata. Archive
   the code revision, dependency versions, pretrained checkpoint revisions,
   third-party encoder code, and resolved effective hyperparameters.
5. Check cross-partition overlaps by canonical lipid SMILES and LNP ID to report
   actual counts. No numerical overlap counts are asserted by this draft.
6. A context-only baseline was removed from the primary input-level section
   earlier; the present code does not establish that the PI-requested baseline
   was evaluated. Do not describe it as completed without separate results.
