# Normative DWI

Scoring one person's diffusion microstructure against voxelwise normative models of the UK Biobank population.

## Inputs

**Marker**:
One diffusion microstructure measure with its own normative models: FA, MD (from DTI), ICVF, ISOVF or OD (from NODDI).
_Avoid_: modality, metric, measure, response variable

**Map**:
One subject's image of one marker, already in template space and harmonised.
_Avoid_: scan, volume, image (alone)

**Template space**:
The UK Biobank 2 mm grid (97×115×97) that every map and every model shares.
_Avoid_: UKB space, MNI space, standard space

**Harmonised**:
Said of a map whose site and scanner effects have already been removed so that it is comparable with UK Biobank data (e.g. by ComBat).
_Avoid_: corrected, normalised

**Background image**:
A structural image of the subject (T1-weighted or FLAIR), registered to template space at any resolution, shown under a z-map for display only.
_Avoid_: anatomy, underlay, template

**Covariates**:
The subject's age and sex, the only facts about the person the models condition on.
_Avoid_: demographics, metadata

## Models

**Store**:
The published set of trained voxel models for every marker, together with the template-space grid they live on.
_Avoid_: model file, database, zarr (alone)

**Voxel model**:
The fitted normative distribution of one marker at one voxel, as a function of the covariates.
_Avoid_: regression, fit, BLR (alone)

**Modelled voxel**:
A voxel of template space for which the store holds a voxel model of the marker in question.
_Avoid_: valid voxel, in-mask voxel

**Training range**:
The span of ages a voxel model was fitted on; ages outside it are scored at its nearest end.
_Avoid_: covariate range, support

## Outputs

**Z-score**:
How many standard deviations a map's value lies from the voxel model's expectation for the subject's covariates.
_Avoid_: deviation, deviation score, centile

**Z-map**:
The image of z-scores for one map, zero wherever the voxel is not modelled or the map has no value.
_Avoid_: deviation map, result
