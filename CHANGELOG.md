# Changelog

## 27–28 September 2026

### Independent evaluation
- **Form pre-filling (J4)**: evaluated on 21 unseen Wikimedia Commons photos, annotated by a human.
  Accuracy 71% [62–80%], pre-filled 61% [45–73%] (bootstrap over photos).
  Notebook: `J4_evaluation.ipynb`, ground truth: `eval_j4/annotations_miryame.csv`.
- **Scene check (J2, CLIP)**: tested on 66 unseen photos with the J2 threshold unchanged.
  Streams accepted 92% [79–97%], off-topic images rejected 79% [62–90%].
  Wide, calm rivers are sometimes taken for lakes. Notebook: `J2_evaluation_clip.ipynb`.
- **Litter detector (J3, YOLO)**: the external test showed that lowering the threshold barely helps
  (best F1 0.557 vs 0.548), so the 0.25 threshold is kept. Recall stays low (0.43).

### What we changed after evaluation
- Pre-filling is **disabled** for channel shape, vegetation (left and right) and overall condition:
  their accuracy was not distinguishable from chance. The citizen answers them.
- Left and right banks are judged together, so both sides follow the same rule.

### Citizen interface (`app.py`)
- Guided form in 6 steps, like the OneAquaHealth app, with the annotated photo always visible.
- A tag under each answer shows its origin: suggested by the AI (with confidence and reason),
  changed by the citizen, or for the citizen to answer.
- Complete form: habitats, natural debris, water withdrawal, water height, invasive species, feelings.
  Vegetation type is only asked when vegetation is present, as in the app.

### Risk score
- Each rule is traced to the OneAquaHealth Key Indicators Factsheets
  (Schmeller et al., 2026, doi:10.5281/zenodo.20345207, CC-BY 4.0) or labelled as a team assumption.
  Scores are unchanged. The factsheets give no thresholds for photo observations.
- New "Why this level?" table for citizens and managers: rule, points, evidence.

### FHIR export (`fhir_export.py`)
- Answers also carry the codes used by the OneAquaHealth app (e.g. `U_SHAPE` → `U`, `SLOW` → `NOR`).
- Follow-up questions are only exported when the app would ask them.
