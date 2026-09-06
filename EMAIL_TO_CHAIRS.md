# Draft email to the competition co-chairs

**To:** imtiaz.ahmed@mail.wvu.edu; zimo.wang@binghamton.edu; shixianz@andrew.cmu.edu
**Subject:** INFORMS 2026 DM Data Challenge — evaluation metric

---

Dear Competition Co-Chairs,

We are a registered team working on the 2026 Data Mining Society Data Challenge. The
challenge announcement states that "the specific evaluation metrics and scoring details
will be communicated to registered participants via the Google Drive folder," and we
have not been able to find this information in the folder's current contents.

Could you confirm (a) which metric will be used to score the four horizon columns, and
(b) how the four horizons are combined into a single ranking? The 2025 challenge used
RMSE with the average rank across horizons; knowing whether 2026 follows the same
approach affects how we balance accuracy across the four horizons.

If there is a document in the folder that we have missed, we would be grateful for a
copy.

Thank you for organising the challenge.

Kind regards,

[Your name]
[Team name]
[Institution]

---

## Internal note (not part of the email)

**Why this is the only question worth asking.** Everything else we considered is either
already answered in the data package or does not block us:

- *Submission instructions* — `problem_description.pdf` §"Submission Requirements"
  specifies the folder name (`TeamName_Submission/`), the three components
  (filled template, `code.zip`, `report.pdf` ≤ 6 pages), the Drive-link-via-Google-Form
  mechanism and the deadline. Complete.
- *External data* — `weather_variables.pdf` lists EIA Form 861, NLCD tree canopy and the
  USDA rural–urban continuum codes under "Infrastructure features (static, optional
  enrichment)" with source links, which sanctions the static county attributes we use.
- *Cross-county causality* — the Problem Description's rule table allows "any outage
  feature value at timestamp ≤ t". We tested the permissive reading; it produced no
  measurable gain (regional outage state is redundant once full future weather is
  given), so our submission is compliant under either interpretation.

**What changes with the answer.** We currently optimise per-horizon RMSE averaged across
the four horizons — the 2025 precedent. If the metric is MAE instead, we re-weight the
ensemble toward the Tweedie/sqrt members (LightGBM-Tweedie MAE 0.00249, sqrt+ExtraTrees
0.00248) using the cached out-of-fold predictions: about ten minutes, no retraining. If
ranking is by *average rank* rather than average error, consistency across all four
horizons matters more than a strong single horizon — our current ensemble is at or near
the top on all four simultaneously, which is the right profile for that rule.
