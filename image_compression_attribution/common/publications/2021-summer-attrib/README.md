# Overview

Notebooks [exp_01.ipynb](exp_01.ipynb) through [exp_04.ipynb](exp_04.ipynb) perform four different image attribution experiments, whose results are reported in the paper:

1. Source verification: The algorithm verifies if an image comes from a purported source, or not.
1. Verification stability over time: does performance persist over time?
1. Verification performance difference between closed-set and open-set conditions.
1. Source identification: (N-class classification)

Notebooks [exp_05-1-v2.ipynb](exp_05-1-v2.ipynb) through [exp_08-4.ipynb](exp_08-4.ipynb) repeat and extend experiment 1 - 4 by including
additional classifier methods, like XGBoost and IsolationForests, in addition to the Naive Bayes 
model from notebooks  1 - 4.  They also compare additional formats for features.

Results are generated and stored in a folder called `results`.
