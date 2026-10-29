# New analyses: saved-result report

## eICU site-level calibration

- These are binary-endpoint ECE values and are not numerically comparable with the three-class ECE in the primary results.

| Endpoint | Model | Method | n | Events | AUROC | ECE | Slope |
|---|---|---|---:|---:|---|---:|---:|
| any_deterioration | xgb_summary | No calibration | 1113.0 | 326.0 | 0.813 | 0.054 | 1.051 |
| any_deterioration | xgb_summary | Intercept update | 1113.0 | 326.0 | 0.814 | 0.029 | 1.035 |
| any_deterioration | xgb_summary | Slope and intercept | 1113.0 | 326.0 | 0.808 | 0.026 | 0.900 |
| any_deterioration | xgb_summary | Flexible logistic recalibration | 1113.0 | 326.0 | 0.783 | 0.042 | 0.286 |
| any_deterioration | xgb_summary | Empirical-Bayes shrinkage | 1113.0 | 326.0 | 0.812 | 0.037 | 0.992 |
| any_deterioration | dggru_tr | No calibration | 1113.0 | 326.0 | 0.807 | 0.082 | 0.971 |
| any_deterioration | dggru_tr | Intercept update | 1113.0 | 326.0 | 0.814 | 0.034 | 1.019 |
| any_deterioration | dggru_tr | Slope and intercept | 1113.0 | 326.0 | 0.811 | 0.032 | 0.915 |
| any_deterioration | dggru_tr | Flexible logistic recalibration | 1113.0 | 326.0 | 0.784 | 0.040 | 0.284 |
| any_deterioration | dggru_tr | Empirical-Bayes shrinkage | 1113.0 | 326.0 | 0.806 | 0.036 | 0.991 |
| severe_deterioration | xgb_summary | No calibration | 779.0 | 141.0 | 0.787 | 0.035 | 1.040 |
| severe_deterioration | xgb_summary | Intercept update | 779.0 | 141.0 | 0.788 | 0.023 | 1.014 |
| severe_deterioration | xgb_summary | Slope and intercept | 779.0 | 141.0 | 0.784 | 0.028 | 0.897 |
| severe_deterioration | xgb_summary | Flexible logistic recalibration | 779.0 | 141.0 | 0.760 | 0.037 | 0.293 |
| severe_deterioration | xgb_summary | Empirical-Bayes shrinkage | 779.0 | 141.0 | 0.785 | 0.024 | 0.982 |
| severe_deterioration | dggru_tr | No calibration | 779.0 | 141.0 | 0.781 | 0.051 | 0.936 |
| severe_deterioration | dggru_tr | Intercept update | 779.0 | 141.0 | 0.782 | 0.032 | 0.932 |
| severe_deterioration | dggru_tr | Slope and intercept | 779.0 | 141.0 | 0.777 | 0.029 | 0.895 |
| severe_deterioration | dggru_tr | Flexible logistic recalibration | 779.0 | 141.0 | 0.744 | 0.043 | 0.302 |
| severe_deterioration | dggru_tr | Empirical-Bayes shrinkage | 779.0 | 141.0 | 0.778 | 0.030 | 0.975 |

## 6/24/48 h landmark models

| Landmark | n | Grade 0/1/2 | Model | Macro-AUC | Grade 1 AUC | Calibration slope | ECE any |
|---:|---:|---|---|---|---:|---:|---:|
| 6 h | 3236 | 2421/255/560 | Summary XGBoost | 0.751 (0.734-0.768) | 0.724 | 1.009 | 0.016 |
| 6 h | 3236 | 2421/255/560 | GRU | 0.742 (0.723-0.760) | 0.707 | 0.910 | 0.022 |
| 6 h | 3236 | 2421/255/560 | DG-GRU | 0.739 (0.721-0.757) | 0.702 | 0.848 | 0.023 |
| 6 h | 3236 | 2421/255/560 | DG-GRU + TR | 0.741 (0.722-0.758) | 0.706 | 0.911 | 0.018 |
| 24 h | 2673 | 2031/243/399 | Summary XGBoost | 0.800 (0.782-0.817) | 0.782 | 0.997 | 0.019 |
| 24 h | 2673 | 2031/243/399 | GRU | 0.795 (0.778-0.812) | 0.768 | 0.916 | 0.020 |
| 24 h | 2673 | 2031/243/399 | DG-GRU | 0.792 (0.774-0.809) | 0.768 | 0.836 | 0.028 |
| 24 h | 2673 | 2031/243/399 | DG-GRU + TR | 0.796 (0.778-0.813) | 0.773 | 0.908 | 0.021 |
| 48 h | 2009 | 1645/151/213 | Summary XGBoost | 0.849 (0.830-0.868) | 0.812 | 1.030 | 0.010 |
| 48 h | 2009 | 1645/151/213 | GRU | 0.852 (0.831-0.873) | 0.809 | 0.904 | 0.019 |
| 48 h | 2009 | 1645/151/213 | DG-GRU | 0.854 (0.832-0.875) | 0.816 | 0.844 | 0.022 |
| 48 h | 2009 | 1645/151/213 | DG-GRU + TR | 0.854 (0.833-0.875) | 0.816 | 0.901 | 0.021 |

## Sensitivity analyses

| Dataset | Scenario | Support | Protocol | n | Grade 0/1/2 | Model | Macro-AUC | Grade 1 AUC | Slope | ECE any |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| mimic4 | Maximum pre-landmark GCS | supported | valid | 2673 | 2031/243/399 | Summary XGBoost | 0.798 | 0.781 | 1.032 | 0.019 |
| mimic4 | Maximum pre-landmark GCS | supported | valid | 2673 | 2031/243/399 | DG-GRU + TR | 0.793 | 0.773 | 0.929 | 0.014 |
| mimic4 | Final pre-landmark GCS | supported | valid | 2673 | 2276/132/265 | Summary XGBoost | 0.732 | 0.744 | 0.873 | 0.017 |
| mimic4 | Final pre-landmark GCS | supported | valid | 2673 | 2276/132/265 | DG-GRU + TR | 0.732 | 0.735 | 0.916 | 0.007 |
| mimic4 | At least one low hourly observation | supported | valid | 2673 | 1517/540/616 | Summary XGBoost | 0.785 | 0.739 | 1.068 | 0.021 |
| mimic4 | At least one low hourly observation | supported | valid | 2673 | 1517/540/616 | DG-GRU + TR | 0.778 | 0.720 | 0.926 | 0.023 |
| mimic4 | At least two consecutive hourly hours | supported | valid | 2673 | 2031/243/399 | Summary XGBoost | 0.798 | 0.781 | 1.032 | 0.019 |
| mimic4 | At least two consecutive hourly hours | supported | valid | 2673 | 2031/243/399 | DG-GRU + TR | 0.793 | 0.773 | 0.929 | 0.014 |
| mimic4 | Exclude GCS observations in continuous sedation windows | supported | valid | 2670 | 2043/240/387 | Summary XGBoost | 0.798 | 0.782 | 1.009 | 0.019 |
| mimic4 | Exclude GCS observations in continuous sedation windows | supported | valid | 2670 | 2043/240/387 | DG-GRU + TR | 0.796 | 0.776 | 0.909 | 0.020 |
| mimic4 | Restrict to the unsedated subgroup | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| mimic4 | Restrict to the unsedated subgroup | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic4 | Exclude early live ICU transfers | supported | valid | 1830 | 1264/197/369 | Summary XGBoost | 0.755 | 0.723 | 0.979 | 0.017 |
| mimic4 | Exclude early live ICU transfers | supported | valid | 1830 | 1264/197/369 | DG-GRU + TR | 0.753 | 0.729 | 0.882 | 0.027 |
| mimic4 | Code early live ICU transfers as grade 0 | supported | valid | 2827 | 2261/197/369 | Summary XGBoost | 0.798 | 0.770 | 1.002 | 0.010 |
| mimic4 | Code early live ICU transfers as grade 0 | supported | valid | 2827 | 2261/197/369 | DG-GRU + TR | 0.794 | 0.768 | 0.938 | 0.011 |
| mimic4 | Inverse-probability weighting for early ICU transfer | supported | valid | 2827 | 2261/197/369 | Summary XGBoost | 0.805 | 0.770 | 1.002 | 0.007 |
| mimic4 | Inverse-probability weighting for early ICU transfer | supported | valid | 2827 | 2261/197/369 | DG-GRU + TR | 0.795 | 0.754 | 0.940 | 0.014 |
| mimic4 | Composite endpoint includes unplanned intubation | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| mimic4 | Composite endpoint includes unplanned intubation | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic4 | GCS/death endpoint excludes unplanned intubation | supported | valid | 2673 | 2031/243/399 | Summary XGBoost | 0.798 | 0.781 | 1.032 | 0.019 |
| mimic4 | GCS/death endpoint excludes unplanned intubation | supported | valid | 2673 | 2031/243/399 | DG-GRU + TR | 0.793 | 0.773 | 0.929 | 0.014 |
| mimic4 | 6 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| mimic4 | 6 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| mimic4 | 24 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| mimic4 | 24 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| mimic4 | 48 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| mimic4 | 48 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| mimic4 | Exclude deaths within 72 h | supported | valid | 2545 | 2031/243/271 | Summary XGBoost | 0.806 | 0.792 | 0.998 | 0.016 |
| mimic4 | Exclude deaths within 72 h | supported | valid | 2545 | 2031/243/271 | DG-GRU + TR | 0.809 | 0.777 | 0.940 | 0.019 |
| mimic4 | eICU GCS without RASS adjustment | not_applicable |  |  |  | Summary XGBoost |  |  |  |  |
| mimic4 | eICU GCS without RASS adjustment | not_applicable |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic4 | eICU GCS with RASS adjustment | not_applicable |  |  |  | Summary XGBoost |  |  |  |  |
| mimic4 | eICU GCS with RASS adjustment | not_applicable |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | Maximum pre-landmark GCS | supported | valid | 1136 | 756/123/257 | Summary XGBoost | 0.765 | 0.754 | 0.976 | 0.023 |
| mimic3 | Maximum pre-landmark GCS | supported | valid | 1136 | 756/123/257 | DG-GRU + TR | 0.758 | 0.731 | 0.859 | 0.022 |
| mimic3 | Final pre-landmark GCS | supported | valid | 1136 | 886/81/169 | Summary XGBoost | 0.722 | 0.756 | 0.802 | 0.032 |
| mimic3 | Final pre-landmark GCS | supported | valid | 1136 | 886/81/169 | DG-GRU + TR | 0.700 | 0.711 | 0.758 | 0.023 |
| mimic3 | At least one low hourly observation | supported | valid | 1136 | 531/234/371 | Summary XGBoost | 0.756 | 0.690 | 0.999 | 0.025 |
| mimic3 | At least one low hourly observation | supported | valid | 1136 | 531/234/371 | DG-GRU + TR | 0.743 | 0.643 | 0.907 | 0.031 |
| mimic3 | At least two consecutive hourly hours | supported | valid | 1136 | 756/123/257 | Summary XGBoost | 0.765 | 0.754 | 0.976 | 0.023 |
| mimic3 | At least two consecutive hourly hours | supported | valid | 1136 | 756/123/257 | DG-GRU + TR | 0.758 | 0.731 | 0.859 | 0.022 |
| mimic3 | Exclude GCS observations in continuous sedation windows | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | Exclude GCS observations in continuous sedation windows | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | Restrict to the unsedated subgroup | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | Restrict to the unsedated subgroup | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | Exclude early live ICU transfers | supported | valid | 709 | 385/100/224 | Summary XGBoost | 0.704 | 0.694 | 0.849 | 0.030 |
| mimic3 | Exclude early live ICU transfers | supported | valid | 709 | 385/100/224 | DG-GRU + TR | 0.702 | 0.686 | 0.780 | 0.035 |
| mimic3 | Code early live ICU transfers as grade 0 | supported | valid | 1232 | 908/100/224 | Summary XGBoost | 0.784 | 0.774 | 0.950 | 0.029 |
| mimic3 | Code early live ICU transfers as grade 0 | supported | valid | 1232 | 908/100/224 | DG-GRU + TR | 0.779 | 0.778 | 0.873 | 0.023 |
| mimic3 | Inverse-probability weighting for early ICU transfer | supported | valid | 1232 | 908/100/224 | Summary XGBoost | 0.798 | 0.778 | 0.947 | 0.019 |
| mimic3 | Inverse-probability weighting for early ICU transfer | supported | valid | 1232 | 908/100/224 | DG-GRU + TR | 0.795 | 0.783 | 0.891 | 0.022 |
| mimic3 | Composite endpoint includes unplanned intubation | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | Composite endpoint includes unplanned intubation | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | GCS/death endpoint excludes unplanned intubation | supported | valid | 1136 | 756/123/257 | Summary XGBoost | 0.765 | 0.754 | 0.976 | 0.023 |
| mimic3 | GCS/death endpoint excludes unplanned intubation | supported | valid | 1136 | 756/123/257 | DG-GRU + TR | 0.758 | 0.731 | 0.859 | 0.022 |
| mimic3 | 6 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | 6 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | 24 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | 24 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | 48 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | 48 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | Exclude deaths within 72 h | supported | valid | 1048 | 756/123/169 | Summary XGBoost | 0.783 | 0.768 | 0.934 | 0.019 |
| mimic3 | Exclude deaths within 72 h | supported | valid | 1048 | 756/123/169 | DG-GRU + TR | 0.783 | 0.751 | 0.985 | 0.029 |
| mimic3 | eICU GCS without RASS adjustment | not_applicable |  |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | eICU GCS without RASS adjustment | not_applicable |  |  |  | DG-GRU + TR |  |  |  |  |
| mimic3 | eICU GCS with RASS adjustment | not_applicable |  |  |  | Summary XGBoost |  |  |  |  |
| mimic3 | eICU GCS with RASS adjustment | not_applicable |  |  |  | DG-GRU + TR |  |  |  |  |
| eicu | Maximum pre-landmark GCS | supported | valid | 1725 | 1315/156/254 | Summary XGBoost | 0.799 | 0.760 | 0.978 | 0.018 |
| eicu | Maximum pre-landmark GCS | supported | valid | 1725 | 1315/156/254 | DG-GRU + TR | 0.790 | 0.738 | 0.934 | 0.021 |
| eicu | Final pre-landmark GCS | supported | valid | 1725 | 1467/96/162 | Summary XGBoost | 0.690 | 0.643 | 0.817 | 0.028 |
| eicu | Final pre-landmark GCS | supported | valid | 1725 | 1467/96/162 | DG-GRU + TR | 0.679 | 0.652 | 0.778 | 0.012 |
| eicu | At least one low hourly observation | supported | valid | 1725 | 1086/292/347 | Summary XGBoost | 0.784 | 0.734 | 1.026 | 0.016 |
| eicu | At least one low hourly observation | supported | valid | 1725 | 1086/292/347 | DG-GRU + TR | 0.770 | 0.709 | 0.896 | 0.030 |
| eicu | At least two consecutive hourly hours | supported | valid | 1725 | 1315/156/254 | Summary XGBoost | 0.799 | 0.760 | 0.978 | 0.018 |
| eicu | At least two consecutive hourly hours | supported | valid | 1725 | 1315/156/254 | DG-GRU + TR | 0.790 | 0.738 | 0.934 | 0.021 |
| eicu | Exclude GCS observations in continuous sedation windows | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| eicu | Exclude GCS observations in continuous sedation windows | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| eicu | Restrict to the unsedated subgroup | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| eicu | Restrict to the unsedated subgroup | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| eicu | Exclude early live ICU transfers | supported | valid | 973 | 636/120/217 | Summary XGBoost | 0.768 | 0.722 | 0.971 | 0.037 |
| eicu | Exclude early live ICU transfers | supported | valid | 973 | 636/120/217 | DG-GRU + TR | 0.743 | 0.677 | 0.830 | 0.036 |
| eicu | Code early live ICU transfers as grade 0 | supported | valid | 2130 | 1793/120/217 | Summary XGBoost | 0.838 | 0.814 | 0.997 | 0.014 |
| eicu | Code early live ICU transfers as grade 0 | supported | valid | 2130 | 1793/120/217 | DG-GRU + TR | 0.823 | 0.781 | 0.977 | 0.016 |
| eicu | Inverse-probability weighting for early ICU transfer | supported | valid | 2130 | 1793/120/217 | Summary XGBoost | 0.838 | 0.806 | 0.989 | 0.014 |
| eicu | Inverse-probability weighting for early ICU transfer | supported | valid | 2130 | 1793/120/217 | DG-GRU + TR | 0.814 | 0.770 | 0.936 | 0.015 |
| eicu | Composite endpoint includes unplanned intubation | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| eicu | Composite endpoint includes unplanned intubation | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
| eicu | GCS/death endpoint excludes unplanned intubation | supported | valid | 1725 | 1315/156/254 | Summary XGBoost | 0.799 | 0.760 | 0.978 | 0.018 |
| eicu | GCS/death endpoint excludes unplanned intubation | supported | valid | 1725 | 1315/156/254 | DG-GRU + TR | 0.790 | 0.738 | 0.934 | 0.021 |
| eicu | 6 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| eicu | 6 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| eicu | 24 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| eicu | 24 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| eicu | 48 h landmark | separate analysis | reported_elsewhere |  |  | Summary XGBoost |  |  |  |  |
| eicu | 48 h landmark | separate analysis | reported_elsewhere |  |  | DG-GRU + TR |  |  |  |  |
| eicu | Exclude deaths within 72 h | supported | valid | 1653 | 1315/156/182 | Summary XGBoost | 0.804 | 0.771 | 0.979 | 0.016 |
| eicu | Exclude deaths within 72 h | supported | valid | 1653 | 1315/156/182 | DG-GRU + TR | 0.800 | 0.758 | 0.895 | 0.029 |
| eicu | eICU GCS without RASS adjustment | supported | valid | 1725 | 1315/156/254 | Summary XGBoost | 0.799 | 0.760 | 0.978 | 0.018 |
| eicu | eICU GCS without RASS adjustment | supported | valid | 1725 | 1315/156/254 | DG-GRU + TR | 0.790 | 0.738 | 0.934 | 0.021 |
| eicu | eICU GCS with RASS adjustment | unsupported |  |  |  | Summary XGBoost |  |  |  |  |
| eicu | eICU GCS with RASS adjustment | unsupported |  |  |  | DG-GRU + TR |  |  |  |  |
