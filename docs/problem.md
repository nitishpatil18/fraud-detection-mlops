# fraud detection mlops project

## problem

predict whether a transaction is fraudulent given transaction and identity features.

## business metric

- false negatives are expensive: fraud slips through, chargebacks cost merchant.
- false positives are annoying: legit customer gets declined, churn risk.
- typical business tradeoff: catch as much fraud as possible while keeping false positive rate below a threshold (e.g. block rate < 2%).

## target metric

- pr-auc (precision-recall area under curve) as the primary metric.
- recall at fixed precision (e.g. recall at precision=0.5) as a business-facing metric.
- accuracy is useless here due to 3.5% class imbalance.

## dataset facts

- ieee-cis fraud dataset (kaggle 2019)
- 590,540 transactions, 394 columns
- target: `isFraud`, 3.5% positive rate
- time span: 182 days
- identity table joinable by `TransactionID`, covers 24% of transactions
- 174 of 394 columns are more than 50% missing
- categorical features (`ProductCD`, `card1-6`, `P_emaildomain`) carry strong signal
- `TransactionAmt` has non-linear u-shape relationship with fraud
- fraud rate drifts upward over 182 days (natural concept drift)

## modeling decisions

- time-based train/val/test split, not random (this is time-series data)
- tree-based models (xgboost) as primary choice
- missingness will be treated as a feature, not imputed away
- class imbalance handled via `scale_pos_weight` in xgboost, not resampling

## scope

- build end-to-end ml system: data pipeline, training, experiment tracking,
  serving api, monitoring, ci/cd.
- not building: state-of-the-art model, full competition leaderboard submission.