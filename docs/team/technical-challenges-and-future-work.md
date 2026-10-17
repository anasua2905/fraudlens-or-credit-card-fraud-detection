# Technical Challenges and Future Work (Assessment 3)

Author: Syed Ittisaf Tazwar

## Technical challenges

| Challenge | Problem | How it was handled |
|---|---|---|
| Extreme class imbalance | About 1 fraud per 172 transactions, so accuracy is misleading | Class weights on the training split only; PR-AUC as the main metric |
| Leakage across time | A random split or SMOTE lets the model see future data | Chronological train, validation and test split; test set used once |
| Two models almost tied | Gradient boosting and the hybrid differ by 0.0001 validation PR-AUC | Pre-set selection rule using a 95% interval on the gap |
| Alert volume | Catching 99.3% of fraud needs about 46 alerts a day | Cost-aware threshold: about 95% recall at roughly 12 alerts a day |

## Open limitation

The dataset is simulated. The method transfers to real transactions, but the absolute scores will not.

## Future work before Assessment 4

1. Lock the final model and threshold.
2. Check performance month by month across the test period.
3. Finish the dashboard for analyst use.
4. Complete the final report and run guide.
