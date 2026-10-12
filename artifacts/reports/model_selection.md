# Model Selection Summary

**Selected model:** Random Forest

**Rationale:** Random Forest selected using held-out ROC-AUC as the primary ranking metric, with precision/recall/F1 and cross-validated ROC-AUC reviewed for robustness.

```
                     accuracy  precision  recall      f1  roc_auc  pr_auc  cv_roc_auc
Random Forest          0.8197     0.7750  0.9394  0.8493   0.8950  0.9161      0.8992
Gradient Boosting      0.7869     0.7632  0.8788  0.8169   0.8929  0.9121      0.8972
Logistic Regression    0.7869     0.7632  0.8788  0.8169   0.8734  0.8903      0.8947
SVM (RBF)              0.8197     0.7619  0.9697  0.8533   0.8615  0.8732      0.8915
```
