# **AML & Fraud Detection Hybrid System**

## Overview

This project implements a **hybrid Anti-Money Laundering (AML) and fraud detection system** using a combination of **rule-based detection, machine learning, and anomaly detection**. It is designed to identify **suspicious transactions and high-risk accounts** in financial transaction datasets, with a focus on both detection accuracy and explainability.

The system leverages the **PaySim dataset** to simulate realistic fintech transactions and provides a framework for both **ML-driven anomaly detection** and **AML rule enforcement**, suitable for portfolio projects, dashboards, or cybersecurity-oriented reporting.

---

## Features

* **AML Rule Engine**

  * Detects money-laundering patterns such as **structuring**, **layering**, and **velocity** of transactions.
  * Assigns **rule scores** per transaction for risk evaluation.

* **Machine Learning-Based Detection**

  * Supervised ML models such as **Random Forest** and **XGBoost** to identify suspicious transactions.
  * **Anomaly detection models** (Isolation Forest, One-Class SVM, Autoencoders) to detect rare or zero-day fraud cases.

* **Hybrid Risk Scoring**

  * Combines **ML probabilities** and **AML rule scores**.
  * Ensures that suspicious transactions flagged by rules but missed by ML are still captured.

* **Feature Engineering**

  * AML-focused features:

    * Transaction velocity per account
    * Aggregated sums over past N transactions
    * Ratio features for sender/receiver patterns
  * Time-based and amount-based features for improved ML detection.

* **Class Imbalance Handling**

  * Techniques include **SMOTE**, **RandomOverSampler**, and **adjusted class weights**.
  * Improves recall and detection of rare fraud events.

* **Graph Analytics**

  * Builds a **transaction network** where nodes are accounts and edges are transactions.
  * Identifies **high-risk hubs and clusters**, aiding in investigation of suspicious networks.

---
### Prerequisites

* Python 3.8+
* Libraries:

  ```bash
  pandas, numpy, scikit-learn, imbalanced-learn, networkx, matplotlib, seaborn
  ```

### Dataset

* Use **PaySim dataset**: realistic fintech transaction simulator.
* Place the dataset in `/data/raw/paysim.csv`.

---
## Evaluation Results (Current)
| Metric                      | Value |
| --------------------------- | ----- |
| **Precision (fraud class)** | 0.88  |
| **Recall (fraud class)**    | 0.29  |
| **F1-Score (fraud class)**  | 0.44  |
| **ROC AUC**                 | 0.956 |


Top suspicious accounts (degree centrality):
[('C985934102', 0.00042), ('C1286084959', 0.00041), ('C248609774', 0.00039),
 ('C1590550415', 0.00039), ('C2083562754', 0.00037), ('C977993101', 0.00036),
 ('C665576141', 0.00035), ('C1360767589', 0.00034), ('C451111351', 0.00033),
 ('C1023714065', 0.00032)]

---
## Interpretation

1. Precision is high — most flagged transactions are correct.
2. Recall is low — many fraudulent transactions are missed due to class imbalance.
3. ROC-AUC shows good separation between fraud and normal transactions.
4. Top hubs indicate accounts potentially involved in layering or structuring networks, useful for AML investigation.
---
## Improvement
### Handle Class Imbalance Better
1. Use SMOTE or RandomOverSampler to upsample fraud class.
2. Tune class weights more aggressively in ML models.
### Try different Anomaly detection model 
### Hybrid Scoring
---

## Next Steps / Extensions

* Integrate **real-time transaction streams** for online AML monitoring.
* Implement **dashboard visualization** using Streamlit or Plotly.
* Add **community detection** and network anomaly alerts for cybercrime investigation.
* Experiment with **advanced ML models** (XGBoost, LightGBM) and **graph neural networks**.

---

## License

MIT License
