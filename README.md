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

## Next Steps / Extensions

* Integrate **real-time transaction streams** for online AML monitoring.
* Implement **dashboard visualization** using Streamlit or Plotly.
* Add **community detection** and network anomaly alerts for cybercrime investigation.
* Experiment with **advanced ML models** (XGBoost, LightGBM) and **graph neural networks**.

---

## License

MIT License
