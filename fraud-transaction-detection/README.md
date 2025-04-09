# Fraud Detection with Optimized Random Forest

A machine learning model to detect fraudulent transactions in financial simulations, achieving 99.9% recall using feature engineering and class balancing.

---

## 🖥️ Recommended Environment

- Run on **Google Colab** for easy setup  
- Set hardware accelerator to **TPU (T4)** via:  
  `Runtime > Change runtime type > Hardware Accelerator > TPU`  
- ✅ Faster training & inference for large datasets


---

## 📁 Dataset
- **Source:** [Original Copy](https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&confirm=t&uuid=0bbb9778-f9c3-4df6-9f40-b7d18775eb74)  
- **Backup Mirror:** [Google Drive](https://drive.google.com/file/d/1bGz0X-ep633QdYV_bMJf54idp4B-_9k-/view?usp=drive_link) (if primary fails)  


---

### Dataset Overview
**6,362,620 transactions** | **11 features** | **30-day simulation (744 hourly steps)**

| # | Column            | Dtype   | Nulls | Description                                                                 | Fraud Pattern                                                                 |
|---|-------------------|---------|-------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 0 | `step`            | `int64` | 0     | Hourly timestep (1-744)                                                     | Fraud clusters in specific hours                                              |
| 1 | `type`            | `object`| 0     | `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`                       | Only `TRANSFER`→`CASH-OUT` sequences are fraudulent                          |
| 2 | `amount`          | `float64`| 0    | Transaction value                                                           | High amounts common but not definitive                                        |
| 3 | `nameOrig`        | `object`| 0     | Sender ID (`C`=customer, `M`=merchant)                                      | Compromised accounts start with `C`                                           |
| 4 | `oldbalanceOrg`   | `float64`| 0    | Sender pre-transaction balance                                              | Fraud: `amount ≈ oldbalanceOrg` (account draining)                            |
| 5 | `newbalanceOrig`  | `float64`| 0    | Sender post-transaction balance                                             | `newbalanceOrig = 0` after transfer indicates fraud                           |
| 6 | `nameDest`        | `object`| 0     | Recipient ID                                                                | `M` recipients are legitimate merchants                                       |
| 7 | `oldbalanceDest`  | `float64`| 0    | Recipient pre-transaction balance                                           | Often `0` for fraud (new accounts)                                            |
| 8 | `newbalanceDest`  | `float64`| 0    | Recipient post-transaction balance                                          | Rapid cash-outs after transfer                                                |
| 9 | `isFraud`         | `int64` | 0     | **Target** (1=fraud)                                                        | Only 0.13% positive class                                                     |
|10 | `isFlaggedFraud`  | `int64` | 0     | Business rule flag (transfer >200K)                                         | Misses 99.9% of fraud cases                                                   |

**Key Insights:**
- ⚠️ **Fraud rate**: 8,213 cases (0.13%) - extreme class imbalance
- 🔍 **Fraud mechanism**: Account takeover → `TRANSFER` full balance → `CASH-OUT`
- 📉 **Business rule failure**: `isFlaggedFraud` only catches 16/8213 fraud cases
---
