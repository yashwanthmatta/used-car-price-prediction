# Used Car Price Prediction — Machine Learning & Explainable AI

> **End-to-end predictive analytics pipeline** comparing 5 ML models for used car price estimation, with a LangChain + GPT-4 explanation module and Streamlit web app — built on real Indian automotive market data.

---

## Project Overview

Pricing a used car accurately is hard. Buyers overpay. Sellers undervalue. Dealerships rely on gut feeling. This project replaces that guesswork with a **data-driven pricing engine** trained on thousands of real vehicle listings — and goes one step further by explaining *why* it predicted that price in plain English.

**Course:** BANA 6660 – Predictive Analytics | University of Colorado Denver
**Team:** Yashwanth Goud Matta · Naik Satti Karthik
**Submission:** May 2025

---

## Results at a Glance

| Model | RMSE | R² | Notes |
|---|---|---|---|
| k-Nearest Neighbors | **0.41** | **0.83** | Best model — captures non-linear pricing patterns |
| CART (Decision Tree) | 0.67 | 0.75 | Best interpretability — clear variable importance |
| Ridge Regression | 0.79 | 0.71 | Handles multicollinearity well |
| Lasso Regression | 0.81 | 0.70 | Feature selection via regularization |
| Multiple Linear Regression | 0.84 | 0.68 | Baseline — limited by linearity assumption |

**k-NN achieved R² = 0.83** — meaning the model explains 83% of variance in used car prices. The remaining 17% is attributable to unobservable factors (accident history, cosmetic condition, negotiation dynamics).

---

## Top 5 Features Driving Price (CART Variable Importance)

1. **Year of manufacture** — newest cars command highest premium
2. **Power (BHP)** — engine power is the strongest technical predictor
3. **Engine size (CC)** — larger displacement correlates strongly with price
4. **Kilometers driven** — each additional km driven reduces value
5. **Transmission type** — automatics price significantly higher than manuals

---

## What I Built

### 1. Data Preprocessing Pipeline
- Loaded real-world used car listings dataset (Kaggle — Indian automotive market)
- Extracted numeric values from text fields: `"1248 CC"` → `1248`, `"18.9 kmpl"` → `18.9`
- Median imputation for missing numeric values — preserves distribution without bias
- Factor encoding for categorical variables: Location, Fuel Type, Transmission, Owner Type
- 70/30 train/test split with stratified partitioning via `caret::createDataPartition`

### 2. Exploratory Data Analysis (EDA)
- Correlation matrix — identified multicollinearity between Engine, Power, and Price
- Price distribution histogram — right-skewed, luxury segment creates long tail
- Boxplots: Price by Fuel Type, Transmission, Owner Type
- Scatter plots: Price vs Kilometers Driven, Engine, Power, Mileage
- VIF analysis on MLR to quantify multicollinearity

### 3. Five Machine Learning Models

**Multiple Linear Regression (baseline)**
- VIF analysis confirmed multicollinearity between Engine and Power
- Residual plots showed heteroscedasticity — motivating regularized alternatives

**k-Nearest Neighbors (best model)**
- Features normalized via `preProcess(method = c("center", "scale"))`
- `tuneLength = 10` — optimal k selected via cross-validation
- R² = 0.83, RMSE = 0.41 — best generalization across all vehicle segments

**CART Regression Tree**
- Visualized full decision tree via `rpart.plot`
- Variable importance plot — Year and Power are dominant predictors
- R² = 0.75, interpretable splits for business stakeholders

**Ridge Regression**
- Cross-validated lambda selection (`cv.glmnet`, alpha = 0)
- Shrinks correlated coefficients — addresses Engine/Power multicollinearity
- Lift chart built on Ridge predictions across 10 deciles

**Lasso Regression**
- Automatic feature selection via L1 penalty (alpha = 1)
- Some coefficients shrunk to zero — confirms feature redundancy
- Comparable performance to Ridge with implicit variable selection

### 4. LangChain + GPT-4 Explanation Module
When the model predicts a price, the system generates a natural language explanation:

> *"This 2018 Honda City with 45,000 km, a 1497 CC engine, and automatic transmission is estimated at ₹7.2 lakhs. The primary drivers of this valuation are its recent manufacture year (2018), above-average engine power (119 BHP), and low kilometer count — all factors associated with higher resale demand in this segment."*

This explainability layer builds user trust and makes the system usable by non-technical buyers and sellers.

### 5. Streamlit Web Application
Interactive UI allowing users to:
- Input vehicle specifications (year, engine, power, mileage, km driven, transmission, fuel type)
- Receive instant price prediction from the k-NN model
- Read a GPT-generated plain-English explanation of the predicted price

---

## Dataset

**Source:** [Used Cars India Dataset — Kaggle](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction)

| Feature | Type | Description |
|---|---|---|
| Year | Numeric | Year of manufacture |
| Kilometers_Driven | Numeric | Total km driven |
| Fuel_Type | Categorical | Petrol / Diesel / CNG / Electric |
| Transmission | Categorical | Manual / Automatic |
| Owner_Type | Categorical | First / Second / Third / Fourth |
| Mileage | Numeric | Fuel efficiency (km/l) — extracted from text |
| Engine | Numeric | Engine displacement in CC — extracted from text |
| Power | Numeric | Engine power in BHP — extracted from text |
| Seats | Numeric | Number of seats |
| Price | Numeric | **Target variable** — selling price in lakhs (INR) |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | R |
| ML & Modeling | caret, glmnet, rpart — Linear Regression, k-NN, CART, Ridge, Lasso |
| Visualization | ggplot2, corrplot, rpart.plot — correlation, residuals, lift chart, variable importance |
| Feature Engineering | stringr, tidyverse — text extraction, imputation, normalization |
| Explainable AI | LangChain + OpenAI GPT-4 — natural language price explanations |
| Web App | Streamlit (Python) — interactive prediction interface |
| Validation | RMSE, R², residual analysis, VIF, lift chart |

---

## Project Structure

```
used-car-price-prediction/
│
├── project-1.R                          # Complete R pipeline — preprocessing to model comparison
├── README.md
└── docs/
    ├── Used_Car_Price_Prediction_Report.pdf   # Full project report
    └── Final_Project.pptx                     # Presentation slides
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/yashwanthmatta/used-car-price-prediction.git
cd used-car-price-prediction
```

**2. Download the dataset**

Get `used_cars_data.csv` from [Kaggle](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction) and place it in the project root.

**3. Run the R pipeline**
```r
# Open project-1.R in RStudio
# Run all — it will prompt you to select the CSV file
source("project-1.R")
```

**4. Install required packages (auto-installs on first run)**
```r
# The script handles this automatically:
required_packages <- c("tidyverse", "stringr", "caret", "ggplot2",
                       "corrplot", "GGally", "glmnet", "rpart",
                       "rpart.plot", "gains", "car")
```

---

## Key Insights

- **Year of manufacture is the single strongest predictor** — a car loses approximately 8–12% of its value per year on average in this dataset
- **Automatic transmission commands a 25–35% price premium** over equivalent manual vehicles
- **Diesel cars price higher than petrol** in the mid-to-large segment, likely due to fuel efficiency for high-mileage use cases
- **First-owner cars price 15–20% higher** than second-owner equivalents, even with identical specifications
- **Multicollinearity between Engine and Power** (r = 0.87) — Ridge and Lasso address this better than plain MLR

---


