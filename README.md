# Data Cleaning and Preprocessing on Titanic Dataset

## Overview

This project focuses on preparing raw data for machine learning by performing data cleaning and preprocessing operations on the Titanic dataset. The goal is to transform the dataset into a structured and usable format for further analysis and modeling.

---

## Objective

* Identify and handle missing values
* Convert categorical variables into numerical format
* Normalize numerical features
* Detect and remove outliers
* Prepare clean data suitable for machine learning models

---

## Dataset

The dataset contains passenger information such as age, gender, fare, class, and survival status.

---

## Methodology

### Data Inspection

* Examined dataset structure and data types
* Identified missing values across columns

### Missing Value Handling

* Filled numerical values (Age) using median
* Filled categorical values (Embarked) using mode
* Removed columns with excessive missing data (Cabin)

### Encoding

* Converted categorical variables into numerical form
* Applied label encoding for binary variables
* Used one-hot encoding for multi-category features

### Normalization

* Standardized numerical features using StandardScaler

### Outlier Detection and Removal

* Visualized outliers using boxplots
* Applied IQR method to remove extreme values

---

## Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## How to Run

1. Place the dataset file in the same directory
2. Run:

```bash
python task1_preprocessing.py
```

---

## Result

The dataset is cleaned, transformed, and ready for machine learning applications with improved consistency and reliability.

---

## Key Concepts

* Data Cleaning
* Missing Value Imputation
* Feature Encoding
* Feature Scaling
* Outlier Detection

---

## Author

Ridhima Panigrahi
