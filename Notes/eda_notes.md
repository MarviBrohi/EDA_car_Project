# Exploratory Data Analysis (EDA) - Car Dataset

## 1. What is EDA?
-EDA is the process of understanding the stricture. quality, and patterns in data before modeling.
EDA is about “getting to know” your data: what variables exist, how they are distributed, how they relate, and where the data has problems (missing values, outliers, wrong types, etc.).
-In the used‑car example, EDA is used to understand what affects a car’s price (age, brand, kilometers driven, fuel type, etc.).
-In real life, think of EDA like a doctor’s initial checkup: measuring temperature, blood pressure, and basic tests before deciding any treatment.
Core Python tools: pandas for tables, numpy for numbers, matplotlib/seaborn for plots.

## How to perform EDA using Python?
A typical EDA flow in Python is:
-Import libraries
-Load data
-Understand structure (rows, columns, dtypes, missing values)
-Clean and reduce data
-Feature engineering
-Univariate, bivariate, multivariate analysis
-Transform and impute where needed

Step 1: Import Python Libraries
The first step involved in ML using python is understanding and playing around with our data using libraries. Here is the link to the dataset.
Import all libraries required for our analysis, such as those for data loading, statistical analysis, visualizations, data transformations, and merging and joining.
Pandas and Numpy have been used for Data Manipulation and numerical Calculations
Matplotlib and Seaborn have been used for Data visualizations. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')