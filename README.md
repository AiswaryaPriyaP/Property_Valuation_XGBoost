# Satellite Imagery–Assisted Property Valuation

This project focuses on predicting residential property prices using a combination of structured housing attributes and satellite imagery. The primary goal is to evaluate whether environmental context extracted from satellite images can complement traditional tabular data for property valuation.

Unlike purely image-driven approaches, this work prioritizes a strong tabular baseline and then integrates satellite imagery to study its incremental value within a multimodal learning pipeline.


## Project Overview

Property prices are influenced not only by physical attributes such as size, number of rooms, and construction quality, but also by neighborhood characteristics such as surrounding infrastructure, density, and open spaces.

In this project:
- A high-performing **tabular regression model** is first developed using housing attributes.
- Satellite images corresponding to each property’s geographic coordinates are programmatically downloaded.
- Image-based features are extracted using a pretrained convolutional neural network.
- Multiple modeling strategies are compared to assess the contribution of visual context.

The emphasis is on building a **complete and reproducible pipeline**, rather than maximizing performance solely through image data.


## Dataset Description

### Tabular Data
The base dataset consists of residential property records containing both numerical attributes and geographic coordinates.

Files:
- `train(1).xlsx` – Training dataset with property features and target price
- `test2.xlsx` – Test dataset without price labels

Key features include:
- Bedrooms and bathrooms  
- Living area and lot size  
- Latitude and longitude  
- Construction grade and condition  
- View rating and waterfront indicator  

### Satellite Imagery
Satellite images are fetched using latitude and longitude coordinates associated with each property. Due to cloud cover filtering and API constraints, only a subset of properties have valid satellite images.

The satellite data is used to capture high-level environmental context rather than fine-grained visual details.


## Repository Structure

Property_Valuation_XGBoost/
│
├── data/
│ ├── train(1).xlsx
│ ├── test2.xlsx
│ ├── X_clean.csv
│ ├── X_test_clean.csv
│ └── y.csv
│
├── notebooks/
│ ├── xgb_tabular_model.ipynb
│ └── preprocessing.ipynb
│
├── images_sample/
│ └── Sample satellite image tiles
│
├── data_fetcher.py
├── 24571002_final.csv
└── README.md


Note: The complete satellite image dataset is not uploaded due to file size limitations. A small representative sample is included.


## Methodology

### 1. Data Preprocessing
The raw housing data is cleaned and relevant numerical features are selected. Basic exploratory analysis is performed to understand price distributions and feature relationships.

### 2. Tabular Model (Baseline)
An **XGBoost regressor** is trained using only tabular features. This model serves as the primary performance benchmark and achieves strong predictive accuracy.

### 3. Satellite Image Acquisition
Satellite images are programmatically downloaded using geographic coordinates. Cloud coverage filtering is applied to retain only usable images.

### 4. Image Feature Extraction
A pretrained **ResNet18** model (ImageNet weights) is used to extract fixed-length feature embeddings from satellite images. The final classification layer is removed so that the network acts purely as a feature extractor.

### 5. Multimodal Fusion
Image embeddings are concatenated with tabular features (early fusion) to form a combined feature vector. This fused representation is then used for regression.

### 6. Evaluation
Model performance is evaluated using:
- Root Mean Squared Error (RMSE)
- R² Score

Results from tabular-only and multimodal models are compared.


## Results Summary

**Tabular Model (XGBoost):**
- R² Score: approximately **0.89**
- RMSE: significantly lower than multimodal model

**Multimodal Model (Tabular + Satellite Images):**
- Performance is slightly lower due to reduced image availability and visual noise
- Demonstrates successful integration of image and tabular data

These results indicate that while satellite imagery provides useful contextual information, a strong tabular model remains dominant when image quality or coverage is limited.


## Notes and Limitations

- Satellite imagery resolution and cloud coverage affect feature quality.
- Only a subset of properties have valid satellite images.
- The project emphasizes **engineering completeness and methodology clarity** over raw image-based performance.
- The approach is extensible to higher-resolution imagery or richer geospatial features.


## How to Run

1. Set up a Python virtual environment
2. Install required dependencies (pandas, numpy, scikit-learn, xgboost, torch, torchvision, pillow)
3. Run `data_fetcher.py` to download satellite images
4. Execute `preprocessing.ipynb` for data cleaning and EDA
5. Run `xgb_tabular_model.ipynb` to train models and generate predictions


## Submission Details

- Enrollment Number: **24571002**
- Prediction File: `24571002_final.csv`
- Report File: `24571002_report.pdf`
- Repository: Public GitHub repository

All code and analysis were implemented specifically for this project.
