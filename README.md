# Satellite Imagery–Assisted Property Valuation

This project focuses on predicting residential property prices using structured housing attributes and satellite imagery. The objective is to study whether environmental context derived from satellite images can complement traditional tabular data for property valuation.

Rather than relying heavily on images alone, the project prioritizes a strong tabular baseline model and then integrates satellite-derived visual features to evaluate their incremental contribution within a multimodal learning pipeline.


## Project Overview

Residential property prices are influenced not only by internal characteristics such as size, number of rooms, and construction quality, but also by neighborhood-level factors such as surrounding infrastructure, density, and open spaces.

In this project:
- A high-performing **tabular regression model (XGBoost)** is first developed using housing attributes.
- Satellite images corresponding to each property’s geographic coordinates are programmatically downloaded.
- Image-based features are extracted using a pretrained convolutional neural network (CNN).
- Tabular and visual features are combined to form a multimodal regression pipeline.

The emphasis of this work is on building a **complete, reproducible, and well-structured pipeline**, rather than maximizing image-based performance alone.


## Dataset Description

### Tabular Data
The base dataset consists of residential property records containing numerical attributes and geographic coordinates.

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
Satellite images are fetched programmatically using latitude and longitude coordinates associated with each property. Due to cloud cover filtering and API constraints, only a subset of properties have valid satellite images.

The satellite imagery is used to capture **high-level environmental context** rather than fine-grained visual details.


## Repository Structure


```
Property_Valuation_XGBoost/
├── data/
│   ├── train(1).xlsx
│   ├── test2.xlsx
│   ├── train_features.csv
│   ├── test_features.csv
│   ├── train_target.csv
│   ├── image_index.csv
│   └── visual_features.csv
│
├── images_sample/
│   └── Sample satellite image tiles
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── image_eda.ipynb
│   ├── cnn_visual_features.ipynb
│   └── model_training.ipynb
│
├── data_fetcher.py
├── 24571002_final.csv
├── 24571002_report.pdf
└── README.md
```

**Note:** The complete satellite image dataset is not uploaded due to file size limitations. A small representative sample is included for reference.


## Methodology

### 1. Data Preprocessing
The raw housing dataset is cleaned and a selected set of relevant numerical features is prepared for modeling. Cleaned feature and target files are saved for reproducibility.

### 2. Tabular Model (Baseline)
An **XGBoost regressor** is trained using only tabular features. This model serves as the primary performance benchmark and achieves strong predictive accuracy.

### 3. Satellite Image Acquisition
Satellite images are programmatically downloaded using geographic coordinates. Cloud coverage filtering is applied to retain only usable images.

### 4. Image Feature Extraction
A pretrained **ResNet18** model (ImageNet weights) is used as a fixed feature extractor. The final classification layer is removed, and the network outputs compact visual embeddings representing neighborhood context.

### 5. Multimodal Fusion
Extracted image embeddings are concatenated with tabular features (early fusion) to form a combined feature vector. This fused representation is then used for regression modeling.

### 6. Evaluation
Model performance is evaluated using:
- Root Mean Squared Error (RMSE)
- R² Score

Results from the tabular-only and multimodal models are compared to assess the contribution of visual features.


## Results Summary

**Tabular Model (XGBoost):**
- R² Score: approximately **0.89**
- RMSE: lower than the multimodal model

**Multimodal Model (Tabular + Satellite Images):**
- Performance is slightly lower due to reduced image availability and visual noise
- Successfully demonstrates integration of tabular and image-based data

These results indicate that while satellite imagery provides useful contextual information, a strong tabular model remains dominant when image coverage or quality is limited.


## Notes and Limitations

- Satellite imagery resolution and cloud coverage affect feature quality.
- Only a subset of properties have valid satellite images.
- The project emphasizes **engineering completeness and methodological clarity** over maximizing image-based accuracy.
- The pipeline can be extended to higher-resolution imagery or richer geospatial features.


## How to Run

1. Set up a Python virtual environment  
2. Install required dependencies:
   - pandas  
   - numpy  
   - scikit-learn  
   - xgboost  
   - torch  
   - torchvision  
   - pillow  
3. Run `data_fetcher.py` to download satellite images  
4. Execute `preprocessing.ipynb` for data cleaning and feature preparation  
5. Run `cnn_visual_features.ipynb` to extract visual embeddings  
6. Execute `model_training.ipynb` to train models and generate predictions  


## Submission Details

- Enrollment Number: **24571002**
- Prediction File: `24571002_final.csv`
- Report File: `24571002_report.pdf`
- Repository: Public GitHub repository

All code and analysis were implemented specifically for this project.


