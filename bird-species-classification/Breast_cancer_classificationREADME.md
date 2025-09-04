# Breast Cancer Classification

This project applies Machine Learning models to classify breast cancer tumors as malignant or benign using the **Breast Cancer Wisconsin (Diagnostic) dataset**.  

## Features
- Implemented **Random Forest, Linear Regression, and Decision Tree** models.  
- Achieved **97% accuracy with Random Forest**, the best-performing model.  
- Applied **feature selection** and **preprocessing** to handle missing values and normalize data.  
- Improved prediction stability and boosted consistency by **25%**.  

## Project Structure
- `data/` → dataset files (not uploaded here, but sourced from the Breast Cancer Wisconsin dataset)  
- `preprocessing.py` → handles missing values, feature selection, and normalization  
- `train_models.py` → training scripts for Random Forest, Decision Tree, and Linear Regression  
- `evaluate.py` → metrics evaluation and comparison of models  
- `requirements.txt` → dependencies  

## How to Run
```bash
pip install -r requirements.txt
python train_models.py
python evaluate.py
