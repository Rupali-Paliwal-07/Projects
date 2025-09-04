# Bird Species Classification

This project classifies bird species from spectrograms using CNN, SVM, and InceptionV3.  
- Dataset: Custom dataset with 10,000+ spectrograms across 10 species  
- Achieved 88% accuracy with optimized models  
- Automated feature extraction with Python + OpenCV, reducing workload by 60%  
- Published in SCOPUS ICAETBM 2024; Copyright L-155587/2024

## Project Structure
- `data/` → spectrogram samples (not uploaded here, due to size)  
- `preprocessing.py` → data cleaning and spectrogram generation  
- `train_model.py` → training CNN, SVM, and InceptionV3 models  
- `evaluate.py` → evaluation metrics  
- `requirements.txt` → dependencies  

## How to Run
```bash
pip install -r requirements.txt
python train_model.py
python evaluate.py
