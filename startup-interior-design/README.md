# AI-Powered Interior Design Recommendation System

An AI startup project that generates optimized room layouts by combining **machine learning** and **recommendation systems**.  
The model integrates multiple design features (room size, furniture constraints, user preferences) to provide smart interior layout suggestions.

## Features
- **Hybrid ML models**: combines clustering + neural recommendations.  
- **Real-time UI**: interactive interface to preview room layouts.  
- **Scalable pipeline**: modular code for training, inference, and feedback loop.  

## Components
- `app.py` – Streamlit/Flask interface for users.  
- `train.py` – trains hybrid recommendation models.  
- `recommender.py` – logic for combining ML + rule-based design constraints.  
- `data_loader.py` – preprocessing and dataset handling.  
- `utils/ui_helpers.py` – generates mock 2D layout visualizations.  
- `utils/model_utils.py` – ML helper functions.  

## Quick Start
```bash
pip install -r requirements.txt

# Train a model
python train.py --data data/interiors.csv --epochs 20 --save models/

# Run the web app
streamlit run app.py
