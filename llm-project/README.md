# LLM Project

This project explores Large Language Models (LLMs) using Python and PyTorch.  
- Includes implementations of a Bigram model and a GPT-like architecture.  
- Uses sample dataset (`wizard_of_oz.txt`) for training and vocabulary building.  
- Experiments with tokenization, embedding layers, and text generation.  

## Structure
- `bigram.ipynb` → Bigram language model experiments  
- `gpt-v1.ipynb` → GPT-style model from scratch  
- `torch-examples.ipynb` → PyTorch practice and demos  
- `data-extract.py` → Data cleaning and preprocessing  
- `vocab.txt` → vocabulary generated from dataset  
- `wizard_of_oz.txt` → public domain dataset (training text)  
- `requirements.txt` → dependencies  

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook bigram.ipynb
