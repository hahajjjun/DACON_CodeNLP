# DACON_CodeNLP
YBIGTA Science Team
DACON Code NLP Solution

### 📑 Docs
- `python preprocessing.py` to make custom dataset (train_data.csv : 3.0GB, val_data.csv: 50MB) <br/>
- `python pipeline.py` to train model with custom dataset <br/>
- `python Inference.py` to make predictions with trained model(\weights\codeBERT.pth) <br/>

### 💡 Ideas
- Aggressive negative sampling
- CodeBERTa PLM
- Test time augmentation based on commutativity

### 📁 Directory Structure

├── Inference.py <br/>
├── code <br/>
│         ├── problem001 <br/>
│         ├── ... <br/>
│         └── problem300 <br/>
├── weights <br/>
│         └── codeBERT.pth <br/>
├── pipeline.ipynb <br/>
├── pipeline.py <br/>
├── preprocessing.py <br/>
├── sample_submission.csv <br/>
├── sample_train.csv <br/>
├── submission_codeBERTa.csv <br/>
├── test.csv <br/>
├── train_data.csv <br/>
└── val_data.csv <br/>
