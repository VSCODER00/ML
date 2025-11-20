#  Next Word Prediction using RNN, LSTM & GRU

A comparative deep learning project that predicts the next word(s) in a sentence using three different neural architectures: **Simple RNN, LSTM, and GRU**. The project also includes an interactive Streamlit UI for real-time prediction.

---

##  Project Overview

This project demonstrates how sequential deep learning models perform on a **Next Word Prediction** task.  
We trained three models **RNN**, **LSTM**, and **GRU**  on the same dataset, with identical preprocessing steps and hyperparameters, to ensure a fair comparison.

A Streamlit interface allows users to enter a sentence and generate multiple next words using any of the models.

---

##  Dataset

We used a text dataset stored in `dataset_book.txt`.  
The dataset is loaded and transformed using Keras Tokenizer.

---

##  Preprocessing

- Convert dataset to lowercase  
- Tokenize text using Keras Tokenizer  
- Create input sequences  
- Apply padding  
- Use an Embedding Layer in all models  

---

##  Model Architecture (Common for all models)

All three architectures use identical hyperparameters for a fair comparison.

- Embedding layer  
- RNN / LSTM / GRU layer (128 units)  
- Dense softmax output layer  

Each model was trained and saved separately:

- `BestRNNModel.h5`  
- `LSTM.h5`  
- `GRU.h5`  
- `tokenizer.pkl`

---

##  Streamlit UI

A minimal Streamlit app provides real-time predictions.

### Streamlit Frontend Features:
- Text input for entering a sentence  
- Slider to select number of words to generate  
- Output shows predictions from all three models  

---

##  How to Run Locally

###  Install dependencies
```bash
pip install -r requirements.txt

```

###  RUn the streamlit app
```bash
streamlit run app.py
```