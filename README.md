🧠 PyTorch Encoder-Decoder RNN Chatbot
A complete sequence-to-sequence chatbot built using an encoder-decoder architecture with recurrent neural networks (RNNs) in PyTorch. This project is an educational implementation demonstrating how chatbots learn conversational patterns from raw dialogue data.

📌 Overview
This project focuses on building a conversational chatbot trained on movie dialogue pairs using:

Encoder-Decoder model with GRU or LSTM

Sequence padding, batching, and masking

Tokenization, vocabulary handling, and special tokens

Teacher forcing during training

Greedy decoding for inference

🛠️ Tech Stack
Component	Details
Framework	PyTorch
Language	Python 3.8+
Model	Seq2Seq with RNN (GRU/LSTM)
Dataset	Cornell Movie Dialog Corpus
Preprocessing	NLTK, custom vocabulary builder
Training Utility	Custom loss masking, batching

📁 Project Structure
bash
Copy
Edit
pytorch_enc-dec_rnn_chatbot/
│
├── data/                        # Raw and preprocessed datasets
│   └── cornell_movie_dialogs/
│
├── models/                      # Encoder, Decoder, Seq2Seq logic
│   ├── encoder.py
│   ├── decoder.py
│   └── seq2seq.py
│
├── utils/                       # Tokenization, cleaning, batching
│   ├── preprocessing.py
│   └── helper.py
│
├── config.py                    # Hyperparameters and paths
├── train.py                     # Training pipeline
├── evaluate.py                  # Greedy decoding and inference
├── requirements.txt
└── README.md
