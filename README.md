English to Assamese Neural Machine Translation (NMT)
---------------------------------------------------
This project implements a Transformer-based Neural Machine Translation (NMT) model for
translating English sentences into Assamese. It demonstrates how transformer architectures,
based on the “Attention Is All You Need” paper, can effectively handle low-resource language
translation tasks.
Features
--------
- Transformer-based encoder–decoder model
- Trains on English–Assamese parallel corpus
- Tokenization and padding handled dynamically
- Supports BLEU score evaluation for accuracy
- Easy to extend for other Indic languages
Model Overview
---------------
The Transformer model uses self-attention and multi-head attention mechanisms for efficient
translation without recurrence.
Key components include:
- Positional Encoding for word order representation
- Scaled Dot-Product Attention
- Feed-Forward Neural Network in each layer
- Softmax-based Decoder Output
Installation & Setup
--------------------
1. Clone the repository:
git clone https://github.com//Language_Translation_ENGtoAssamesse.git
cd Language_Translation_ENGtoAssamesse
2. Install dependencies:
pip install torch torchvision torchaudio pandas numpy matplotlib tqdm nltk sentencepiece
3. Open the notebook:
jupyter notebook Language_Translation_ENGtoAssamesse.ipynb
Dataset
--------
Use any English–Assamese parallel corpus such as:
- OPUS Corpus (Tatoeba, GNOME, OpenSubtitles)
- AI4Bharat IndicTrans Dataset
or a custom CSV file with columns:
english_sentence, assamese_sentence
Training Details
----------------
Model: Transformer (Encoder–Decoder)
Embedding Dim: 512
Num Heads: 8
Layers: 6
Optimizer: Adam
Loss: CrossEntropyLoss
Epochs: 20–30 (configurable)
Example Usage
--------------
Input: "Good morning, how are you?"
Output: "■■■■■■■■, ■■■■■ ■■■■ ■■■?"
Evaluation
-----------
Performance is evaluated using BLEU score and perplexity on test data.
Saving and Loading Model
------------------------
torch.save(model.state_dict(), 'models/transformer_eng_to_assamese.pth')
model.load_state_dict(torch.load('models/transformer_eng_to_assamese.pth'))
model.eval()
License
--------
MIT License
Author
-------
Aadi Jain
