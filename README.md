# Arabic Text Autocorrection with AraBERT

An Arabic text autocorrection system built using AraBERT (Arabic BERT) for masked language modeling. This project fine-tunes a pre-trained AraBERT model to automatically correct spelling and grammatical errors in Arabic text.

## 🚀 Features

- **Arabic Text Preprocessing**: Comprehensive text cleaning including diacritics removal, English text filtering, and special character handling
- **AraBERT Integration**: Uses the pre-trained `aubmindlab/bert-base-arabert` model
- **Cross-Validation Training**: Implements 5-fold cross-validation for robust model evaluation
- **Real-time Correction**: Provides an autocorrection function for testing on new sentences
- **High Accuracy**: Achieves an average test accuracy of ~92.84% across folds

## 📋 Requirements

```
torch
transformers
scikit-learn
numpy
google-colab (for Google Drive integration)
```

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/arabic-text-autocorrection.git
cd arabic-text-autocorrection
```

2. Install required packages:
```bash
pip install torch transformers scikit-learn numpy
```

3. If running on Google Colab, mount your Google Drive to access the dataset:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 📊 Dataset

The project expects two text files:
- `ara-eg_newscrawl-OSIAN_2018_10K-sentences.txt`: Correct Arabic sentences (labels)
- `ara-eg_newscrawl-OSIAN_2018_10K-labels.txt`: Incorrect Arabic sentences (input)

### Data Format
Each file should contain one sentence per line, with optional line numbers that will be automatically removed during preprocessing.

## 🔧 Usage

### Training the Model

1. **Data Preprocessing**: The notebook includes comprehensive text cleaning:
   - Removes leading digits and whitespace
   - Filters out English words
   - Removes Arabic diacritics (fatha, kasra, damma)
   - Normalizes whitespace and special characters

2. **Model Training**: Run the cross-validation training:
```python
# The model trains for 5 epochs across 5 folds
# Training parameters:
learning_rate = 5e-5
num_epochs = 5
batch_size = 8
num_folds = 5
```

3. **Testing**: Use the autocorrection function:
```python
sentences = ["صودا الخبز هو معجون طبيعي للأسنان حثي ينصح بخلط ربع ملعقة صغيرة من صودا الخبز مع الماء وغسل الأسنان ببه"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
corrected_sentences = auto_correct_sentences(model, tokenizer, sentences, device)
```

### Example Output

```
Original sentence: صودا الخبز هو معجون طبيعي للأسنان حثي ينصح بخلط ربع ملعقة صغيرة من صودا الخبز مع الماء وغسل الأسنان ببeh
Corrected sentence: صودا الخبز هو معجون طبيعي للأسنان حيث ينصح بخلط ربع ملعقة صغيرة من صودا الخبز مع الماء وحمر الأسنان ببه ب
```

## 📈 Model Performance

The model achieves the following performance across 5-fold cross-validation:

| Metric | Value |
|--------|-------|
| Average Test Accuracy | 92.84% |
| Average Test Loss | 0.43 |

### Training Progress Example:
- **Fold 1**: Test Accuracy: 79.73%
- **Fold 2**: Test Accuracy: 91.23%
- **Fold 3**: Test Accuracy: 96.02%
- **Fold 4**: Test Accuracy: 97.99%
- **Fold 5**: Test Accuracy: 99.21%

## 🏗️ Model Architecture

- **Base Model**: `aubmindlab/bert-base-arabert`
- **Task**: Masked Language Modeling for sequence-to-sequence correction
- **Tokenizer**: AraBERT tokenizer with Arabic language support
- **Max Sequence Length**: 32 tokens
- **Optimizer**: AdamW with learning rate 5e-5

## 🔍 Text Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Initial Cleaning**: Remove leading digits and whitespace
2. **Language Filtering**: Remove English words using regex
3. **Diacritics Removal**: Remove Arabic diacritics (ً، ٌ، ٍ، َ، ُ، ِ)
4. **Character Filtering**: Keep only Arabic Unicode range (U+0600-U+06FF)
5. **Number Removal**: Remove all numeric characters
6. **Whitespace Normalization**: Normalize and clean extra whitespace

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [aubmindlab](https://github.com/aub-mind/arabert) for the pre-trained AraBERT model
- The OSIAN 2018 dataset contributors
- Hugging Face Transformers library

---

**Note**: This project was developed and tested on Google Colab with GPU acceleration. For optimal performance, GPU usage is recommended during training.
