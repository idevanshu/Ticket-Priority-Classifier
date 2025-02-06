import streamlit as st
import torch
import torch.nn as nn
import spacy
import os
import pickle

# --------------------
# GLOBALS / CONSTANTS
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_DIM = 50  # Ensure this matches what you used in training
NUM_CLASSES = 3
MAX_LENGTH = 50  # The max length used for padding
VOCAB_FILE = "vocab.pkl"  # Ensure this file exists
MODEL_FILE = "cnn_model"  # Ensure this file exists

# Load SpaCy for text preprocessing
nlp = spacy.load('en_core_web_sm')

# ----------------------
# MODEL DEFINITION
# ----------------------
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(CNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully Connected Layer
        self.flatten_size = None  # Will be determined dynamically
        self.fc = None  # Will be set after first forward pass

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # Convert to (batch, embed_dim, seq_len)
        
        # Convolutional Blocks
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x)))

        # Dynamically determine `flatten_size` on first forward pass
        if self.flatten_size is None:
            self.flatten_size = x.shape[1] * x.shape[2]
            self.fc = nn.Linear(self.flatten_size, NUM_CLASSES).to(x.device)  # Define fully connected layer

        # Flatten and Fully Connected
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ----------------------
# TEXT PREPROCESSING
# ----------------------
def preprocess_text(text: str) -> str:
    """
    Preprocess input text using spaCy:
    - Convert to lowercase
    - Lemmatization
    - Remove stopwords and non-alphabetic tokens
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# ----------------------
# LOAD VOCAB & MODEL
# ----------------------
@st.cache_resource
def load_vocab():
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, "rb") as f:
            vocab = pickle.load(f)
        return vocab
    else:
        st.error("Vocabulary file not found! Make sure `vocab.pkl` exists.")
        return None

@st.cache_resource
def load_model():
    vocab = load_vocab()
    if vocab is None:
        return None, None
    
    # Initialize the model with the correct vocab size
    model_cnn = CNNClassifier(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES).to(DEVICE)

    # Dummy forward pass to set flatten_size
    dummy_input = torch.zeros((1, MAX_LENGTH), dtype=torch.long).to(DEVICE)
    model_cnn(dummy_input)

    # Load trained weights
    if os.path.exists(MODEL_FILE):
        model_cnn.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
        model_cnn.eval()  # Set to evaluation mode
        return model_cnn, vocab
    else:
        st.error(f"Model file '{MODEL_FILE}' not found. Make sure it exists.")
        return None, None

# ----------------------
# PREDICTION FUNCTION
# ----------------------
def predict(model, text, vocab):
    """
    1) Preprocess the text
    2) Encode it using vocab
    3) Pad/truncate to MAX_LENGTH
    4) Pass through model
    5) Return predicted class label
    """
    text = preprocess_text(text)
    encoded_text = torch.tensor([vocab.get(word, 0) for word in text.split()])

    # Pad or truncate to MAX_LENGTH
    if len(encoded_text) < MAX_LENGTH:
        padded_text = torch.cat([encoded_text, torch.zeros(MAX_LENGTH - len(encoded_text))])
    else:
        padded_text = encoded_text[:MAX_LENGTH]

    padded_text = padded_text.unsqueeze(0).long().to(DEVICE)

    output_mapping = {0: "P1", 1: "P2", 2: "P3"}

    with torch.no_grad():
        output = model(padded_text)
        _, predicted = torch.max(output.data, 1)
    
    return output_mapping[predicted.cpu().numpy()[0]]

# ----------------------
# STREAMLIT UI
# ----------------------
def main():
    st.title("🔍 Ticket Priority Classifier")
    st.write(
        """
        Enter a support ticket description, and the model will predict whether it's **P1**, **P2**, or **P3** priority.
        """
    )

    # Load the model and vocabulary
    model_cnn, vocab = load_model()

    if model_cnn is None or vocab is None:
        return  # Stop execution if model loading failed

    # User input
    user_text = st.text_area("📩 Enter Ticket Description:", 
        "Cannot access the following link; getting a blank page. Please help ASAP."
    )

    if st.button("Predict Priority"):
        prediction = predict(model_cnn, user_text, vocab)
        st.success(f"**Predicted Priority:** {prediction}")

if __name__ == "__main__":
    main()
