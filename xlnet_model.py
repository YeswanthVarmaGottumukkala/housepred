import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
from torch import nn
from transformers import XLNetModel, XLNetTokenizer

# Flag to track if custom XLNet is available
xlnet_available = False

# Path to your custom model
CUSTOM_MODEL_PATH = r"C:\Users\Lenovo\Downloads\answer-evaluation\utils\xlnet_answer_assessment_model.pt"

# Define a custom model class that matches your saved model structure
class XLNetAnswerAssessmentModel(nn.Module):
    def __init__(self):
        super(XLNetAnswerAssessmentModel, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        # Add the additional layers that are in your saved model
        hidden_size = 768  # Standard size for xlnet-base-cased
        self.dense1 = nn.Linear(hidden_size, 256)
        self.dense2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)  # Assuming final output is a single score
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get XLNet embeddings
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the pooled output or mean of last hidden states
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        
        # Pass through additional layers
        x = torch.relu(self.dense1(pooled))
        x = torch.relu(self.dense2(x))
        x = torch.sigmoid(self.output(x))  # Sigmoid for 0-1 output
        
        return x

# Try to load custom XLNet model, but fall back gracefully if it fails
try:
    # Check if the model file exists
    if not os.path.exists(CUSTOM_MODEL_PATH):
        raise FileNotFoundError(f"Custom model not found at {CUSTOM_MODEL_PATH}")
    
    print("Loading tokenizer...")
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    
    print("Creating custom model architecture...")
    model = XLNetAnswerAssessmentModel()
    
    print(f"Loading custom weights from {CUSTOM_MODEL_PATH}...")
    # Load the entire model directly
    state_dict = torch.load(CUSTOM_MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    
    xlnet_available = True
    print("Custom XLNet model loaded successfully!")
except Exception as e:
    print(f"Error loading custom XLNet model: {e}")
    print("Falling back to TF-IDF similarity method")
    xlnet_available = False

# Create a simple cache for embeddings
embedding_cache = {}

def get_model_prediction(question_text, student_answer_text, reference_answer_text):
    """
    Get prediction from the custom model
    """
    if not xlnet_available:
        raise ValueError("XLNet model is not available")
    
    # Prepare inputs (this may need adjustment based on your model's exact requirements)
    # Combine texts with separators
    combined_text = f"{question_text} [SEP] {student_answer_text} [SEP] {reference_answer_text}"
    
    # Tokenize
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        # Convert to percentage score
        score = float(outputs.squeeze()) * 100
    
    return round(score)

def get_similarity_score(question_text, student_answer_text, reference_answer_text):
    """
    Calculate similarity score between student answer and reference answer,
    taking the question into context
    """
    try:
        if xlnet_available:
            # Use the custom model for prediction
            return get_model_prediction(question_text, student_answer_text, reference_answer_text)
        else:
            # Fall back to TF-IDF similarity if XLNet is not available
            return tfidf_similarity(student_answer_text, reference_answer_text)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        # Fallback to a simpler similarity calculation
        return fallback_similarity(student_answer_text, reference_answer_text)

def tfidf_similarity(text1, text2):
    """
    Calculate similarity using TF-IDF vectorization
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Convert to percentage
    return round(similarity * 100)

def fallback_similarity(text1, text2):
    """
    Fallback method for calculating text similarity using simple word overlap
    """
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    similarity = intersection / union if union > 0 else 0
    
    return round(similarity * 100)
