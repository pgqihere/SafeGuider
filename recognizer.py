import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
import os

class ThreeLayerClassifier(nn.Module):  
    def __init__(self, dim):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        prob = F.softmax(x, dim=-1) 
        return x, prob

class PromptSafetyClassifier:
    def __init__(self):
        """
        Initialize the safety classifier with fixed model paths
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 77  # Maximum length for CLIP tokenizer
        self.eos_token_id = 49407  # CLIP EOS token ID

        # Fixed model paths
        self.model_dir = "stable_diffusion_clip"
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer")
        self.text_encoder_path = os.path.join(self.model_dir, "text_encoder")
        self.classifier_path = os.path.join("Models", "SD1.4_safeguider.pt")
        
        # Load models
        self.load_text_encoder()
        self.load_classifier()
    
    def load_text_encoder(self):
        """Load CLIP text encoder and tokenizer"""
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_path)
            self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder_path).to(self.device)
            self.text_encoder.eval()
            
            print("Successfully loaded text encoder and tokenizer")
        except Exception as e:
            print(f"Error loading text encoder: {e}")
            raise
    
    def load_classifier(self):
        """Load classifier model"""
        try:
            # Get text encoder embedding dimension
            embedding_dim = self.text_encoder.config.hidden_size
            
            self.classifier = ThreeLayerClassifier(embedding_dim).to(self.device)
            classifier_state = torch.load(self.classifier_path, map_location=self.device, weights_only=False)
            self.classifier.load_state_dict(classifier_state)
            self.classifier.eval()
            
            print("Successfully loaded classifier model")
        except Exception as e:
            print(f"Error loading classifier: {e}")
            raise
    
    def get_text_embedding(self, prompt):
        """
        Get text embedding for the prompt
        
        Args:
            prompt: Input text
            
        Returns:
            embedding: Text embedding
            eos_position: Position of the EOS token
        """
        # Encode text using tokenizer
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Get text embedding
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        
        # Find EOS token position (original method without attention mask)
        tokens = input_ids.squeeze(0)
        eos_positions = (tokens == self.eos_token_id).nonzero()
        if len(eos_positions) > 0:
            eos_position = eos_positions[0].item()
        else:
            # If no EOS token found, use the last valid token
            eos_position = min(len(tokens) - 1, self.max_length - 1)
        
        return text_embeddings, eos_position
    
    def classify_prompt(self, prompt):
        """
        Classify prompt for safety
        
        Args:
            prompt: Input text
            
        Returns:
            safety_score: Safety score (probability of belonging to class 1)
            predicted_class: Predicted class (0 or 1)
            probabilities: Probabilities for both classes
        """
        # Get text embedding
        text_embeddings, eos_position = self.get_text_embedding(prompt)
        
        # Extract EOS token embedding
        eos_embedding = text_embeddings[:, eos_position, :]
        
        # Make prediction using classifier
        with torch.no_grad():
            logits, probabilities = self.classifier(eos_embedding)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            safety_score = probabilities[0, 1].item()  # Probability of class 1 as safety score
        
        return safety_score, predicted_class, probabilities[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Prompt Safety Classifier")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to classify")
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = PromptSafetyClassifier()
    
    # Classify prompt
    safety_score, predicted_class, probabilities = classifier.classify_prompt(args.prompt)
    
    # Output results
    print(f"\nPrompt: {args.prompt}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Safety Score (Class 1 probability): {safety_score:.4f}")
    print(f"Class Probabilities: [Class 0: {probabilities[0]:.4f}, Class 1: {probabilities[1]:.4f}]")
    
    # Provide assessment based on safety score
    if safety_score <= 0.5:
        print("Assessment: UNSAFE content detected")
    else:
        print("Assessment: SAFE content")

if __name__ == "__main__":
    main()