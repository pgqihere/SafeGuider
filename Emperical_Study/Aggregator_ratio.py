import json
import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from datetime import datetime
import os

def calculate_eos_top1_aggregator_ratio(output_file="top1_aggregator_results.json"):
    # Load model and tokenizer
    model = CLIPTextModel.from_pretrained("../../Models/stable-diffusion-v1-4/text_encoder", output_attentions=True)
    tokenizer = CLIPTokenizer.from_pretrained("../../Models/stable-diffusion-v1-4/tokenizer")
    
    # Define EOS token ID
    eos_token_id = 49407
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load datasets
    benign_path = "../../Datasets/Benign/coco2017-2k.json"
    malicious_path = "../../Datasets/NSFW/sexual/P4D.json"

    with open(benign_path, 'r') as f:
        benign_data = json.load(f)

    with open(malicious_path, 'r') as f:
        malicious_data = json.load(f)

    # Process benign dataset
    print("Processing benign dataset...")
    benign_ratio = calculate_top1_aggregator_ratio(benign_data[:2000], model, tokenizer, device, eos_token_id)
    
    # Process malicious dataset
    print("Processing malicious dataset...")
    malicious_ratio = calculate_top1_aggregator_ratio(malicious_data, model, tokenizer, device, eos_token_id)
    
    # Print results
    print(f"\nBenign dataset EOS token as Top-1 aggregator ratio: {benign_ratio:.4f}")
    print(f"Malicious dataset EOS token as Top-1 aggregator ratio: {malicious_ratio:.4f}")
    
    # Save results to JSON
    save_results_to_json(benign_ratio, malicious_ratio, output_file)

def calculate_top1_aggregator_ratio(dataset, model, tokenizer, device, eos_token_id):
    """Calculate the ratio of EOS token as Top-1 aggregator"""
    total_samples = 0
    eos_as_top1_count = 0
    
    # Define attention threshold
    attention_threshold = 0.00  # Can adjust this threshold
    
    # CLIP model's maximum sequence length is 77
    max_length = 77
    
    for item in tqdm(dataset):
        prompt = item["prompt"]
        
        try:
            # Tokenization - added truncation=True and max_length=77 parameters
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            tokens = inputs.input_ids[0]
            
            # Get EOS token positions
            eos_positions = (tokens == eos_token_id).nonzero()
            if eos_positions.numel() == 0:
                continue  # Skip if no EOS token found
            
            # Get the last EOS position
            eos_position = eos_positions[-1, 0].item()
            sequence_length = tokens.size(0)
            
            # Skip prompts with only one token
            if sequence_length <= 1:
                continue
            
            # Get model outputs and attention weights
            with torch.no_grad():
                outputs = model(**inputs)
            
            attentions = outputs.attentions  # All layers' attention
            num_layers = len(attentions)
            
            # Calculate aggregator ratio for each token
            token_coverage_ratios = []
            
            for i in range(sequence_length):
                # Calculate token i's average attention to other tokens
                coverage_count = 0
                other_tokens_count = sequence_length - 1  # Excluding itself
                
                # Check if token i attends to other tokens (across all layers and heads)
                attention_matrix = torch.zeros(sequence_length)
                
                for layer_idx in range(num_layers):
                    layer_attention = attentions[layer_idx][0].mean(dim=0)  # Average all heads
                    for j in range(sequence_length):
                        if i != j:  # Exclude self-attention
                            attention_matrix[j] = max(attention_matrix[j], layer_attention[i, j].item())
                
                # Count attended tokens
                for j in range(sequence_length):
                    if i != j and attention_matrix[j] > attention_threshold:
                        coverage_count += 1
                
                # Calculate coverage ratio
                coverage_ratio = coverage_count / other_tokens_count if other_tokens_count > 0 else 0
                token_coverage_ratios.append(coverage_ratio)
            
            # Find Top-1 aggregator
            if token_coverage_ratios:
                top1_idx = np.argmax(token_coverage_ratios)
                
                # Check if EOS token is Top-1
                if top1_idx == eos_position:
                    eos_as_top1_count += 1
                
                total_samples += 1
        
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            continue
    
    # Calculate EOS as Top-1 aggregator ratio
    eos_top1_ratio = eos_as_top1_count / total_samples if total_samples > 0 else 0
    
    return eos_top1_ratio

def save_results_to_json(benign_ratio, malicious_ratio, output_file):
    """Save results to JSON file, append if exists, create if not"""
    
    # Create new result entries
    new_entries = [
        {
            'dataset_name': 'benign_ood',
            'token_type': 'eos_token',
            'top1_aggregator_ratio': round(benign_ratio, 4),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'dataset_name': 'p4d',
            'token_type': 'eos_token',
            'top1_aggregator_ratio': round(malicious_ratio, 4),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    ]
    
    # Check if file exists
    if os.path.exists(output_file):
        # Read existing data
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            
            # If existing data is a list, append directly
            if isinstance(existing_data, list):
                existing_data.extend(new_entries)
            # If existing data is a dict, convert to list format
            else:
                existing_data = [existing_data] + new_entries
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading existing file: {e}")
            # If reading fails, create new list
            existing_data = new_entries
    else:
        # File doesn't exist, create new list
        existing_data = new_entries
    
    # Save updated data
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate EOS token Top-1 aggregator ratio')
    parser.add_argument('--output_file', type=str, default='top1_aggregator_results.json',
                       help='Output JSON file to save results (default: top1_aggregator_results.json)')
    
    args = parser.parse_args()
    
    try:
        calculate_eos_top1_aggregator_ratio(args.output_file)
        print(f"\nCalculation completed successfully!")
        print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()