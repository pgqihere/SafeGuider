import json
import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from scipy.stats import entropy
from tqdm import tqdm
import os
from datetime import datetime
import argparse

def analyze_attention_patterns(output_file="attention_sac_results.json"):
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = CLIPTextModel.from_pretrained("../../Models/stable-diffusion-v1-4/text_encoder", output_attentions=True)
    tokenizer = CLIPTokenizer.from_pretrained("../../Models/stable-diffusion-v1-4/tokenizer")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load datasets
    print("Loading datasets...")
    benign_path = "../../Datasets/Benign/coco2017-2k.json"
    malicious_path = "../../Datasets/NSFW/sexual/P4D.json"

    with open(benign_path, 'r') as f:
        benign_data = json.load(f)

    with open(malicious_path, 'r') as f:
        malicious_data = json.load(f)

    # Process benign dataset
    print("Processing benign dataset...")
    benign_results = calculate_metrics(benign_data[:2000], model, tokenizer, device, "benign")  
    
    # Process malicious dataset
    print("Processing malicious dataset...")
    malicious_results = calculate_metrics(malicious_data, model, tokenizer, device, "malicious")

    # Print entropy values for each layer
    print("\nAttention entropy values for each layer:")
    print("Layer | Benign Mean Entropy | Std | Malicious Mean Entropy | Std")
    print("-" * 80)
    
    num_layers = len(benign_results["layer_stats"])
    for layer_idx in range(num_layers):
        print(f"Layer {layer_idx} | {benign_results['layer_stats'][layer_idx]['mean_entropy']:.3f} | "
              f"{benign_results['layer_stats'][layer_idx]['std_entropy']:.3f} | "
              f"{malicious_results['layer_stats'][layer_idx]['mean_entropy']:.3f} | "
              f"{malicious_results['layer_stats'][layer_idx]['std_entropy']:.3f}")

    # Print SAC values for each layer
    print("\nSemantic Attention Concentration (SAC) for each layer:")
    print("Layer | Benign Mean SAC | Std | Malicious Mean SAC | Std")
    print("-" * 80)
    
    for layer_idx in range(num_layers):
        print(f"Layer {layer_idx} | {benign_results['layer_stats'][layer_idx]['mean_sac']:.3f} | "
              f"{benign_results['layer_stats'][layer_idx]['std_sac']:.3f} | "
              f"{malicious_results['layer_stats'][layer_idx]['mean_sac']:.3f} | "
              f"{malicious_results['layer_stats'][layer_idx]['std_sac']:.3f}")

    # Print group average entropy values
    print("\nTable: Group Average Entropy Results")
    print("Dataset | Shallow Layers(0-5) Mean Entropy | Std | Deep Layers(6-11) Mean Entropy | Std")
    print("-" * 80)
    print(f"Benign | {benign_results['shallow_mean_entropy']:.3f} | {benign_results['shallow_std_entropy']:.3f} | {benign_results['deep_mean_entropy']:.3f} | {benign_results['deep_std_entropy']:.3f}")
    print(f"Malicious | {malicious_results['shallow_mean_entropy']:.3f} | {malicious_results['shallow_std_entropy']:.3f} | {malicious_results['deep_mean_entropy']:.3f} | {malicious_results['deep_std_entropy']:.3f}")

    # Print group average SAC values
    print("\nTable: Group Average SAC Results")
    print("Dataset | Shallow Layers(0-5) Mean SAC | Std | Deep Layers(6-11) Mean SAC | Std")
    print("-" * 80)
    print(f"Benign | {benign_results['shallow_mean_sac']:.3f} | {benign_results['shallow_std_sac']:.3f} | {benign_results['deep_mean_sac']:.3f} | {benign_results['deep_std_sac']:.3f}")
    print(f"Malicious | {malicious_results['shallow_mean_sac']:.3f} | {malicious_results['shallow_std_sac']:.3f} | {malicious_results['deep_mean_sac']:.3f} | {malicious_results['deep_std_sac']:.3f}")

    # Save results to JSON
    save_results_to_json(benign_results, malicious_results, output_file)

def calculate_metrics(dataset, model, tokenizer, device, dataset_type):
    """Calculate entropy and SAC metrics on dataset"""
    # Store entropy and SAC values for each layer
    layer_entropies = {}
    layer_sacs = {}
    
    # Define EOS token ID
    eos_token_id = 49407
    
    # Add progress bar
    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_type} samples")):
        prompt = item["prompt"]
        
        try:
            # Tokenization
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=77).to(device)
            tokens = inputs.input_ids[0]
            
            # Get model outputs and attention weights
            with torch.no_grad():
                outputs = model(**inputs)
            
            attentions = outputs.attentions  # Attention tensors for each layer
            num_layers = len(attentions)
            
            # Initialize layer metric lists (if not already initialized)
            if not layer_entropies:
                for i in range(num_layers):
                    layer_entropies[i] = []
                    layer_sacs[i] = []
            
            # Get EOS token positions
            eos_positions = (tokens == eos_token_id).nonzero()
            if eos_positions.numel() == 0:
                continue  # Skip if no EOS token found
                
            eos_position = eos_positions[-1, 0].item()
            sequence_length = tokens.size(0)
            
            if sequence_length <= 1:
                continue
            
            # Determine semantic keywords - use tokens with high attention from EOS in the last layer (layer 11)
            # Get attention distribution of EOS token in the last layer
            last_layer_attn = attentions[-1][0].mean(dim=0).cpu().numpy()
            eos_attn_dist = last_layer_attn[eos_position]
            
            # Exclude EOS token self-attention
            token_indices = np.array([i for i in range(sequence_length) if i != eos_position])
            token_attentions = np.array([eos_attn_dist[i] for i in range(sequence_length) if i != eos_position])
            
            # Define top 30% high attention tokens as semantic keywords
            if len(token_attentions) > 0:
                semantic_threshold = np.percentile(token_attentions, 70)
                semantic_indices = token_indices[token_attentions >= semantic_threshold]
                
                # Calculate entropy and SAC for each layer
                for layer_idx in range(num_layers):
                    attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()
                    
                    # Get EOS token's attention to other tokens
                    other_indices = [j for j in range(sequence_length) if j != eos_position]
                    eos_attn_dist = np.array([attn[eos_position, j] for j in other_indices])
                    
                    if eos_attn_dist.size > 0 and eos_attn_dist.sum() > 0:
                        # Normalize to probability distribution
                        eos_attn_dist = eos_attn_dist / eos_attn_dist.sum()
                        
                        # Calculate entropy
                        entropy_value = entropy(eos_attn_dist)
                        layer_entropies[layer_idx].append(entropy_value)
                        
                        # Calculate SAC (Semantic Attention Concentration)
                        semantic_attn_sum = sum(attn[eos_position, j] for j in semantic_indices)
                        all_attn_sum = sum(attn[eos_position, j] for j in range(sequence_length) if j != eos_position)
                        
                        if all_attn_sum > 0:
                            sac = semantic_attn_sum / all_attn_sum
                            layer_sacs[layer_idx].append(sac)
                    
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            continue
    
    # Calculate mean entropy and standard deviation for each layer, as well as mean SAC and standard deviation
    layer_stats = {}
    for layer_idx in range(num_layers):
        layer_stats[layer_idx] = {
            "mean_entropy": np.mean(layer_entropies[layer_idx]),
            "std_entropy": np.std(layer_entropies[layer_idx]),
            "mean_sac": np.mean(layer_sacs[layer_idx]),
            "std_sac": np.std(layer_sacs[layer_idx])
        }
    
    # Calculate statistics for shallow and deep layers
    shallow_entropies = []
    deep_entropies = []
    shallow_sacs = []
    deep_sacs = []
    
    for i in range(6):
        shallow_entropies.extend(layer_entropies[i])
        shallow_sacs.extend(layer_sacs[i])
    for i in range(6, num_layers):
        deep_entropies.extend(layer_entropies[i])
        deep_sacs.extend(layer_sacs[i])
    
    overall_stats = {
        "layer_stats": layer_stats,
        "shallow_mean_entropy": np.mean(shallow_entropies),
        "shallow_std_entropy": np.std(shallow_entropies),
        "deep_mean_entropy": np.mean(deep_entropies),
        "deep_std_entropy": np.std(deep_entropies),
        "shallow_mean_sac": np.mean(shallow_sacs),
        "shallow_std_sac": np.std(shallow_sacs),
        "deep_mean_sac": np.mean(deep_sacs),
        "deep_std_sac": np.std(deep_sacs)
    }
    
    return overall_stats

def save_results_to_json(benign_results, malicious_results, output_file):
    """Save results to JSON file, append if exists, create if not"""
    
    # Create new result entries
    new_entries = [
        {
            'dataset_name': 'benign_ood',
            'eos_token_shallow_layers_sac': round(benign_results['shallow_mean_sac'], 4),
            'eos_token_deep_layers_sac': round(benign_results['deep_mean_sac'], 4),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'dataset_name': 'p4d',
            'eos_token_shallow_layers_sac': round(malicious_results['shallow_mean_sac'], 4),
            'eos_token_deep_layers_sac': round(malicious_results['deep_mean_sac'], 4),
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

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze attention patterns and calculate SAC metrics')
    parser.add_argument('--output_file', type=str, default='attention_sac_results.json',
                       help='Output JSON file to save results (default: attention_sac_results.json)')
    
    args = parser.parse_args()
    
    try:
        analyze_attention_patterns(args.output_file)
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()