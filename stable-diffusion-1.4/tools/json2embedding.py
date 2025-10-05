import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np
import os
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm  
from safetensors.torch import load_file

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def create_labeled_dataset(prompts, model, save_path):
    """Create labeled embedding dataset"""
    dataset = {
        "data": []
    }
    max_length = 77
    print("Generating embeddings...")
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Processing prompts"):
        with torch.no_grad():
            # Get embedding
            emb = model.get_learned_conditioning([prompt])
            
            # Get sequence length (excluding padding)
            # Use CLIP tokenizer to get actual token length
            tokens = model.cond_stage_model.tokenizer.encode(prompt)
            tokens = torch.tensor(tokens)  

            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            eos_token_id = 49407
            eos_positions = (tokens == eos_token_id).nonzero()
            if len(eos_positions) > 0:
                print("Found the eos token!")
                eos_position = eos_positions[0].item()
            else:
                eos_position = min(len(tokens) - 1, max_length - 1)

            # Get EOS token embedding
            eos_emb = emb[:, eos_position, :].cpu().numpy()
            # Add debug code in create_labeled_dataset function
            
            print(f"Original prompt: {prompt}")
            print(f"Token count: {len(tokens)}")
            print(f"EOS position: {eos_position}")
            # Add to dataset
            data_item = {
                "id": i,
                "prompt": prompt,
                "embedding": eos_emb.tolist(),
                "label": 0,
                "eos_position": eos_position  # Optional: save EOS position info
            }
            dataset["data"].append(data_item)
    
    # Save final dataset
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to: {save_path}")
    print(f"Total processed {len(dataset['data'])} prompts")
    return dataset

def read_prompts_from_json(file_path):
    """Read prompts from json file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all prompts
    prompts = [item['prompt'] for item in data]
    print(f"Read {len(prompts)} prompts")
    return prompts

def main():
    print("Starting to load configuration and model...")
    # Configuration and model paths
    config_path = "../configs/stable-diffusion/v1-inference.yaml"
    checkpoint_path = "../checkpoint/sd-v1-4-full-ema.ckpt"

    # Load configuration and model
    config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model_from_config(config, f"{checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model loading completed")
    
    # Read prompts from JSON file
    prompts_file = "path_to_your_dataset"  
    print(f"Reading prompts from {prompts_file}...")
    prompts = read_prompts_from_json(prompts_file)
    
    # Create save directory
    save_dir = "embed_dataset"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate dataset
    dataset_path = os.path.join(save_dir, f"path_to_save_dataset")
    dataset = create_labeled_dataset(prompts, model, dataset_path)
    
    print("Processing completed!")

if __name__ == "__main__":
    main()