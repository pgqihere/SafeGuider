import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from safetensors.torch import load_file
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torch.nn import CosineSimilarity

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import sys
sys.path.append('../../Models/stable-diffusion-1.4')
from tools.classifier import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from tools.classifier import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime 
import json

def get_embedding_dim(model):
    try:
        test_prompt = ["test"]
        with torch.no_grad():
            embeddings = model.cond_stage_model(test_prompt)
            return embeddings.shape[-1]
    except:
        return 768
    


def get_last_token_embedding(tokens, embeddings):
    """Get the embedding of the last token"""
    last_token_pos = (tokens != 0).sum(dim=1) - 1
    batch_size = embeddings.shape[0]
    last_embeddings = embeddings[torch.arange(batch_size), last_token_pos]
    return last_embeddings

def calculate_semantic_similarity(orig_embeddings, modified_embeddings):
    cos = CosineSimilarity(dim=1)
    similarity = cos(orig_embeddings, modified_embeddings)
    return similarity

def compute_safety_score(safety_classifier, embedding):
        safety_classifier.eval() 
        logits, probs = safety_classifier(embedding.unsqueeze(0))
        probs=torch.squeeze(probs,1)
        predictions = probs.argmax(dim=1)
        if len(probs.shape) == 3:
            probs = probs.squeeze(1)

        safety_score = probs[:, 1] 
        print("prediction",predictions)
        return safety_score

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="../../Images/SDv1.4/Sexual/Safeguider/META"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    opt = parser.parse_args()
    
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = model.to(device)

    embedding_dim = get_embedding_dim(model)
    classifier = ThreeLayerClassifier(embedding_dim).to(device)
    
    # Load classifier parameters
    try:
        classifier_state = torch.load("../../Models/SD1.4_safeguider.pt", map_location=device)
        classifier.load_state_dict(classifier_state)
        print("Successfully loaded classifier model")
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return
    
    classifier.eval()
    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if opt.from_file:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r", encoding='utf-8') as f:
            json_data = json.load(f)
            prompts = [item['prompt'] for item in json_data]
            data = []
            for prompt in prompts:
                data.append(batch_size * [prompt])
                
           
    else:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                max_length = 77
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        print(f"\nCurrent iteration {n+1}/{opt.n_iter}")
                        print(f"Processing prompts:")
                        for i, prompt in enumerate(prompts):
                            print(f"Batch Prompt {i+1}: {prompt}")

                        c_original = model.get_learned_conditioning(prompts)
                        original_embeddings = c_original  

                        tokens = model.cond_stage_model.tokenizer.encode(prompts[0])  
                        tokens = torch.tensor(tokens).squeeze(0)
                        if len(tokens) > max_length:
                            tokens = tokens[:max_length]

                        eos_token_id = 49407
                        eos_positions = (tokens == eos_token_id).nonzero()
                        eos_position = eos_positions[0].item() if len(eos_positions) > 0 else min(len(tokens) - 1, max_length - 1)

                        original_eos_embedding = original_embeddings[:1, eos_position, :]
                        classifier.eval()
                        _, probs = classifier(original_eos_embedding.unsqueeze(0))
                        predictions = torch.argmax(probs, dim=-1)

                        if predictions == 1:
                            print("Original prompt is safe, proceeding with original embeddings")
                            c = original_embeddings
                        else:
                            tokens_list = [model.cond_stage_model.tokenizer.encode(p) for p in prompts]

                            modified_tokens = []
                            modified_prompts = []
                            for i, tokens in enumerate(tokens_list):
                                original_words = prompts[i].split()
                                best_modified_prompt = None
                                best_safety_improvement = 0
                                best_similarity = 0
                                best_tokens_removed = []
                                
                                print(f"\nProcessing prompt {i+1}: {prompts[i]}")
                                original_embedding = model.get_learned_conditioning([prompts[i]])
                                tokens = model.cond_stage_model.tokenizer.encode(prompts[i])
                                tokens = torch.tensor(tokens)  
                                tokens = tokens.squeeze(0)

                                if len(tokens) > max_length:
                                    tokens = tokens[:max_length]

                                eos_token_id = 49407
                                eos_positions = (tokens == eos_token_id).nonzero()
                                if len(eos_positions) > 0:
                                    print("Found the eos token!")
                                    eos_position = eos_positions[0].item()
                                else:
                                    eos_position = min(len(tokens) - 1, max_length - 1)
                                print("eos_position:",eos_position)
                                original_eos_embedding = original_embedding[:, eos_position, :]
                                original_safety = compute_safety_score(classifier, original_eos_embedding)
                                print(f"Original safety score: {original_safety.item():.4f}")

                                token_impacts = []
                                for idx in range(len(original_words)):
                                    current_words = original_words.copy()
                                    current_words.pop(idx)
                                    current_prompt = " ".join(current_words)
                                    
                                    current_embedding = model.get_learned_conditioning([current_prompt])
                                    current_tokens = model.cond_stage_model.tokenizer.encode(current_prompt)
                                    current_tokens = torch.tensor(current_tokens) 
                                    current_tokens = current_tokens.squeeze(0)
                                    
                                    if len(current_tokens) > max_length:
                                        current_tokens = current_tokens[:max_length]

                                    eos_token_id = 49407
                                    current_eos_positions = (current_tokens == eos_token_id).nonzero()
                                    if len(current_eos_positions) > 0:
                                        print("Found the eos token!")
                                        current_eos_position = current_eos_positions[0].item()
                                    else:
                                        current_eos_position = min(len(current_tokens) - 1, max_length - 1)
                                    
                                    current_eos_embedding = current_embedding[:, current_eos_position, :]
                                    current_safety = compute_safety_score(classifier, current_eos_embedding)
                                    
                                    safety_improvement = current_safety.item() - original_safety.item()
                                    token_impacts.append((idx, safety_improvement))

                                token_impacts.sort(key=lambda x: x[1], reverse=True)

                                beam_width = 6
                                max_depth = min(25,len(original_words) - 1)  
                                candidates = [([], 0, 1.0)]  

                                print("\nStarting beam search with width:", beam_width)
                                print("Original prompt:", prompts[i])
                                print("Original safety score:", original_safety.item())

                                all_new_candidates_all_depth = []
                                for depth in range(max_depth):
                                    print(f"\nDepth {depth + 1}:")
                                    all_new_candidates = [] 
                                    qualified_candidates = []  
                                    
                                    for candidate_idx, (removed_indices, current_improvement, current_similarity) in enumerate(candidates):
                                        print(f"\nExpanding candidate {candidate_idx + 1}:")
                                        print(f"Current removed tokens: {[original_words[idx] for idx in removed_indices]}")
                                        print(f"Current safety improvement: {current_improvement:.4f}")
                                        print(f"Current similarity: {current_similarity:.4f}")
                                        
                                        for idx, impact in token_impacts:
                                            if idx not in removed_indices:
                                                new_indices = removed_indices + [idx]
                                                
                                                current_words = original_words.copy()
                                                for remove_idx in sorted(new_indices, reverse=True):
                                                    current_words.pop(remove_idx)
                                                
                                                if not current_words: 
                                                    continue
                                                    
                                                current_prompt = " ".join(current_words)
                                                current_words = original_words.copy()

                                                current_embedding = model.get_learned_conditioning([current_prompt])
                                                current_tokens = model.cond_stage_model.tokenizer.encode(current_prompt)
                                                current_tokens = torch.tensor(current_tokens)  # Convert to tensor
                                                current_tokens = current_tokens.squeeze(0)
                                                if len(current_tokens) > max_length:
                                                    current_tokens = current_tokens[:max_length]

                                                eos_token_id = 49407
                                                current_eos_positions = (current_tokens == eos_token_id).nonzero()
                                                if len(current_eos_positions) > 0:
                                                    print("Found the eos token!")
                                                    current_eos_position = current_eos_positions[0].item()
                                                else:
                                                    current_eos_position = min(len(current_tokens) - 1, max_length - 1)
                                                
                                                current_eos_embedding = current_embedding[:, current_eos_position, :]
                                                current_safety = compute_safety_score(classifier, current_eos_embedding)
                                                
                                                similarity = calculate_semantic_similarity(
                                                    original_eos_embedding,
                                                    current_eos_embedding
                                                ).item()
                                                
                                                safety_improvement = current_safety.item() - original_safety.item()
                                                
                                                print(f"\nTrying to remove token '{original_words[idx]}':")
                                                print(f"Modified prompt: {current_prompt}")
                                                print(f"Safety score: {current_safety.item():.4f}")
                                                print(f"Safety improvement: {safety_improvement:.4f}")
                                                print(f"Similarity: {similarity:.4f}")
                                                
                                                all_new_candidates.append((new_indices, safety_improvement, similarity, current_safety.item()))
                                                all_new_candidates_all_depth.append((new_indices, safety_improvement, similarity, current_safety.item()))
                                                
                                                if current_safety.item() >= 0.95 and similarity >= 0.1:
                                                    qualified_candidates.append((new_indices, safety_improvement, similarity))
                                                    print("Added to qualified candidates")
                                                    
                                                    if (not best_modified_prompt or 
                                                        safety_improvement > best_safety_improvement or 
                                                        (safety_improvement == best_safety_improvement and len(new_indices) < len(best_tokens_removed))):
                                                        best_modified_prompt = current_prompt
                                                        best_safety_improvement = safety_improvement
                                                        best_similarity = similarity
                                                        best_tokens_removed = [original_words[idx] for idx in new_indices]
                                                        print("Updated best solution")

                                    if qualified_candidates:  # If there are qualified candidates
                                        candidates = sorted(qualified_candidates, key=lambda x: (x[1], -len(x[0])))[-beam_width:]
                                        print("\nUsing qualified candidates:")
                                    else:  # If no qualified candidates, use those with highest safety scores
                                        # Note: Need to remove the last safety_score value to maintain tuple format consistency
                                        candidates = [(indices, improvement, sim) for indices, improvement, sim, _ in 
                                                    sorted(all_new_candidates, key=lambda x: (x[3], -len(x[0])))[-beam_width:]]
                                        print("\nNo qualified candidates found, using candidates with highest safety scores:")

                                    print(f"\nBest {beam_width} candidates at depth {depth + 1}:")
                                    for idx, (indices, improvement, sim) in enumerate(candidates):
                                        removed_tokens = [original_words[i] for i in indices]
                                        print(f"Candidate {idx + 1}:")
                                        print(f"Removed tokens: {removed_tokens}")
                                        print(f"Safety improvement: {improvement:.4f}")
                                        print(f"Similarity: {sim:.4f}")
                                    
                                    # If satisfactory solution is found, can exit early
                                    if best_modified_prompt and (best_safety_improvement+original_safety.item()) >= 0.95:
                                        print("\nFound satisfactory solution, stopping search")
                                        break

                                if best_modified_prompt:
                                    print(f"\nFound best solution:")
                                    print(f"Tokens removed: {best_tokens_removed}")
                                    print(f"Modified prompt: {best_modified_prompt}")
                                    print(f"Safety improvement: {best_safety_improvement:.4f}")
                                    print(f"Similarity: {best_similarity:.4f}")
                                    modified_prompts.append(best_modified_prompt)
                                else:
                                    print("\nNo suitable token combination found that satisfies the requirements.")
                                    print("Selecting the candidate with highest safety improvement while maintaining similarity >= 0.75:")
                                    
                                    valid_candidates = [candidate for candidate in all_new_candidates_all_depth 
                                                    if candidate[2] >= 0.1]  # candidate[2] is similarity
                                    
                                    if valid_candidates:
                                        best_candidate = max(valid_candidates, key=lambda x: x[3])  # x[3] is safety_score
                                        best_indices, _, best_sim, best_safety = best_candidate
                                        
                                        final_words = original_words.copy()
                                        for idx in sorted(best_indices, reverse=True):
                                            final_words.pop(idx)
                                        best_alternative = " ".join(final_words)
                                        
                                        print(f"Selected prompt: {best_alternative}")
                                        print(f"Removed tokens: {[original_words[idx] for idx in best_indices]}")
                                        print(f"Safety score: {best_safety:.4f}")
                                        print(f"Similarity: {best_sim:.4f}")
                                        
                                        modified_prompts.append(best_alternative)
                                    else:
                                        print("No candidates found with similarity >= 0.1. Using original prompt.")
                                        modified_prompts.append(prompts[i])
                            if modified_prompts:
                                print("\nFinal modified prompts:", modified_prompts)
                                c = model.get_learned_conditioning(modified_prompts)
                                modified_embeddings = c

                                # Get and print final safety scores
                                for i in range(len(prompts)):
                                    original_tokens = model.cond_stage_model.tokenizer.encode(prompts[i])
                                    original_tokens = torch.tensor(original_tokens)  # Convert to tensor
                                    original_tokens = original_tokens.squeeze(0)
                                    if len(original_tokens) > max_length:
                                        original_tokens = original_tokens[:max_length]

                                    eos_token_id = 49407
                                    original_eos_positions = (original_tokens == eos_token_id).nonzero()
                                    if len(original_eos_positions) > 0:
                                        print("Found the eos token!")
                                        original_eos_position = original_eos_positions[0].item()
                                    else:
                                        original_eos_position = min(len(original_tokens) - 1, max_length - 1)

                                    modified_tokens = model.cond_stage_model.tokenizer.encode(modified_prompts[i])
                                    modified_tokens = torch.tensor(modified_tokens)  # Convert to tensor
                                    modified_tokens = modified_tokens.squeeze(0)

                                    if len(modified_tokens) > max_length:
                                        modified_tokens = modified_tokens[:max_length]

                                    modified_eos_positions = (modified_tokens == eos_token_id).nonzero()
                                    if len(modified_eos_positions) > 0:
                                        modified_eos_position = modified_eos_positions[0].item()
                                    else:
                                        modified_eos_position = min(len(modified_tokens) - 1, max_length - 1)

                                    # Get EOS token embeddings
                                    original_eos_embedding = original_embeddings[i:i+1, original_eos_position, :]
                                    modified_eos_embedding = modified_embeddings[i:i+1, modified_eos_position, :]

                                    # Calculate safety scores and similarity
                                    original_score = compute_safety_score(classifier, original_eos_embedding)
                                    modified_score = compute_safety_score(classifier, modified_eos_embedding)
                                    similarity = calculate_semantic_similarity(original_eos_embedding, 
                                                                            modified_eos_embedding)

                                    print(f"\nFinal results for prompt {i+1}:")
                                    print(f"Original: {prompts[i]}")
                                    print(f"Modified: {modified_prompts[i]}")
                                    print(f"Semantic similarity: {similarity[0].item():.4f}")
                                    print(f"Safety scores - Original: {original_score.item():.4f}, Modified: {modified_score.item():.4f}")

                                c = modified_embeddings
                            else:
                                print("No valid prompts to modify")
                                c = original_embeddings      

                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            prompt_group_idx = data.index(prompts)
                                
                            for img_idx, x_sample in enumerate(x_checked_image_torch):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                
                                filename = f"{base_count:05}_round{n}_prompt{prompt_group_idx}_image{img_idx}.png"
                                img.save(os.path.join(sample_path, filename))
                                base_count += 1

                        print("Model device:", next(model.parameters()).device)
                        print("CUDA memory:", torch.cuda.memory_allocated())
                        print("samples_ddim shape:", samples_ddim.shape)
                        print("x_samples_ddim shape:", x_samples_ddim.shape)
                        print("x_checked_image_torch shape:", x_checked_image_torch.shape)
                        
                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
    
if __name__ == "__main__":
    main()