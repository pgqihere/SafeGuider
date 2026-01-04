import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import argparse
import os

def load_embeddings_from_json(benign_path, adv1_path, adv2_path):
    # Read data
    with open(benign_path, 'r') as f:
        benign_data = json.load(f)['data']
    with open(adv1_path, 'r') as f:
        adv1_data = json.load(f)['data']
    with open(adv2_path, 'r') as f:
        adv2_data = json.load(f)['data']
    
    # Extract all embeddings
    try:
        benign_embeddings = []
        adv1_embeddings = []
        adv2_embeddings = []
        
        # Process all datasets
        for item in benign_data:
            embedding = item['embedding']
            if isinstance(embedding[0], list):
                benign_embeddings.extend(embedding)
            else:
                benign_embeddings.append(embedding)
        
        for item in adv1_data:
            embedding = item['embedding']
            if isinstance(embedding[0], list):
                adv1_embeddings.extend(embedding)
            else:
                adv1_embeddings.append(embedding)
        
        for item in adv2_data:
            embedding = item['embedding']
            if isinstance(embedding[0], list):
                adv2_embeddings.extend(embedding)
            else:
                adv2_embeddings.append(embedding)
        
        # Convert to numpy arrays
        benign_embeddings = np.array(benign_embeddings)
        adv1_embeddings = np.array(adv1_embeddings)
        adv2_embeddings = np.array(adv2_embeddings)
        
        print(f"Total embeddings in benign dataset: {len(benign_embeddings)}")
        print(f"Total embeddings in VS Attacks dataset: {len(adv1_embeddings)}")
        print(f"Total embeddings in SJ Attacks dataset: {len(adv2_embeddings)}")
        
    except (TypeError, KeyError) as e:
        print("Error processing data:", e)
        raise e
    
    return benign_embeddings, adv1_embeddings, adv2_embeddings

def mmd_distance(X, Y, gamma=None):
    """
    Calculate MMD distance between two groups of samples
    """
    # If gamma is None, use median heuristic method
    if gamma is None:
        # Calculate distances between all sample pairs
        all_samples = np.vstack([X, Y])
        pairwise_distances = euclidean_distances(all_samples, squared=True)
        # Use median heuristic method
        gamma = 1.0 / np.median(pairwise_distances[pairwise_distances > 0])

    # Calculate kernel matrices
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    # Calculate MMD
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    mmd = (np.sum(K_XX) / (n_x * n_x) + 
           np.sum(K_YY) / (n_y * n_y) - 
           2 * np.sum(K_XY) / (n_x * n_y))
    
    return np.sqrt(max(mmd, 0))  # Take square root and ensure non-negative

def calculate_group_distance_matrix_mmd(benign_embeddings, adv1_embeddings, adv2_embeddings):
    """
    Calculate inter-group distance matrix using MMD
    """
    # Standardize data
    scaler = StandardScaler()
    all_data = np.vstack([benign_embeddings, adv1_embeddings, adv2_embeddings])
    scaler.fit(all_data)
    
    benign_scaled = scaler.transform(benign_embeddings)
    adv1_scaled = scaler.transform(adv1_embeddings)
    adv2_scaled = scaler.transform(adv2_embeddings)
    
    groups = {
        'Benign': benign_scaled,
        'VS_Attacks': adv1_scaled,
        'SJ_Attacks': adv2_scaled
    }
    
    labels = list(groups.keys())
    n_groups = len(labels)
    distance_matrix = np.zeros((n_groups, n_groups))
    
    # Calculate MMD distances between all groups
    for i in range(n_groups):
        for j in range(n_groups):
            group1_name = labels[i]
            group2_name = labels[j]
            
            distance = mmd_distance(
                groups[group1_name],
                groups[group2_name]
            )
            distance_matrix[i, j] = distance
    
    # Normalize distance matrix to [0,1] range
    if np.max(distance_matrix) - np.min(distance_matrix) > 0:
        distance_matrix = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix))
    
    return distance_matrix, labels

def save_results_to_json(distance_matrix, labels, output_file):
    """Save MMD distance results to JSON file, append if exists, create if not"""
    
    # Create distance results dictionary
    distance_results = {}
    n = len(labels)
    for i in range(n):
        for j in range(n):
            key = f"{labels[i]}_vs_{labels[j]}"
            distance_results[key] = round(float(distance_matrix[i, j]), 4)
    
    # Create new result entry
    new_entry = {
        'analysis_type': 'MMD_Distance_Matrix',
        'distances': distance_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check if file exists
    if os.path.exists(output_file):
        # Read existing data
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            
            # If existing data is a list, append directly
            if isinstance(existing_data, list):
                existing_data.append(new_entry)
            # If existing data is a dict, convert to list format
            else:
                existing_data = [existing_data, new_entry]
                
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error reading existing file: {e}")
            # If reading fails, create new list
            existing_data = [new_entry]
    else:
        # File doesn't exist, create new list
        existing_data = [new_entry]
    
    # Save updated data
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def analyze_mmd_distances(output_file="mmd_distance_results.json"):
    """Main analysis function"""
    # Replace with your actual file paths
    benign_path = "../../Metrics/Emperical_Study/embedding/benign_testend.json"
    adv1_path = "../../Metrics/Emperical_Study/embedding/nsfw_meta_testend.json"
    adv2_path = "../../Metrics/Emperical_Study/embedding/nsfw_mma_testend.json"
    
    try:
        # Load data
        print("Loading embeddings from JSON files...")
        benign_embeddings, adv1_embeddings, adv2_embeddings = load_embeddings_from_json(
            benign_path, adv1_path, adv2_path
        )
        
        # Calculate MMD distance matrix
        print("Calculating MMD distance matrix...")
        distance_matrix, labels = calculate_group_distance_matrix_mmd(
            benign_embeddings, adv1_embeddings, adv2_embeddings
        )
        
        # Print analysis results
        print("\nMMD Distance Matrix Analysis:")
        n = len(labels)
        for i in range(n):
            for j in range(n):
                print(f"{labels[i]} vs {labels[j]}: {distance_matrix[i,j]:.4f}")
        
        # Save results to JSON
        save_results_to_json(distance_matrix, labels, output_file)
        
        return distance_matrix, labels
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate MMD distances between embedding groups')
    parser.add_argument('--output_file', type=str, default='mmd_distance_results.json',
                       help='Output JSON file to save results (default: mmd_distance_results.json)')
    
    args = parser.parse_args()
    
    try:
        distance_matrix, labels = analyze_mmd_distances(output_file=args.output_file)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
