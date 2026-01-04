import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import json

def load_embeddings_from_json(benign_path, adv1_path, adv2_path):
    with open(benign_path, 'r') as f:
        benign_data = json.load(f)['data']
    
    with open(adv1_path, 'r') as f:
        adv1_data = json.load(f)['data']
    
    with open(adv2_path, 'r') as f:
        adv2_data = json.load(f)['data']
    
    try:
        benign_embeddings = []
        adv1_embeddings = []
        adv2_embeddings = []
        
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
        

        benign_embeddings = np.array(benign_embeddings)
        adv1_embeddings = np.array(adv1_embeddings)
        adv2_embeddings = np.array(adv2_embeddings)
        
        print(f"Total embeddings in benign dataset: {len(benign_embeddings)}")
        print(f"Total embeddings in VS Attack dataset: {len(adv1_embeddings)}")
        print(f"Total embeddings in SJ Attack dataset: {len(adv2_embeddings)}")
        
        return benign_embeddings, adv1_embeddings, adv2_embeddings
        
    except (TypeError, KeyError) as e:
        print("Error processing data:", e)
        raise e
    
def plot_multiple_visualizations(benign_embeddings, adv1_embeddings, adv2_embeddings, save_dir='./'):
    n_benign = len(benign_embeddings)
    n_adv1 = len(adv1_embeddings)
    all_embeddings = np.vstack([benign_embeddings, adv1_embeddings, adv2_embeddings])
    
    # 1. t-SNE 
    print("Performing t-SNE...")
    plt.figure(figsize=(10, 8))
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(all_embeddings)
    
    benign_tsne = embeddings_tsne[:n_benign]
    adv1_tsne = embeddings_tsne[n_benign:n_benign+n_adv1]
    adv2_tsne = embeddings_tsne[n_benign+n_adv1:]
    
    plt.scatter(benign_tsne[:, 0], benign_tsne[:, 1], c='blue', label='Benign', alpha=0.6, s=20)
    plt.scatter(adv1_tsne[:, 0], adv1_tsne[:, 1], c='red', label='VS Attack', alpha=0.6, s=20)
    plt.scatter(adv2_tsne[:, 0], adv2_tsne[:, 1], c='green', label='SJ Attack', alpha=0.6, s=20)
    plt.title('t-SNE Visualization')
    plt.legend()
    plt.savefig(f'{save_dir}/tsne_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. UMAP 
    print("Performing UMAP...")
    plt.figure(figsize=(10, 8))
    reducer = umap.UMAP(random_state=42)
    embeddings_umap = reducer.fit_transform(all_embeddings)
    
    benign_umap = embeddings_umap[:n_benign]
    adv1_umap = embeddings_umap[n_benign:n_benign+n_adv1]
    adv2_umap = embeddings_umap[n_benign+n_adv1:]
    
    plt.scatter(benign_umap[:, 0], benign_umap[:, 1], c='blue', label='Benign', alpha=0.6, s=20)
    plt.scatter(adv1_umap[:, 0], adv1_umap[:, 1], c='red', label='VS Attack', alpha=0.6, s=20)
    plt.scatter(adv2_umap[:, 0], adv2_umap[:, 1], c='green', label='SJ Attack', alpha=0.6, s=20)
    plt.title('UMAP Visualization')
    plt.legend()
    plt.savefig(f'{save_dir}/umap_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. PCA 
    print("Performing PCA...")
    plt.figure(figsize=(10, 8))
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(all_embeddings)
    
    benign_pca = embeddings_pca[:n_benign]
    adv1_pca = embeddings_pca[n_benign:n_benign+n_adv1]
    adv2_pca = embeddings_pca[n_benign+n_adv1:]
    
    for data, color, label in [(benign_pca, 'blue', 'Benign'),
                              (adv1_pca, 'red', 'VS Attack'),
                              (adv2_pca, 'green', 'SJ Attack')]:
        xy = np.vstack([data[:,0], data[:,1]])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = data[idx,0], data[idx,1], z[idx]
        scatter = plt.scatter(x, y, c=z, 
                            cmap=plt.cm.get_cmap('Blues' if color=='blue' else 'Reds' if color=='red' else 'Greens'),
                            label=label, alpha=0.6, s=20)
        plt.colorbar(scatter)
    
    plt.title('PCA with Density Visualization')
    plt.legend()
    plt.savefig(f'{save_dir}/pca_density_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 3D PCA
    print("Performing 3D PCA...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    pca3d = PCA(n_components=3)
    embeddings_pca3d = pca3d.fit_transform(all_embeddings)
    
    benign_pca3d = embeddings_pca3d[:n_benign]
    adv1_pca3d = embeddings_pca3d[n_benign:n_benign+n_adv1]
    adv2_pca3d = embeddings_pca3d[n_benign+n_adv1:]
    
    ax.scatter(benign_pca3d[:, 0], benign_pca3d[:, 1], benign_pca3d[:, 2], 
              c='blue', label='Benign', alpha=0.6, s=20)
    ax.scatter(adv1_pca3d[:, 0], adv1_pca3d[:, 1], adv1_pca3d[:, 2], 
              c='red', label='VS Attack', alpha=0.6, s=20)
    ax.scatter(adv2_pca3d[:, 0], adv2_pca3d[:, 1], adv2_pca3d[:, 2], 
              c='green', label='SJ Attack', alpha=0.6, s=20)
    
    ax.set_title('3D PCA Visualization')
    ax.legend()
    ax.view_init(elev=20, azim=45)
    plt.savefig(f'{save_dir}/pca_3d_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nPCA Explained variance ratio:")
    print(f"First component: {explained_variance_ratio[0]:.4f}")
    print(f"Second component: {explained_variance_ratio[1]:.4f}")
    print(f"Total explained variance: {sum(explained_variance_ratio):.4f}")

if __name__ == "__main__":
    benign_path = "../../Metrics/Emperical_Study/embedding/benign_testend.json"
    adv1_path = "../../Metrics/Emperical_Study/embedding/nsfw_meta_testend.json"
    adv2_path = "../../Metrics/Emperical_Study/embedding/nsfw_mma_testend.json"

    try:
        benign_embeddings, adv1_embeddings, adv2_embeddings = load_embeddings_from_json(
            benign_path, adv1_path, adv2_path
        )

        save_dir = '../../Results/Empirical_study/visualization_results'  
        plot_multiple_visualizations(benign_embeddings, adv1_embeddings, adv2_embeddings, 
                                   save_dir=save_dir)
                                       
    except Exception as e:
        print(f"An error occurred: {str(e)}")