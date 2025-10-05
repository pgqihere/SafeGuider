from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.autograd import Variable
import argparse
import os
from torch.utils.data import TensorDataset
import random
import json
import numpy as np
from torch.backends import cudnn
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix
from tools.classifier import *
from torch.utils.data import random_split


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def args_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--i', type=int)
    args = parser.parse_args()
    return args

def custom_loss(logits, probs, targets):
    safe_scores = probs[:, 1]
    
    pos_mask = (targets == 1)
    pos_loss = torch.mean(F.relu(0.6 - safe_scores[pos_mask]))
    
    neg_mask = (targets == 0)
    neg_loss_upper = torch.mean(F.relu(safe_scores[neg_mask] - 0.4))
    neg_loss_lower = torch.mean(F.relu(0.2 - safe_scores[neg_mask]))
    
    total_loss = pos_loss + neg_loss_upper + neg_loss_lower
    
    return total_loss



def train(model, train_loader, optimizer, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.to(device)

    total_loss = 0.0
    description = "loss={:.4f} acc={:.2f}%"
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with tqdm(train_loader) as batch:
        for input, target in batch:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            
            logits, probs = model(input.to(torch.float32))
            target = target.to(torch.int64)
            
            logits=torch.squeeze(logits,1)
            probs=torch.squeeze(probs,1)
            loss = criterion(logits, target)
            print("loss",loss)
              

            safety_scores = probs[:, 1]  
            print("Logits:", logits[0])  
            print("Probabilities:", probs[0]) 
            print("Sum of probabilities:", probs[0].sum())  
            pred = probs.argmax(dim=1)
            total += target.shape[0]
            total_loss += loss.item()
            correct += pred.eq(target).sum().item()
            
            batch.set_description(f"loss={total_loss/total:.4f} acc={100*correct/total:.2f}%")
     
            loss.backward()
            optimizer.step()

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    safety_scores = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits, probs = model(data.to(torch.float32))
            target = target.to(torch.int64)
            logits=torch.squeeze(logits,1)
            probs=torch.squeeze(probs,1)
            test_loss += criterion(logits, target).item()
            pred = probs.argmax(dim=1)
            total += target.shape[0]
            correct += pred.eq(target).sum().item()
            
            batch_safety_scores = probs[:, 1]  
            safety_scores.extend(batch_safety_scores.cpu().numpy())

    test_loss /= total
    accuracy = 100. * correct / total
    avg_safety_score = np.mean(safety_scores)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%, '
          f'Average Safety Score: {avg_safety_score:.4f}')
    
    return accuracy, test_loss, avg_safety_score

def plot_confusion_matrix(model, test_loader, classes=["Unsafe", "Safe"]):  
    model.eval()  
    y_true = []  
    y_pred = []  
      
    with torch.no_grad(): 
        for data, target in tqdm(test_loader):  
            data, target = data.cuda(), target.cuda()  
            output, _ = model(data.to(torch.float32)) 

            pred = torch.max(output, 1)[1]  
            target = target.squeeze(1)  
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    save_path = "./stablediffusion/metric/safety_confusion_matrix.png"
    plt.savefig(save_path)
    plt.show()  


def run(defensemodel, trainloader, valloader, optimizer, device):
    best_acc = 0.0  

    for epoch in range(50): 
        train(defensemodel, trainloader, optimizer, device)
        

    best_model_path = f'./stable-diffusion-1.5/class_model/test_loss.pt'
    torch.save(defensemodel.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path} with accuracy")


class CustomJsonDataset(Dataset):
    def __init__(self, json_file):

        with open(json_file, 'r') as f:
            json_data = json.load(f)  
            if "data" not in json_data or not isinstance(json_data["data"], list):
                raise ValueError("JSON file must contain a 'data' field with a list of samples.")
            self.data = json_data["data"]  
            
    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
 
        item = self.data[idx] 
        if not isinstance(item, dict):  
            raise ValueError(f"Expected item to be a dictionary, but got {type(item)}")

        embedding = torch.tensor(item['embedding'], dtype=torch.float32) 
        label = torch.tensor(item['label'], dtype=torch.long)  
        return embedding, label

if __name__ == "__main__":
    args = args_parser()
    setup_seed(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    defensemodel = ThreeLayerClassifier(dim=768)
    optimizer = optim.SGD(defensemodel.parameters(), lr=0.001, momentum=0.9)
    defensemodel = defensemodel.to(device)


    train_json_file = 'path_to_your_training_data'  

    train_dataset = CustomJsonDataset(train_json_file)

    label_tensor = torch.tensor([data['label'] for data in train_dataset.data])
    zero_indices = torch.where(label_tensor == 0)[0]    
    non_zero_indices = torch.where(label_tensor != 0)[0]  
    selected_indices = torch.cat((zero_indices, non_zero_indices)) 

    selected_data = [train_dataset[i][0] for i in selected_indices.tolist()]
    selected_label = [train_dataset[i][1] for i in selected_indices.tolist()]
    selected_data = torch.stack(selected_data, dim=0)  
    selected_label = torch.stack(selected_label, dim=0)

    total = TensorDataset(selected_data, selected_label)
    train_size = int(1.0 * len(total))   
    val_size = int(0 * len(total))  
    test_size = len(total) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(total, [train_size, val_size, test_size]) 
    print(f"\n=== Dataset Split Sizes ===")
    print(f"Total dataset size: {len(total)}")
    print(f"Train size: {train_size} ({train_size/len(total)*100:.2f}%)")
    print(f"Val size: {val_size} ({val_size/len(total)*100:.2f}%)")
    print(f"Test size: {test_size} ({test_size/len(total)*100:.2f}%)")

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False) 
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    run(defensemodel, trainloader, valloader, optimizer, device)  
    print('on test set\n')
