import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, FlaubertModel
import torch.nn as nn
from tqdm import tqdm

# Define the device to use for computation (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AllocineDataset(Dataset):
    """
    Custom Dataset for the Allocine movie review dataset.
    """
    def __init__(self, df, model_name_or_path="flaubert/flaubert_base_uncased", max_len=250):
        self.df = df.drop(columns=["film-url"])  # Drop the 'film-url' column
        self.labels = sorted(self.df["polarity"].unique())  # Unique polarity labels
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}  # Label to index mapping
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        review_text = self.df["review"].iloc[index]
        labels = self.labels_dict[self.df["polarity"].iloc[index]]
        inputs = self.tokenizer(review_text, padding="max_length", max_length=self.max_len, truncation=True, return_tensors="pt")
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }

class CustomBert(nn.Module):
    """
    Custom BERT-based model for sentiment classification.
    """
    def __init__(self, name_or_model_path="flaubert/flaubert_base_uncased", num_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = FlaubertModel.from_pretrained(name_or_model_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT model and classifier
        outputs = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Class token output
        x = self.classifier(cls_output)
        return x

def training_step(model, dataloader, loss_fn, optimizer):
    """
    Perform a single training step.
    """
    model.train()
    total_loss = 0
    for data in tqdm(dataloader, total=len(dataloader)):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)
        
        # Forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute loss
        loss = loss_fn(output, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader.dataset)

def evaluation(model, test_dataloader, loss_fn):
    """
    Evaluate the model performance on the test dataset.
    """
    model.eval()
    correct_predictions = 0
    losses = []
    with torch.no_grad():
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["labels"].to(device)
            
            # Forward pass
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate predictions
            _, pred = output.max(1)
            correct_predictions += torch.sum(pred == labels).item()
            
            # Compute loss
            loss = loss_fn(output, labels)
            losses.append(loss.item())
            
    return np.mean(losses), correct_predictions / len(test_dataloader.dataset)

def main():
    """
    Main function to train and evaluate the model.
    """
    # Load datasets
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/valid.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Create datasets and dataloaders
    train_dataset = AllocineDataset(train_df, max_len=100)
    val_dataset = AllocineDataset(val_df, max_len=100)
    test_dataset = AllocineDataset(test_df, max_len=100)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=2)
    
    # Set training parameters
    n_epoch = 8
    model = CustomBert()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.005)
    
    # Training loop
    for epoch in range(n_epoch):
        loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_val, accuracy = evaluation(model, val_dataloader, loss_fn)
        
        print(f"Epoch {epoch+1}/{n_epoch} - Train Loss: {loss_train:.4f} | Val Loss: {loss_val:.4f} | Accuracy: {accuracy:.4f}")
    
    # Evaluation on the test set
    loss_test, accuracy_test = evaluation(model, test_dataloader, loss_fn)
    print(f"Test Loss: {loss_test:.4f} | Test Accuracy: {accuracy_test:.4f}")

    # Save the trained model
    os.makedirs('model', exist_ok=True)  # Create model directory if it doesn't exist
    torch.save(model.state_dict(), 'model/model_flaubert.pth')

if __name__ == "__main__":
    main()
