import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class PreferenceDataset(Dataset):
    def __init__(self, feedback_data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process feedback data
        self.examples = []
        
        for item in feedback_data:
            # Skip examples where no clear preference
            if item["preferred_response"] not in ["A", "B"]:
                continue
                
            query = item["query"]
            chosen = item["response_A"] if item["preferred_response"] == "A" else item["response_B"]
            rejected = item["response_B"] if item["preferred_response"] == "A" else item["response_A"]
            
            self.examples.append({
                "query": query,
                "chosen": chosen,
                "rejected": rejected
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize inputs for reward model
        chosen_input = f"Query: {example['query']}\nResponse: {example['chosen']}"
        rejected_input = f"Query: {example['query']}\nResponse: {example['rejected']}"
        
        chosen_encodings = self.tokenizer(
            chosen_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encodings = self.tokenizer(
            rejected_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_encodings["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encodings["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(),
        }

def load_feedback_data(feedback_csv):
    """Load and prepare feedback data for training"""
    print(f"Loading feedback data from {feedback_csv}")
    df = pd.read_csv(feedback_csv)
    
    # Filter for clear preferences only
    clear_prefs = df[df["preferred_response"].isin(["A", "B"])]
    print(f"Found {len(clear_prefs)} examples with clear preferences out of {len(df)} total")
    
    # Convert to list of dictionaries
    feedback_data = clear_prefs.to_dict("records")
    return feedback_data

def compute_reward_loss(model, batch, device):
    """Compute preference-based reward model loss"""
    chosen_outputs = model(
        input_ids=batch["chosen_input_ids"].to(device),
        attention_mask=batch["chosen_attention_mask"].to(device)
    )
    rejected_outputs = model(
        input_ids=batch["rejected_input_ids"].to(device),
        attention_mask=batch["rejected_attention_mask"].to(device)
    )
    
    chosen_rewards = chosen_outputs.logits
    rejected_rewards = rejected_outputs.logits
    
    # Use a margin loss: max(0, 1 - (chosen_rewards - rejected_rewards))
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    
    # Calculate accuracy (what percentage of times does the model prefer the chosen response)
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, accuracy

def evaluate_model(model, dataloader, device):
    """Evaluate the reward model on a dataset"""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, acc = compute_reward_loss(model, batch, device)
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc

def train_reward_model(
    feedback_data, 
    model_name="distilroberta-base", 
    output_dir="reward_model",
    learning_rate=5e-5,
    batch_size=8,
    num_epochs=3,
    max_length=512,
    train_test_ratio=0.9
):
    """Train a reward model on human preference data"""
    # Initialize tokenizer and model
    print(f"Initializing model from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # Create dataset
    print("Creating preference dataset")
    dataset = PreferenceDataset(feedback_data, tokenizer, max_length=max_length)
    
    # Split into train and validation
    train_size = int(train_test_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} training examples, {val_size} validation examples")
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop
    print(f"Starting reward model training for {num_epochs} epochs")
    
    # For tracking metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, acc = compute_reward_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{acc.item():.4f}"
            })
        
        avg_train_loss = train_loss / num_batches
        avg_train_acc = train_acc / num_batches
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_dataloader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs
    }
    
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    
    print(f"Model saved to {output_dir}")
    return model, tokenizer

def test_reward_model(model_path, test_examples):
    """Test a trained reward model on example pairs"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = []
    
    for example in test_examples:
        query = example["query"]
        response_a = example["response_a"]
        response_b = example["response_b"]
        
        # Score each response
        inputs_a = tokenizer(
            f"Query: {query}\nResponse: {response_a}",
            return_tensors="pt",
            truncation=True
        ).to(device)
        
        inputs_b = tokenizer(
            f"Query: {query}\nResponse: {response_b}",
            return_tensors="pt", 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            score_a = model(**inputs_a).logits.item()
            score_b = model(**inputs_b).logits.item()
        
        preferred = "A" if score_a > score_b else "B"
        margin = abs(score_a - score_b)
        
        results.append({
            "query": query,
            "score_a": score_a,
            "score_b": score_b,
            "preferred": preferred,
            "margin": margin
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train a reward model for RLHF")
    parser.add_argument("--feedback_csv", type=str, default="feedback_data/feedback.csv",
                        help="Path to feedback CSV file")
    parser.add_argument("--model_name", type=str, default="distilroberta-base",
                        help="Base model to use for reward model")
    parser.add_argument("--output_dir", type=str, default="reward_model",
                        help="Directory to save the trained model")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--train_test_ratio", type=float, default=0.9,
                        help="Ratio of training to test data")
    
    args = parser.parse_args()
    
    # Load feedback data
    if not os.path.exists(args.feedback_csv):
        print(f"Feedback file {args.feedback_csv} not found. Please run the feedback collection app first.")
        return
    
    feedback_data = load_feedback_data(args.feedback_csv)
    
    # Check if we have enough data
    if len(feedback_data) < 10:
        print(f"Only {len(feedback_data)} examples found. We recommend at least 100 for meaningful training.")
        proceed = input("Proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Train reward model
    model, tokenizer = train_reward_model(
        feedback_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        train_test_ratio=args.train_test_ratio
    )
    
    print("Reward model training complete!")

if __name__ == "__main__":
    main()