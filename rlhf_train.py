import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset as HFDataset
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

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
    df = pd.read_csv(feedback_csv)
    
    # Filter for clear preferences only
    df = df[df["preferred_response"].isin(["A", "B"])]
    
    # Convert to list of dictionaries
    feedback_data = df.to_dict("records")
    return feedback_data

def train_reward_model(feedback_data, model_name="distilroberta-base", output_dir="reward_model"):
    """Train a reward model on human preference data"""
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    # Create dataset
    dataset = PreferenceDataset(feedback_data, tokenizer)
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Define reward model training function
    def compute_reward_loss(model, batch):
        chosen_outputs = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"]
        )
        rejected_outputs = model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"]
        )
        
        chosen_rewards = chosen_outputs.logits
        rejected_rewards = rejected_outputs.logits
        
        # Use a margin loss: max(0, 1 - (chosen_rewards - rejected_rewards))
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return loss
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)
    
    print(f"Starting reward model training with {len(train_dataset)} examples")
    
    num_epochs = 3
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = compute_reward_loss(model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = compute_reward_loss(model, batch)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_dataloader):.4f}, Val Loss: {val_loss/len(val_dataloader):.4f}")
    
    # Save model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def prepare_rlhf_data(feedback_data, tokenizer):
    """Prepare data for RLHF fine-tuning"""
    # Create a dataset suitable for PPO training
    # We'll focus on the preferred responses
    rlhf_data = []
    
    for item in feedback_data:
        if item["preferred_response"] not in ["A", "B"]:
            continue
            
        query = item["query"]
        chosen = item["response_A"] if item["preferred_response"] == "A" else item["response_B"]
        
        rlhf_data.append({
            "query": query,
            "response": chosen
        })
    
    # Convert to HuggingFace dataset
    hf_dataset = HFDataset.from_pandas(pd.DataFrame(rlhf_data))
    return hf_dataset

def train_with_rlhf(base_model_name, reward_model_path, rlhf_dataset, output_dir="rlhf_model"):
    """Fine-tune LLM with RLHF using PPO"""
    # Initialize models
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    
    # Initialize PPO model
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
    ppo_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Ensure padding token is set
    if ppo_tokenizer.pad_token is None:
        ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
    
    # Define reward function
    def reward_fn(query, response):
        inputs = reward_tokenizer(
            f"Query: {query}\nResponse: {response}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            reward_outputs = reward_model(**inputs)
            rewards = reward_outputs.logits.squeeze()
        
        return rewards
    
    # Configure PPO training
    ppo_config = PPOConfig(
        model_name=base_model_name,
        learning_rate=1.41e-5,
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=4,
        seed=42
    )
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        tokenizer=ppo_tokenizer,
        dataset=rlhf_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(ppo_tokenizer, mlm=False)
    )
    
    # Training loop
    for epoch in range(3):  # Start with a small number of epochs
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            # Generate responses with current model
            query_tensors = batch["input_ids"]
            query_texts = [ppo_tokenizer.decode(q) for q in query_tensors]
            
            # Generate responses
            response_tensors = ppo_trainer.generate(query_tensors)
            response_texts = [ppo_tokenizer.decode(r) for r in response_tensors]
            
            # Compute rewards
            rewards = [reward_fn(q, r) for q, r in zip(query_texts, response_texts)]
            rewards_tensor = torch.tensor(rewards)
            
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
            ppo_trainer.log_stats(stats, batch, rewards_tensor)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: {stats}")
    
    # Save the RL fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    ppo_model.save_pretrained(output_dir)
    ppo_tokenizer.save_pretrained(output_dir)
    
    return ppo_model, ppo_tokenizer

def main():
    # Load feedback data
    feedback_csv = "feedback_data/feedback.csv"
    if not os.path.exists(feedback_csv):
        print("No feedback data found. Please run the feedback collection app first.")
        return
    
    feedback_data = load_feedback_data(feedback_csv)
    print(f"Loaded {len(feedback_data)} feedback examples")
    
    # Check if we have enough data
    if len(feedback_data) < 10:
        print("Not enough feedback data for training. Please collect more feedback.")
        return
    
    # Train reward model
    print("Training reward model...")
    reward_model, reward_tokenizer = train_reward_model(feedback_data)
    
    # Prepare RLHF data
    rlhf_dataset = prepare_rlhf_data(feedback_data, reward_tokenizer)
    
    # Fine-tune with RLHF
    # Note: In a real setting, you would replace "gpt2" with your base model
    print("Training with RLHF...")
    rlhf_model, rlhf_tokenizer = train_with_rlhf("gpt2", "reward_model", rlhf_dataset)
    
    print("RLHF training complete!")

if __name__ == "__main__":
    main()