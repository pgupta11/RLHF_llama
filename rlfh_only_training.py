import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset as HFDataset
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import wandb
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"rlhf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def load_feedback_data(feedback_csv, reward_model_path=None):
    """Load feedback data and optionally verify reward model predictions"""
    df = pd.read_csv(feedback_csv)
    
    # Filter for clear preferences only
    df = df[df["preferred_response"].isin(["A", "B"])]
    
    # Convert to list of dictionaries
    feedback_data = df.to_dict("records")
    
    # If reward model path is provided, analyze how well it matches human preferences
    if reward_model_path:
        validate_reward_model(reward_model_path, feedback_data)
        
    return feedback_data

def validate_reward_model(model_path, feedback_data, sample_size=50):
    """Validate how well the reward model matches human preferences"""
    logger.info(f"Validating reward model from {model_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Sample data if needed
    if len(feedback_data) > sample_size:
        import random
        samples = random.sample(feedback_data, sample_size)
    else:
        samples = feedback_data
        
    logger.info(f"Validating on {len(samples)} examples")
    
    correct = 0
    total = 0
    
    for item in tqdm(samples, desc="Validating reward model"):
        query = item["query"]
        response_a = item["response_A"]
        response_b = item["response_B"]
        human_preference = item["preferred_response"]
        
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
        
        model_preference = "A" if score_a > score_b else "B"
        
        # Check if model agrees with human
        if model_preference == human_preference:
            correct += 1
        total += 1
    
    agreement = correct / total
    logger.info(f"Reward model agreement with human preferences: {agreement:.4f} ({correct}/{total})")
    
    return agreement

def prepare_rlhf_data(feedback_data, tokenizer, max_length=512):
    """Prepare data for RLHF fine-tuning"""
    # Create a dataset focused on the preferred responses
    rlhf_data = []
    
    for item in feedback_data:
        if item["preferred_response"] not in ["A", "B"]:
            continue
            
        query = item["query"]
        chosen = item["response_A"] if item["preferred_response"] == "A" else item["response_B"]
        
        rlhf_data.append({
            "query": query,
            "response": chosen,
            "text": f"Query: {query}\nResponse: {chosen}"
        })
    
    # Convert to HuggingFace dataset
    hf_dataset = HFDataset.from_pandas(pd.DataFrame(rlhf_data))
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def create_reward_model_fn(reward_model_path):
    """Create a reward computation function using the trained reward model"""
    # Load reward model and tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model.to(device)
    reward_model.eval()
    
    def reward_fn(query_response_pairs):
        """
        Compute rewards for a batch of query-response pairs
        Args:
            query_response_pairs: List of (query, response) tuples
        Returns:
            Tensor of reward scores
        """
        rewards = []
        
        for query, response in query_response_pairs:
            # Tokenize input
            inputs = reward_tokenizer(
                f"Query: {query}\nResponse: {response}",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Compute reward
            with torch.no_grad():
                reward = reward_model(**inputs).logits.squeeze()
            
            rewards.append(reward.item())
        
        return torch.tensor(rewards)
    
    return reward_fn

def train_with_rlhf(
    base_model_name, 
    reward_model_path, 
    rlhf_dataset, 
    output_dir="rlhf_model",
    use_lora=True,
    learning_rate=1.41e-5,
    batch_size=8,
    mini_batch_size=4,
    ppo_epochs=4,
    epochs=3,
    max_steps=None,
    target_kl=0.1,
    use_wandb=False
):
    """Fine-tune LLM with RLHF using PPO"""
    # Initialize PPO model
    ppo_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Ensure padding token is set
    if ppo_tokenizer.pad_token is None:
        if ppo_tokenizer.eos_token:
            ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
        else:
            ppo_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Initialize the model with or without LoRA
    if use_lora:
        logger.info(f"Using LoRA for fine-tuning {base_model_name}")
        
        # Initialize base model
        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out"]
        )
        
        # Apply LoRA
        ppo_model = get_peft_model(base_model, lora_config)
        ppo_model.print_trainable_parameters()
    else:
        logger.info(f"Loading model {base_model_name} for full fine-tuning")
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_name)
    
    # Create reward function
    reward_fn = create_reward_model_fn(reward_model_path)
    
    # Configure PPO training
    ppo_config = PPOConfig(
        model_name=base_model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=target_kl,
        ppo_epochs=ppo_epochs,
        max_grad_norm=1.0,
        seed=42
    )
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="llama-rlhf",
            name=f"rlhf-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "base_model": base_model_name,
                "reward_model": reward_model_path,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "ppo_epochs": ppo_epochs,
                "target_kl": target_kl,
                "use_lora": use_lora
            }
        )
    
    # Create PPO trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=ppo_model,
        tokenizer=ppo_tokenizer,
        dataset=rlhf_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(ppo_tokenizer, mlm=False)
    )
    
    device = trainer.accelerator.device
    
    # Save all metrics for later analysis
    all_metrics = {
        "objective/kl": [],
        "objective/kl_dist": [],
        "objective/logprobs": [],
        "objective/ref_logprobs": [],
        "objective/kl_coef": [],
        "objective/entropy": [],
        "ppo/mean_non_score_reward": [],
        "ppo/mean_scores": [],
        "ppo/std_scores": [],
        "ppo/min_scores": [],
        "ppo/max_scores": [],
    }
    
    # Create a demo query for testing throughout training
    demo_query = "How do I annotate a genome in KBase?"
    
    # Store generated responses throughout training
    demo_responses = []
    
    # Training loop
    total_steps = epochs * len(trainer.dataloader) if max_steps is None else max_steps
    progress_bar = tqdm(total=total_steps, desc="Training with RLHF")
    
    step_count = 0
    best_mean_reward = float("-inf")
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(trainer.dataloader):
            # Extract queries
            query_tensors = batch["input_ids"]
            query_texts = [ppo_tokenizer.decode(q) for q in query_tensors]
            
            # Extract actual queries without the "Query: " prefix
            queries = []
            for text in query_texts:
                if "Query: " in text and "\nResponse: " in text:
                    query_part = text.split("Query: ")[1].split("\nResponse: ")[0]
                    queries.append(query_part)
                else:
                    queries.append(text)
            
            # Generate responses with current model
            response_tensors = []
            
            for query in tqdm(queries, desc=f"Generating responses (batch {batch_idx})", leave=False):
                inputs = ppo_tokenizer(f"Query: {query}\nResponse:", return_tensors="pt").to(device)
                outputs = trainer.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=ppo_tokenizer.pad_token_id
                )
                response_tensors.append(outputs[0])
            
            # Decode responses
            response_texts = []
            for response_tensor in response_tensors:
                response_text = ppo_tokenizer.decode(response_tensor, skip_special_tokens=True)
                # Extract just the response part
                if "Response:" in response_text:
                    response_part = response_text.split("Response:")[1].strip()
                    response_texts.append(response_part)
                else:
                    response_texts.append(response_text)
            
            # Create query-response pairs for reward computation
            query_response_pairs = list(zip(queries, response_texts))
            
            # Compute rewards using reward model
            rewards = reward_fn(query_response_pairs)
            
            # Run PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)
            
            # Log stats
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: "
                      f"Mean score: {stats['ppo/mean_scores']:.4f}, "
                      f"KL: {stats['objective/kl']:.4f}")
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'mean_score': f"{stats['ppo/mean_scores']:.4f}",
                'kl': f"{stats['objective/kl']:.4f}"
            })
            
            # Track metrics
            for key in all_metrics:
                if key in stats:
                    all_metrics[key].append(stats[key])
            
            # Log to wandb if enabled
            if use_wandb:
                wandb.log(stats)
            
            # Generate demo response periodically
            if batch_idx % 10 == 0:
                demo_inputs = ppo_tokenizer(f"Query: {demo_query}\nResponse:", return_tensors="pt").to(device)
                with torch.no_grad():
                    demo_outputs = trainer.model.generate(
                        **demo_inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=ppo_tokenizer.pad_token_id
                    )
                demo_response = ppo_tokenizer.decode(demo_outputs[0], skip_special_tokens=True)
                if "Response:" in demo_response:
                    demo_response = demo_response.split("Response:")[1].strip()
                
                demo_responses.append({
                    "step": step_count,
                    "response": demo_response
                })
                
                logger.info(f"Demo response at step {step_count}:\n{demo_response}\n")
                
                if use_wandb:
                    wandb.log({"demo_response": demo_response})
            
            # Save best model based on mean reward
            if stats['ppo/mean_scores'] > best_mean_reward:
                best_mean_reward = stats['ppo/mean_scores']
                # Save best model
                best_model_dir = os.path.join(output_dir, "best_model")
                os.makedirs(best_model_dir, exist_ok=True)
                trainer.model.save_pretrained(best_model_dir)
                ppo_tokenizer.save_pretrained(best_model_dir)
                logger.info(f"Saved best model with mean reward {best_mean_reward:.4f}")
            
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                logger.info(f"Reached max steps {max_steps}. Stopping training.")
                break
                
        if max_steps is not None and step_count >= max_steps:
            break
    
    progress_bar.close()
    
    # Save the final model
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    ppo_tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save demo responses
    demo_file = os.path.join(output