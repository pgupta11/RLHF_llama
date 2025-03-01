RLHF Llama
A project for fine-tuning Llama models using Reinforcement Learning from Human Feedback (RLHF).
Overview
This project implements RLHF techniques to improve Llama language models by aligning them with human preferences. The approach involves three main components:

Supervised Fine-Tuning (SFT): Initial fine-tuning of the base model on example data.
Reward Modeling: Training a model to predict human preferences.
Reinforcement Learning: Fine-tuning the SFT model using Proximal Policy Optimization (PPO) with the reward model as feedback.

