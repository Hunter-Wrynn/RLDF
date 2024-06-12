# RLDF

The experimental code of paper "RLDF:Reinforcement Learning from Multi-role Debates as Feedback for Bias Mitigation in LLMs".

## Introduction

Welcome to the repository for our paper "RLDF: Reinforcement Learning from Multi-role Debates as Feedback for Bias Mitigation in LLMs". This repository contains the implementation of the RLDF methodology, which is designed to reduce biases in large language models (LLMs) through reinforcement learning guided by multi-role debates.

+ Data Construction Scripts: Tools for generating datasets in both self-reflection and teacher-student modes.
+ Reward Model Training: Implementation of the reward model training process using the constructed datasets.
+ Reinforcement Learning Fine-Tuning: Scripts for fine-tuning the LLM using PPO, guided by the reward model.
  
## Requirements

conda create --name 'RLDF' --file requirements.txt

conda activate 'RLDF'

## Dataset and Models

+ The datasets used in our experiments are constructed through our provided code. You can generate the datasets using ‘API/data.py’ or ‘OSS/data.py’
+ Our main models used in the experiments are Llama2-7B ,ChatGLM3-6B, Qwen1.5-7B and Baichuan2-7B. These models need to be downloaded separately.

## RL
1. Reward Model Training
   
    1.1 `pip install -r llama_factory_requirements.txt`  
        `conda activate llama_factory`  
        `cd reward-model-training`
   
    1.2 Download LLaMA2-7b, Qwen1.5-7b, ChatGLM3-6B, Baichuan2-7B to "base-model" folder
   
    1.3 Run `train_rm_multi_role.sh` and `train_sft_multi_role.sh` separately.
   
    1.4 Run `merge_rm_lora_multi_role.sh` and `merge_sft_lora_multi_role.sh` separately.

3. RL Finetuning
   
    2.1 `pip install -r llm_requirements.txt`  
        `conda activate llm`  
        `cd rl-finetuning`
   
    2.2 Run `rl-finetuning\src\run_llama2_7b_multi_role.sh`

5. Predict
   
    3.1 `conda activate llama_factory`  
        `cd reward-model-training`
   
    3.2 Run `reward-model-training\do_predict_RLDF_multi_role.sh`

