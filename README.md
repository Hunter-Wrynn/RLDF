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
+ Our main models used in the experiments are llama2-7B and chatglm-6B. These models need to be downloaded separately.
