---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /home/student2021/srcao/base-model/llama2-7b
metrics:
- accuracy
model-index:
- name: llama2-7b-multi_role-lr=3e-4-warmup=200
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# llama2-7b-multi_role-lr=3e-4-warmup=200

This model is a fine-tuned version of [/home/student2021/srcao/base-model/llama2-7b](https://huggingface.co//home/student2021/srcao/base-model/llama2-7b) on the multi_role dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1126
- Accuracy: 0.8497

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 200
- num_epochs: 1.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.7048        | 0.1   | 10   | 0.7109          | 0.3642   |
| 0.7035        | 0.2   | 20   | 0.6891          | 0.3642   |
| 0.6615        | 0.29  | 30   | 0.6421          | 0.3873   |
| 0.6231        | 0.39  | 40   | 0.5553          | 0.3584   |
| 0.4861        | 0.49  | 50   | 0.3623          | 0.4046   |
| 0.2601        | 0.59  | 60   | 0.1516          | 0.5838   |
| 0.1209        | 0.68  | 70   | 0.1557          | 0.7052   |
| 0.0913        | 0.78  | 80   | 0.1269          | 0.8555   |
| 0.1784        | 0.88  | 90   | 0.1193          | 0.8382   |
| 0.0987        | 0.98  | 100  | 0.1037          | 0.8439   |


### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.0.0+cu117
- Datasets 2.18.0
- Tokenizers 0.15.2