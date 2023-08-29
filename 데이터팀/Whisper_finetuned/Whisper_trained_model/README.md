---
language:
- ko
license: apache-2.0
base_model: openai/whisper-base
tags:
- hf-asr-leaderboard
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_13_0
model-index:
- name: oceanstar-bridze
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# oceanstar-bridze

This model is a fine-tuned version of [openai/whisper-base](https://huggingface.co/openai/whisper-base) on the bridzeDataset dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1880
- Cer: 7.3894

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000

### Training results

| Training Loss | Epoch | Step | Cer     | Validation Loss |
|:-------------:|:-----:|:----:|:-------:|:---------------:|
| 0.3652        | 0.06  | 500  | 11.3504 | 0.3574          |
| 0.2788        | 0.13  | 1000 | 9.1325  | 0.2645          |
| 0.2213        | 0.1   | 1500 | 0.2388  | 9.3132          |
| 0.2257        | 0.13  | 2000 | 0.2194  | 8.6295          |
| 0.1941        | 0.16  | 2500 | 0.2068  | 7.5109          |
| 0.1395        | 0.19  | 3000 | 0.1969  | 7.3247          |
| 0.1787        | 0.23  | 3500 | 0.1905  | 7.5517          |
| 0.1639        | 0.26  | 4000 | 0.1880  | 7.3894          |


### Framework versions

- Transformers 4.32.0.dev0
- Pytorch 1.10.1
- Datasets 2.14.2
- Tokenizers 0.13.3
