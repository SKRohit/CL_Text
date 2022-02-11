# CL_Text
This repository is an attempt to apply SCAN LOSS as discussed in [SCAN: Learning to Classify Images without Labels](https://arxiv.org/abs/2005.12320) on text data to perform text classification without any label.

SCAN paper discusses three step appraoch to do this task for images:
- First train a model, on a contrastive loss, to get good feature embeddings.
- Finetune the model obtained in first step on SCAN LOSS (defined in paper).
- Again Finetune the model obtained in second step on Cross-Entropy loss by considering only confident predictions above a threshold as label.

Here instead of training the model, for the first step, from scratch, I am finetuning Pretrained SimCSE. And rest of the steps are followed as it is.

This repo is not complete yet and will be updated frequently. If interested to collaborate feel free to open an issue.
