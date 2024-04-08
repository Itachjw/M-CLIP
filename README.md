# M-CLIP
Few-shot Text-based Person Search

# few-shot-text-to-image-person-retrieval
The data and losses of M-CLIP: Multi-view Contrastive Learning for Few-Shot Text-to-Image Person Retrieval

-------data folder:
There are the json files of three datasets for one-shot setting.
The images of three datasets can be download from the official code.

-------loss.py
Compact_Matching:   compact cross-modal matching loss
hard_loss: cross view hard pair maining loss, there is an example with 3 view.

Requirements
we use single RTX4090 24G GPU for training and evaluation.

pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
Prepare Datasets
Download the CUHK-PEDES dataset, ICFG-PEDES dataset, RSTPReid dataset.
Replace the json file.

Training and testing are the same as IRRA(https://github.com/anosorae/IRRA).

This project is based on IRRA(https://github.com/anosorae/IRRA).
Thanks of Jiang and Ye(IRRA).
