This repository contain code for ICML 25
# Logit redistribution for better domain generalization in low-shot classification with foundation models


We used the official code of ICML24 published paper  [Open-Vocabulary Calibration for Fine-tuned CLIP 🔗](https://arxiv.org/abs/2402.04655) , the code link for their paper is (https://github.com/ml-stat-Sustech/CLIP_Calibration).


## Setup

**1. Installation** 

For packages installation please follow the instructions given in [INSTALL.md](docs/INSTALL.md).

**2. Data preparation**

For dataset preparation please follow the instructions given at [DATASETS.md](docs/DATASETS.md) to prepare all datasets.


## Quick Start

Please refer to ``./run`` for more info about our scripts. 

**1. Tuning & Evaluation** 

```bash
GPU_ID=XXXX # replace it with your GPU ID
bash run/classification/zeroshot.sh ${GPU_ID} # zero-shot CLIP



