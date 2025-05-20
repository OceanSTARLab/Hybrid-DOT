# Hybrid-DOT

This repository contains the code for the paper "Hybrid data- and model-driven three-dimensional ocean sound speed field super-resolution: Diffusion model meets low-rank tensor".

## Description

Please find the 3DSSF dataset [here](https://drive.google.com/file/d/1mDZ29nNAQso_4TWC1N1bAjJc3rls8OSR/view?usp=drive_link). For model evaluation, you can either train a new model or use the pre-trained model available at [here](https://drive.google.com/file/d/15P59cckEFyhxVtwhUn3vCF7IbMz7cCae/view?usp=drive_link). Put the 3DSSF data in './data/' and the checkpoint in './ckpt/'.

## Usage

Run the following command to perform SR:
```
python Inference.py --model_path=./ckpt/ssf.pt --diffusion_config=configs/diffusion_config.yaml --task_config=configs/super_resolution_config.yaml
```

## References and Acknowlegments

If you find the code useful for your research, please consider citing
```
@article{sun2025hybrid,
  title={Hybrid data-and model-driven three-dimensional ocean sound speed field super-resolution: Diffusion model meets low-rank tensor},
  author={Sun, Yifan and Li, Siyuan and Fang, Shikai and Cheng, Lei and Li, Jianlong and Gerstoft, Peter},
  journal={The Journal of the Acoustical Society of America},
  volume={157},
  number={5},
  pages={3756--3770},
  year={2025},
  publisher={AIP Publishing}
}
```

The implementation is based on
```
@inproceedings{
chung2023diffusion,
title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```
