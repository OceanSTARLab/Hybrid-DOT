from functools import partial
import os
import argparse
import yaml
import time

import torch
import numpy as np
from scipy.io import savemat, loadmat

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.gaussian_diffusion import create_sampler
from util.logger import get_logger

from torch.utils.data import DataLoader, Dataset
#
# from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


class MyDataset(Dataset):
    def __init__(self, mat_file):
        a = loadmat(mat_file)
        a = np.array(a['data']).astype('float32').transpose(0, 3, 1, 2)

        data = torch.Tensor(a)
        data = data.float()
        labels = torch.zeros(data.shape[0], dtype=torch.long)

        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    default_model_diffusion_config = model_and_diffusion_defaults()
    model, diffusion = create_model_and_diffusion(**default_model_diffusion_config)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    sr_factor = measure_config['operator']['sr_factor']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    noise_sigma = measure_config['noise']['sigma']
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, step=diffusion_config['steps'], sr_factor=sr_factor)
   
    # Working directory
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    all_dataset = MyDataset('./data/ssp.mat')
    train_data, eval_data = torch.utils.data.random_split(
        dataset=all_dataset,
        lengths=[7000, 1760],
        generator=torch.Generator().manual_seed(42)
    )
    loader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=0)

    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        ref_img = ref_img[0]
        ref_img = ref_img.to(device)

        y_n = operator.forward(ref_img)
        y_n = noiser(y_n)

        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        time1 = time.time()
        sample, step = sample_fn(x_start=x_start, measurement=y_n, operator=operator)
        time2 = time.time()
        print(f'The sampling time image {i} is {(time2 - time1)}s')

        if i == 0:
            input_grp = y_n.cpu().numpy()
            label_grp = ref_img.cpu().numpy()
            recon_grp = sample.cpu().numpy()
        else:
            input_grp = np.concatenate((input_grp, y_n.cpu().numpy()), 0)
            label_grp = np.concatenate((label_grp, y_n.cpu().numpy()), 0)
            recon_grp = np.concatenate((recon_grp, y_n.cpu().numpy()), 0)
        break

    savemat(
        f'./results/input/ssf-sr{sr_factor}-noise{noise_sigma}-step{step}.mat',
        {'ssf_sr_input': input_grp})
    savemat(
        f'./results/label/ssf-sr{sr_factor}-noise{noise_sigma}-step{step}.mat',
        {'ssf_sr_label': label_grp})
    savemat(
        f'./results/recon/ssf-sr{sr_factor}-noise{noise_sigma}-step{step}.mat',
        {'ssf_sr_recon': recon_grp})


if __name__ == '__main__':
    main()
