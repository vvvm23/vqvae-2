# Generating Diverse High-Fidelity Images with VQ-VAE-2 [Work in Progress]
PyTorch implementation of Hierarchical, Vector Quantized, Variational Autoencoders (VQ-VAE-2) 
from the paper "Generating Diverse High-Fidelity Images with VQ-VAE-2"

Original paper can be found [here](https://arxiv.org/abs/1906.00446)

Vector Quantizing layer based off implementation by [@rosinality](https://github.com/rosinality) 
found [here](https://github.com/rosinality/vq-vae-2-pytorch).

Aiming for a focus on supporting an arbitrary number of VQ-VAE "levels". Most implementations in 
PyTorch typically only use 2 which is limiting at higher resolutions. This repository contains 
checkpoints for a 3-level and 5-level VQ-VAE-2, trained on FFHQ1024.

This project will not only contain the VQ-VAE-2 architecture, but also an example autoregressive 
prior and latent dataset extraction.

## Usage
### VQ-VAE-2 Usage
Run VQ-VAE-2 training using the config `task_name` found in `hps.py`. Defaults to `cifar10`:
```
python main-vqvae.py --task task_name
```

Evaluate VQ-VAE-2 from parameters `state_dict_path` on task `task_name`. Defaults to `cifar10`:
```
python main-vqvae.py --task task_name --load-path state_dict_path --evaluate
```

Other useful flags:
```
--no-save       # disables saving of files during training
--cpu           # do not use GPU
--batch-size    # overrides batch size in cfg.py, useful for evaluating on larger batch size
--no-tqdm       # disable tqdm status bars
--no-save       # disables saving of files
--no-amp        # disables using native AMP (Automatic Mixed Precision) operations
--save-jpg      # save all images as jpg instead of png, useful for extreme resolutions
```

### Latent Dataset Generation
Run latent dataset generation using VQ-VAE-2 saved at `path` that was trained on task `task_name`. Defaults to `cifar10`:
```
python generate-latents.py path --task task_name
```
Result is saved in `latent-data` directory.

Other useful flags:
```
--cpu           # do not use GPU
--batch-size    # overrides batch size in cfg.py, useful for evaluating on larger batch size
--no-tqdm       # disable tqdm status bars
--no-save       # disables saving of files
--no-amp        # disables using native AMP (Automatic Mixed Precision) operations
```

### Discrete Prior Usage
Run level `level` PixelSnail discrete prior training using the config `task_name` found in `hps.py` using latent dataset saved at path `latent_dataset.pt`. Defaults to `cifar10`:
```
python main-pixelsnail.py latent_dataset.pt level --task task_name
```

Other useful flags:
```
--cpu           # do not use GPU
--load-path     # resume from saved state on disk
--batch-size    # overrides batch size in cfg.py, useful for evaluating on larger batch size
--no-tqdm       # disable tqdm status bars
--no-save       # disables saving of files
--no-amp        # disables using native AMP (Automatic Mixed Precision) operations
```

### Sample Generation 
`TODO: this.`

## Modifications
- Replacing residual layers with ReZero layers.

## Samples
*Reconstructions from FFHQ1024 using a 3-level VQ-VAE-2*
![Reconstructions from FFHQ1024 using a 3-level VQ-VAE-2](recon-example.png)

## Checkpoints
[FFHQ1024 - 3-level VQ-VAE-2](ffhq1024-state-dict-0017.pt)

[FFHQ1024 - 5-level VQ-VAE-2](ffhq1024-large-state-dict-0010.pt)

### Roadmap
- [ ] Prettier outputs
- [X] Server mode (no fancy printing)
- [X] Experiment directories (containing logs / checkpoints / etc)
- [ ] Output logging
- [X] Accumulated gradient training (for larger batch sizes on limited resources)
- [X] Samples and checkpoints on FFHQ1024
- [X] Latent dataset generation
- [X] Autoregressive prior models / training scripts
- [X] Full system sampling

### Citations
**Generating Diverse High-Fidelity Images with VQ-VAE-2**
```
@misc{razavi2019generating,
      title={Generating Diverse High-Fidelity Images with VQ-VAE-2}, 
      author={Ali Razavi and Aaron van den Oord and Oriol Vinyals},
      year={2019},
      eprint={1906.00446},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**PixelSNAIL: An Improved Autoregressive Generative Model**
```
@misc{chen2017pixelsnail,
      title={PixelSNAIL: An Improved Autoregressive Generative Model}, 
      author={Xi Chen and Nikhil Mishra and Mostafa Rohaninejad and Pieter Abbeel},
      year={2017},
      eprint={1712.09763},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**ReZero is All You Need: Fast Convergence at Large Depth**
```
@misc{bachlechner2020rezero,
      title={ReZero is All You Need: Fast Convergence at Large Depth}, 
      author={Thomas Bachlechner and Bodhisattwa Prasad Majumder and Huanru Henry Mao and Garrison W. Cottrell and Julian McAuley},
      year={2020},
      eprint={2003.04887},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
