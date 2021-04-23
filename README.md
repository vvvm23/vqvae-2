# Generating Diverse High-Fidelity Images with VQ-VAE-2 [Work in Progress]
PyTorch implementation of Hierarchical, Vector Quantized, Variational Autoencoders (VQ-VAE-2) 
from the paper "Generating Diverse High-Fidelity Images with VQ-VAE-2"

Original paper can be found [here](https://arxiv.org/abs/1906.00446)

Vector Quantizing layer based off implementation by [@rosinality](https://github.com/rosinality) 
found [here](https://github.com/rosinality/vq-vae-2-pytorch).

Aiming for a focus on supporting an arbitrary number of VQ-VAE "levels". Most implementations in 
PyTorch typically only use 2 which is limiting at higher resolutions.

This project will not only contain the VQ-VAE-2 architecture, but also an example autoregressive 
prior and latent dataset extraction.

## Usage
`python main-vqvae.py --task task_name` will run VQ-VAE-2 training using the config `task_name` found in `hps.py`. Defaults to `cifar10`.

## Modifications
- Replacing residual layers with ReZero layers.

## Samples
`TODO: Add some samples`

## Checkpoints
`TODO: Add model checkpoints`

### Roadmap
- [ ] Prettier outputs
- [ ] Server mode (no fancy printing)
- [ ] Experiment directories (containing logs / checkpoints / etc)
- [ ] Accumulated gradient training (for larger batch sizes on limited resources)
- [ ] Learning rate schedulers
- [ ] Samples and checkpoints on FFHQ1024
- [ ] Autoregressive prior models / training scripts
- [ ] Full system sampling

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
