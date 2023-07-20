The repository for Can Diffusion Model Conditionally Generate Astrophysical Images? ([Zhao et al. 2023](https://arxiv.org/abs/2307.09568)). This script is mainly built upon:[Conditional Diffusion MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST), [guided-diffusion](https://github.com/openai/guided-diffusion) and [improved-diffusion](https://github.com/openai/improved-diffusion).

The requirements to run the codes are rather simple, please check the `requirements.txt`, where I list the versions of the packages I used, while probably your existed PyTorch environment can work. In this repository (mainly in a single file `main.py`), you can know all about the basics of the paper [DDPM](https://arxiv.org/abs/2006.11239). 

You can directly run the `main.py` which will give some outputs to `outputs`.

The StyleGAN2 implementation is based on the related branch in [this repository](https://github.com/dkn16/stylegan2-pytorch).
