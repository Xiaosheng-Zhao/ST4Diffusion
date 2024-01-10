The repository for Can Diffusion Model Conditionally Generate Astrophysical Images? ([Zhao et al. 2023](https://arxiv.org/abs/2307.09568)). This script is mainly built upon:[Conditional Diffusion MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST), [guided-diffusion](https://github.com/openai/guided-diffusion) and [improved-diffusion](https://github.com/openai/improved-diffusion).

The requirements to run the codes are rather simple, please check the `requirements.txt`, where I list the versions of the packages I used (with py3.8), while probably your existed PyTorch environment can work. In this repository (mainly in a single file `main.py`), you can know all about the basics of the paper [DDPM](https://arxiv.org/abs/2006.11239). 

You can directly run the `main.py` which will give some outputs to `outputs`.
<p align = "center">
<img width="750" src="ST4Diffusion.png"/img>
</p>
<p align = "center">
  DDPM for image generation conditional on the astrophysical parameter
</p>

The wavelet scattering transform used for quantification is calculated with [This Repository](https://github.com/SihaoCheng/scattering_transform) by Sihao Cheng.

The StyleGAN2 implementation is based on the related branch in [this repository](https://github.com/dkn16/stylegan2-pytorch).
