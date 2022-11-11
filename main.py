"""
This script is mainly built upon:
Conditional Diffusion MNIST, https://github.com/TeaPearce/Conditional_Diffusion_MNIST
GLIDE, https://github.com/openai/glide-text2im
glide-finetune, https://github.com/afiaka87/glide-finetune
other reference:
guided-diffusion: https://github.com/openai/guided-diffusion
improved-diffusion: https://github.com/openai/improved-diffusion
"""
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from loader import TextImageDataset
from create_model import create_nnmodel

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T,c))

    def sample(self, n_sample, size, device, test_param, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with parameter (tokens) and the other with empty (0) tokens.
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.tensor(np.float32(test_param))[None,:].to(device) 
        c_i = c_i.repeat(int(n_sample),1)
        
        uncond_tokens = torch.tensor(np.float32(np.array([0,0]))).to(device) 
        uncond_tokens=uncond_tokens.repeat(int(n_sample),1)
        
        c_i = torch.cat((c_i, uncond_tokens), 0)

        x_i_store = [] # keep track of generated steps in case want to plot something 
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, t_is,c_i)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def train_eor():

    ## hardcoding these here
    n_epoch = 160 # 120
    batch_size =16 # 16
    n_T = 500 # 500; DDPM time steps
    device = "cuda" # using gpu
    #device = "cpu"
    n_param = 2 # dimension of parameters
    lrate = 1e-4
    save_model = True
    save_dir = './outputs/'
    ws_test = [2.0] #[0,0.5,2] strength of generative guidance
    test_param = np.array([0.2 , 0.80000023]) # context for us for condional testing
    n_sample = 64 # 64, the number of samples in sampling process
    drop_prob = 0.28 # the probability to drop the captions (parameters) for unconditional training in classifier free guidance.
    image_size=64
    data_dir = '/scratch/zxs/scripts/Diffuse/glide-finetune/data'
    save_freq = 40 # the period of saving model
    sample_freq = 10 # the period of sampling
    ema=True # whether to use ema
    ema_rate=0.999

    nn_model=create_nnmodel(n_param=n_param,image_size=image_size)
    ddpm = DDPM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=drop_prob)
    ddpm.train()
    number_of_params = sum(x.numel() for x in ddpm.parameters())
    print(f"Number of parameters: {number_of_params}")
    number_of_trainable_params = sum(x.numel() for x in ddpm.parameters() if x.requires_grad)
    print(f"Trainable parameters: {number_of_trainable_params}")
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./outputs/model_0_test.pth"))

    # load the local data, image file names: *_param2_param1*.npy, values rescaled to [-1,1], check the loader.py file.
    dataset = TextImageDataset(
            folder=data_dir,
            image_size=image_size,
            uncond_p=drop_prob, # only used when drop_para=True
            shuffle=True,
            n_param=n_param,
            drop_para=True
        )

    # data loader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == device),
    )
    
    # initialize optimizer
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    
    if ema:
        ema = EMA(ddpm, ema_rate)
        ema.register()
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()

            pbar.set_description(f"loss: {loss.item():.4f}")
            optim.step()
            
            if ema:
                ema.update()
        
        if ema:
            ema.apply_shadow()
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            if ep%sample_freq==0:
                for w_i, w in enumerate(ws_test):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (1, image_size, image_size), device, test_param=test_param, guide_w=w)

                    sample_save_path_final = os.path.join(save_dir, f"train-{ep}xscale_{w}_test_ema49.npy")
                    np.save(str(sample_save_path_final),x_gen.cpu())

        # optionally save model
        if save_model and ep%save_freq==0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}_test_ema49.pth")
            print('saved model at ' + save_dir + f"model_{ep}_test_ema49.pth")
        if ema:
            ema.restore()

if __name__ == "__main__":
    train_eor()
