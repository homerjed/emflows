import os 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid

from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from einops import rearrange

from azula.denoise import PreconditionedDenoiser
from azula.nn.embedding import SineEncoding
from azula.nn.unet import UNet
from azula.noise import RectifiedSchedule
from azula.sample import DDIMSampler
from azula.guidance import MMPSDenoiser

import jaxtyping
from beartype import beartype as typechecker

plt.set_cmap("gray")


def get_dirs(dataset_name):
    exp_path = os.path.join(os.getcwd(), "exps/", dataset_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    for _dir in ["figs/samples/", "figs/posterior_samples/", "out/"]:
        os.makedirs(os.path.join(exp_path, _dir), exist_ok=True)
    return exp_path
    

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[torch.Tensor, shapes]


Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

typecheck = jaxtyping.jaxtyped(typechecker=typechecker)

device = "cuda"

dataset_name = "mnist"

exp_dir = get_dirs(dataset_name)

n_pix = 28
n_channels = 1
image_shape = (n_channels, n_pix, n_pix)
n_x_steps = 50
image_key = "image"

n_em_steps = 2000
sigma_y = 1.
covariance_y = torch.eye(n_pix ** 2) * sigma_y # var?

n_ddim_steps = 64

n_latents_sample = 64
n_plot = int(np.sqrt(n_latents_sample))

N_DATA = 60_000 # E.g. limit dataset for testing

batch_size = 512
lr = 3e-4

loader_kwargs = dict(
    shuffle=True,
    drop_last=False, # Think about this
    pin_memory=True,
    batch_size=batch_size,
    num_workers=2
)


class UNetWrapper(nn.Module):
    def __init__(self, channels: int = 1, emb_features: int = 256):
        super().__init__()

        self.unet = UNet(
            in_channels=channels,
            out_channels=channels,
            hid_channels=[16, 32, 64],
            hid_blocks=[2, 2, 2],
            attention_heads={2: 1},
            mod_features=emb_features,
        )

        self.time_encoding = nn.Sequential(
            SineEncoding(emb_features, omega=1000),
            nn.Linear(emb_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
        )

    @typecheck
    def forward(
        self, 
        x_t: Float["n 1 28 28"] | Float["n 784"], 
        log_snr_t: Float["n"] | Float[""]
    ) -> Float["n 1 28 28"] | Float["n 784"]:

        # If tensor arrives flat, send it out flat
        if x_t.shape[1:] != image_shape:
            flatten = True
            x_t = x_t.reshape(x_t.shape[0], *image_shape)
        else:
            flatten = False

        emb = self.time_encoding(log_snr_t) 
        x_0 = self.unet(x_t, emb) # e.g. mod=emb, cond=None

        if flatten:
            out = x_0.flatten(start_dim=1)
        else:
            out = x_0

        return out


@typecheck
def probabilistic_pca(X: Float["n 784"], d: int) -> tuple[Float["784"], Float["784 784"]]:
    n, p = X.shape

    mu_x = X.mean(dim=0)
    X_centered = X - mu_x

    # Empirical covariance matrix
    cov_empirical = X_centered.T @ X_centered / (n - 1)

    # Eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_empirical)  # Sorted in ascending order
    eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)

    # Top-d components
    Lambda_d = torch.diag(eigvals[:d])
    V_d = eigvecs[:, :d]

    # Estimate noise variance (average of discarded eigenvalues)
    sigma2 = eigvals[d:].mean()

    # Probabilistic PCA covariance estimate
    cov_x = V_d @ (Lambda_d - sigma2 * torch.eye(d, device=X.device)) @ V_d.T + sigma2 * torch.eye(p, device=X.device)

    return mu_x.to(device), cov_x.to(device)


def A(x):
    return x


def A_inv(y):
    return y


def transform(rows):
    rows[image_key] = list(map(to_tensor, rows[image_key]))
    return rows


def preprocess(x):
    return 2. * x - 1.


def postprocess(x):
    x = torch.clip((x + 1.) / 2., min=0., max=1.)
    return x


@typecheck
def get_x_y_sampler(denoiser: nn.Module, y: Float["n 784"]) -> DDIMSampler:
    cond_denoiser = MMPSDenoiser(
        denoiser, y=y, A=A, var_y=sigma_y ** 2., iterations=3
    )
    cond_sampler = DDIMSampler(
        cond_denoiser, steps=n_ddim_steps, eta=1.
    )
    return cond_sampler.to(device)


class GaussianDenoiser(nn.Module):

    @typecheck
    def __init__(self, mu_x: Float["784"], cov_x: Float["784 784"]):
        super().__init__()
        self.mu_x = mu_x
        self.cov_x = cov_x

    # def __call__(self, xt: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:

    #     cov_t = sigma_t[..., None] ** 2.

    #     I = torch.eye(xt.shape[-1], device=xt.device).unsqueeze(0)  
    #     _cov = self.cov_x.unsqueeze(0) + cov_t.view(xt.shape[0], 1, 1) * I  

    #     return xt - cov_t * torch.linalg.solve(_cov, (xt - self.mu_x).unsqueeze(-1)).squeeze(-1)

    @typecheck
    def forward(self, xt: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        B, D = xt.shape
        cov_t = sigma_t.view(-1, 1) ** 2  # shape [B, 1] or [1, 1]

        # Broadcast cov_x and eye(D) into batch shape
        I = torch.eye(D, device=xt.device).unsqueeze(0)  # [1, D, D]
        A = self.cov_x.unsqueeze(0) + cov_t.view(-1, 1, 1) * I  # [B, D, D] or [1, D, D]

        b = xt - self.mu_x  # [B, D]

        x_sol = torch.linalg.solve(A.expand(B, D, D), b.unsqueeze(-1)).squeeze(-1)  # [B, D]

        return xt - cov_t * x_sol


# Load dataset
dataset = load_dataset(
    dataset_name, 
    split="train", 
    keep_in_memory=True, 
    # data_dir="data/"
)

# Extract latents 
x_dataset = torch.stack(dataset.with_transform(transform)[image_key]) # NOTE: transform before or after noise?

# Add noise to latents
X = x_dataset.flatten(start_dim=1)[:N_DATA]
Y = A(X) + torch.rand_like(X) * sigma_y

noisy_dataset_loader = DataLoader(TensorDataset(Y), **loader_kwargs)

# Denoising UNet for flow matching
denoiser = PreconditionedDenoiser(
    backbone=UNetWrapper(channels=n_channels), 
    schedule=RectifiedSchedule()
).to(device)

optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)

model_name = "denoiser_{}_fm_uncond.pt".format(dataset_name)


def get_X_Y(Y: Float["n 784"], denoiser: nn.Module, k: int = 0) -> DataLoader:
    """
        Loop through dataset of noisy data Y, denoising the batch
        with MMPS sampling X|Y, in order to create a dataset of X|Y.
    """

    # if k == 0:
    #     pass
    #     # mu_x, cov_x = probabilistic_pca(X, d=np.prod(image_shape) - 1)

    #     # Cheat
    #     mu_x = torch.mean(X.flatten(start_dim=1), dim=0)
    #     cov_x = torch.cov((X - mu_x).T) 

    #     print(mu_x.shape, cov_x.shape)

    #     plt.figure()
    #     plt.imshow(cov_x.cpu().numpy())
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.colorbar()
    #     plt.savefig("figs/em/cov_x.png")
    #     plt.close()

    #     is_singular = torch.linalg.matrix_rank(cov_x) < cov_x.shape[-1]
    #     print("cov_x singular:", is_singular)

    #     denoiser = PreconditionedDenoiser(
    #         backbone=GaussianDenoiser(mu_x.to(device), cov_x.to(device)),
    #         schedule=RectifiedSchedule()
    #     ).to(device)

    X_Y = torch.zeros_like(Y)

    bar = tqdm(range(len(Y) // batch_size + 1), desc="Posterior sampling", colour="green")

    denoiser.eval()

    with torch.no_grad():
        for i, batch in zip(bar, noisy_dataset_loader):
            y = batch[0].to(device=device)

            cond_sampler = get_x_y_sampler(denoiser, y)

            x1 = cond_sampler.init(y.shape)
            x_y = cond_sampler(x1.flatten(start_dim=1))

            X_Y[i * batch_size : (i + 1) * batch_size] = x_y

            del x1, x_y, cond_sampler
            torch.cuda.empty_cache()

        x_y_dataset_dataloader = DataLoader(TensorDataset(X_Y), **loader_kwargs)

        # Monitor posterior samples with fixed data conditioning
        cond_sampler = get_x_y_sampler(denoiser, Y[:n_latents_sample])
        x1 = cond_sampler.init((n_latents_sample,) + image_shape)
        x_y = cond_sampler(x1.flatten(start_dim=1))

    image = postprocess(x_y).squeeze().cpu().numpy()
    plt.figure()
    plt.title(r"$x\sim p_{\theta}(x|y)$")
    plt.imshow(rearrange(image, "(r c) (h w) -> (r h) (c w)", r=n_plot, c=n_plot, h=n_pix, w=n_pix))
    plt.xticks([])
    plt.yticks([])
    plt.savefig("{}/figs/posterior_samples/image_em_k={:05d}.png".format(exp_dir, k), bbox_inches="tight")
    plt.close()

    return x_y_dataset_dataloader


plt.figure()
plt.title(r"$x\sim p(x)$")
plt.imshow(rearrange(X[:n_latents_sample], "(p q) (c h w) -> (p h) (q w) c", p=n_plot, q=n_plot, c=n_channels, h=n_pix, w=n_pix))
plt.xticks([])
plt.yticks([])
plt.savefig("{}/figs/latents.png".format(exp_dir), bbox_inches="tight")
plt.close()

plt.figure()
plt.title(r"$y\sim p(y|x)$")
plt.imshow(rearrange(Y[:n_latents_sample], "(p q) (c h w) -> (p h) (q w) c", p=n_plot, q=n_plot, c=n_channels, h=n_pix, w=n_pix))
plt.xticks([])
plt.yticks([])
plt.savefig("{}/figs/data.png".format(exp_dir), bbox_inches="tight")
plt.close()

# Expectation-maximisation iterations
losses_k = []
for k in (
    bar := tqdm(range(n_em_steps), desc="Expectation max.", colour="red")
):

    x_y_dataset_dataloader = get_X_Y(Y, denoiser=denoiser, k=k)

    denoiser.train()
    
    losses_e = []
    for _ in (
        bar := tqdm(range(n_x_steps), desc="Training (latents)", colour="blue")
    ):
        losses = []

        # Implement validation loop
        for batch in x_y_dataset_dataloader:
            x = batch[0].to(device=device)

            # x = preprocess(x) # Necessary? Done in dataset initially?
            t = torch.rand(len(x), device=device) # Implement logit-normal time sampling

            loss = denoiser.loss(x, t)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())

            optimizer.step()
            optimizer.zero_grad()

        losses_e.append(np.mean(losses))

        bar.set_postfix(loss=np.mean(losses))

    losses_k.append(np.mean(losses_e))

    torch.save(denoiser, model_name)

    # Sample x ~ p(x) 
    sampler = DDIMSampler(denoiser, steps=n_ddim_steps).to(device)

    x1 = sampler.init((n_latents_sample,) + image_shape)
    x0 = sampler(x1)

    image = make_grid(postprocess(x0).squeeze()).cpu().numpy()
    plt.figure()
    plt.title(r"$x\sim p_{\theta}(x)$")
    plt.imshow(rearrange(image, "(p q) c h w -> (p h) (q w)", p=n_plot, q=n_plot, c=n_channels))
    plt.xticks([])
    plt.yticks([])
    plt.savefig("{}figs/samples/image_em_k={:05d}.png".format(exp_dir, k), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(losses_k)
    plt.xlabel(r"$k$")
    plt.ylabel(r"FM loss (EM)")
    plt.savefig("{}/figs/losses.png".format(exp_dir), bbox_inches="tight")
    plt.close()

    del x0, x1, sampler
    torch.cuda.empty_cache()