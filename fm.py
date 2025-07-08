import os
import torch
import torch.nn as nn

from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from azula.denoise import PreconditionedDenoiser
from azula.nn.embedding import SineEncoding
from azula.nn.unet import UNet
from azula.noise import VPSchedule, RectifiedSchedule
from azula.sample import DDIMSampler
from azula.guidance import MMPSDenoiser

plt.set_cmap("gray")

def get_dirs(dataset_name):
    exp_path = os.path.join(os.getcwd(), "exps/", dataset_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    for _dir in ["figs/samples/", "out/"]:
        os.makedirs(os.path.join(exp_path, _dir), exist_ok=True)
    return exp_path

device = "cuda"

dataset_name = "mnist"

exp_dir = get_dirs(dataset_name)

if dataset_name == "mnist":
    n_pix = 28
    n_channels = 1
    n_steps = 10
    img_key = "image" 
if dataset_name == "cifar10":
    n_pix = 32
    n_channels = 3
    n_steps = 2000
    img_key = "img"


def transform(rows):
    rows[img_key] = list(map(to_tensor, rows[img_key]))
    rows["label"] = list(map(torch.as_tensor, rows["label"]))
    return rows


dataset = load_dataset(dataset_name, split="train", keep_in_memory=True)
dataset = dataset.with_transform(transform)


def preprocess(x):
    return 2 * x - 1


def postprocess(x):
    return torch.clip((x + 1) / 2, min=0, max=1)


class UNetWrapper(nn.Module):
    def __init__(self, channels: int = 1, emb_features: int = 256):
        super().__init__()

        self.unet = UNet(
            in_channels=channels,
            out_channels=channels,
            hid_channels=[16, 32, 64] if dataset_name == "mnist" else [32, 64, 128],
            hid_blocks=[2, 2, 2],
            attention_heads={2: 1},
            mod_features=emb_features,
        )

        self.time_encoding = nn.Sequential(
            SineEncoding(emb_features, omega=1e3),
            nn.Linear(emb_features, emb_features),
            nn.SiLU(),
            nn.Linear(emb_features, emb_features),
        )

    def forward(self, x_t, log_snr_t):
        emb = self.time_encoding(log_snr_t) 
        x_0 = self.unet(x_t, emb)
        return x_0


denoiser = PreconditionedDenoiser(
    backbone=UNetWrapper(channels=n_channels), schedule=RectifiedSchedule()
).to(device)

optimizer = torch.optim.Adam(denoiser.parameters(), lr=3e-4)

model_name = "denoiser_{}_fm_uncond.pt".format(dataset_name)

try:
    denoiser = torch.load(model_name, weights_only=False)
except Exception as e:
    print(e)

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=2,
    )

    for _ in (bar := tqdm(range(n_steps), desc="Training")):
        losses = []

        for batch in loader:
            x = batch[img_key].to(device=device)
            label = batch["label"].to(device=device)

            x = preprocess(x)
            t = torch.rand(len(x), device=device)

            loss = denoiser.loss(x, t)
            loss.backward()
            losses.append(loss.detach())

            optimizer.step()
            optimizer.zero_grad()

        bar.set_postfix(loss=torch.stack(losses).mean().item())

    torch.save(denoiser, model_name)

sampler = DDIMSampler(denoiser, steps=64).to(device)

x1 = sampler.init((1, n_channels, n_pix, n_pix))
x0 = sampler(x1)

image = postprocess(x0).squeeze().reshape((n_pix, n_pix, n_channels)).cpu().numpy()
plt.figure()
plt.imshow(image)
plt.savefig("{}figs/image.png".format(exp_dir))
plt.close()

x = preprocess(to_tensor(image)).to(device)

n_pix_downsample = n_pix // 2
sigma_y = 1.0


def A(x):
    return torch.nn.functional.interpolate(
        x, (n_pix_downsample, n_pix_downsample), mode="bicubic", antialias=True
    ).flatten(-3)


def A_inv(y):
    return torch.nn.functional.interpolate(
        y.unflatten(-1, (n_channels, n_pix_downsample, n_pix_downsample)), (n_pix, n_pix), mode="nearest"
    )


y = A(x.unsqueeze(0))
y = y + torch.randn_like(y) * sigma_y

image = make_grid(postprocess(A_inv(y))).cpu().numpy().transpose(1, 2, 0)
plt.figure()
plt.imshow(image)
plt.savefig("{}figs/measurement.png".format(exp_dir))
plt.close()

cond_denoiser = MMPSDenoiser(denoiser, y=y, A=A, var_y=sigma_y ** 2, iterations=3)
cond_sampler = DDIMSampler(cond_denoiser, steps=64, eta=1.0).to(device)

x1 = cond_sampler.init((64, n_channels, n_pix, n_pix))
x0 = cond_sampler(x1)

image = make_grid(postprocess(x0)).cpu().numpy().transpose(1, 2, 0)
plt.figure()
plt.imshow(image)
plt.savefig("{}figs/reconstruction.png".format(exp_dir))
plt.close()