import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim): #MNIST
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            # LeakyReLU for GANs
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256)
            nn.LeakyReLU(0.1)
            nn.Linear(256, img_dim) # 28 x 28 x 1 = 256
            nn.Tanh() #Tanh so output of pixel values are between -1 and 1
        )
    def forward(self, x):
        return self.gen(x)

# hyperparameters
lr = 3e-4
z_dim = 64 #128, 256, etc
img_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

disc = Discriminator(img_dim)
gen = Generator(z_dim, img_dim)

# see how generated images change over epochs
fixed_noise = torch.randn((batch_size, z_dim))
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
disc_opt = optim.Adam(disc.parameters(), lr=lr)
gen_opt = optim.Adam(gen.parameters(), lr=lr)
loss = nn.BCELoss()

#tensorboard - only outputs fake images (what generator generates)
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real_image, _) in enumerate(loader):
        real = real.view(-1, 784)
        batch_size = real.shape[0]

        # train discriminator: maximize log(D(real)) + log(1-D(G(z)))
        noise = torch.randn((batch_size, z_dim))
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = loss(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = loss(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        disc_opt.step()

        # train generator, want to minimize log(1-D(G(Z))) <-> max log(D(G(Z)))
        output = disc(fake).view(-1)
        lossG = loss(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        gen_opt.step()

