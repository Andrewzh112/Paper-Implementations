import torch
import numpy as np
from tqdm import tqdm
import os
import itertools
import random
from .utils import LambdaLR, save_images, ReplayBuffer
from .engine import CycleImageDataset
from .model import Discriminator, Generator
from .loss import get_disc_loss, get_gen_loss


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.G_AB = Generator(
            args.dim_A,
            args.dim_B,
            args.hidden_dim,
            args.n_res_blocks
        ).to(self.device)
        self.G_BA = Generator(
            args.dim_B,
            args.dim_A,
            args.hidden_dim,
            args.n_res_blocks
        ).to(self.device)
        self.D_A = Discriminator(args.dim_A, args.hidden_dim).to(self.device)
        self.D_B = Discriminator(args.dim_B, args.hidden_dim).to(self.device)

        self.dataset = CycleImageDataset(args.data_root)

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(),
            self.G_BA.parameters()),
            lr=args.lr_G,
            betas=args.betas
        )
        self.optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(),
            lr=args.lr_D,
            betas=args.betas
        )
        self.optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(),
            lr=args.lr_D,
            betas=args.betas
        )

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A,
            lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B,
            lr_lambda=LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step
        )

        if args.starting_epoch > 0:
                self.load_state_dicts()

        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.fake_As = ReplayBuffer()
        self.fake_Bs = ReplayBuffer()

    def load_state_dicts(self):
        assert os.path.isfile(
            f'{self.args.checkpoint_dir}/cycleGAN_{self.args.starting_epoch}.pth'
        ), f'Please make sure cycleGAN_{self.args.starting_epoch}.pth is in the \
            {self.args.checkpoint_dir} directory'
        model_state_dicts = torch.load(
            f'{self.args.checkpoint_dir}/cycleGAN_{self.args.starting_epoch}.pth')
        self.G_AB.load_state_dict(model_state_dicts['G_AB'])
        self.G_BA.load_state_dict(model_state_dicts['G_BA'])
        self.optimizer_G.load_state_dict(model_state_dicts['optimizer_G'])
        self.D_A.load_state_dict(model_state_dicts['D_A'])
        self.optimizer_D_A.load_state_dict(model_state_dicts['optimizer_D_A'])
        self.D_B.load_state_dict(model_state_dicts['D_B'])
        self.optimizer_D_B.load_state_dict(model_state_dicts['optimizer_D_B'])

    def save_progress(self, epoch, real_A, real_B, fake_A, fake_B):
        if not os.path.isdir(self.args.save_img_dir):
            os.makedirs(self.args.save_img_dir)
        save_images(
            torch.cat([real_A, real_B]),
            f'{self.args.save_img_dir}/Real_{epoch}.png',
            size=(self.args.dim_A, self.args.target_shape, self.args.target_shape)
        )
        save_images(
            torch.cat([fake_B, fake_A]),
            f'{self.args.save_img_dir}/Fake_{epoch}.png',
            size=(self.args.dim_B, self.args.target_shape, self.args.target_shape)
        )
        if not os.path.isdir(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        torch.save({
            'G_AB': self.G_AB.state_dict(),
            'G_BA': self.G_BA.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'D_A': self.D_A.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict()
        }, f"{self.args.checkpoint_dir}/cycleGAN_{epoch}.pth")

    def train(self):
        device, args = self.device, self.args
        progress_interval = args.progress_interval
        pbar = tqdm(
            range(args.starting_epoch, args.n_epochs),
            total=(args.n_epochs - args.starting_epoch)
        )

        def run_epoch(loader):
            disc_A_losses, disc_B_losses, gen_AB_losses = [], [], []
            real_As, real_Bs, fake_As, fake_Bs = [], [], [], []
            sampled_idx = np.random.choice(
              list(range(len(loader))),
              size=args.sample_batches,
              replace=False)
            for batch_idx, (real_A, real_B) in enumerate(loader):
                real_A = torch.nn.functional.interpolate(real_A, size=args.target_shape).to(device)
                real_B = torch.nn.functional.interpolate(real_B, size=args.target_shape).to(device)

                ### Update discriminator A ###
                self.optimizer_D_A.zero_grad()
                with torch.no_grad():
                    fake_A = self.G_BA(real_B)
                    fake_A = self.fake_As.sample(fake_A)
                disc_A_loss = get_disc_loss(real_A, fake_A, self.D_A, self.criterion_GAN)
                disc_A_loss.backward(retain_graph=True)
                self.optimizer_D_A.step()
                disc_A_losses.append(disc_A_loss.item())

                ### Update discriminator B ###
                self.optimizer_D_B.zero_grad()
                with torch.no_grad():
                    fake_B = self.G_AB(real_A)
                    fake_B = self.fake_Bs.sample(fake_B)
                disc_B_loss = get_disc_loss(real_B, fake_B, self.D_B, self.criterion_GAN)
                disc_B_loss.backward(retain_graph=True)
                self.optimizer_D_B.step()
                disc_B_losses.append(disc_B_loss.item())

                ### Update Generators ###
                self.optimizer_G.zero_grad()
                gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A,
                    real_B,
                    self.G_AB,
                    self.G_BA,
                    self.D_A,
                    self.D_B,
                    self.criterion_GAN,
                    self.criterion_identity,
                    self.criterion_cycle
                )
                gen_loss.backward()
                self.optimizer_G.step()
                gen_AB_losses.append(gen_loss.item())

                if batch_idx in sampled_idx:
                  real_As.append(real_A)
                  real_Bs.append(real_B)
                  fake_As.append(fake_A)
                  fake_Bs.append(fake_B)
            images = [torch.cat(real_As), torch.cat(real_Bs), torch.cat(fake_As), torch.cat(fake_Bs)]
            return images, np.mean(disc_A_losses), np.mean(disc_B_losses), np.mean(gen_AB_losses)

        disc_A_losses, disc_B_losses, gen_AB_losses = [], [], []
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        for epoch in pbar:
            self.G_AB.train()
            self.G_BA.train()
            self.D_A.train()
            self.D_B.train()
            images, disc_A_loss, disc_B_loss, gen_AB_loss = run_epoch(loader=dataloader)
            disc_A_losses.append(disc_A_loss)
            disc_B_losses.append(disc_B_loss)
            gen_AB_losses.append(gen_AB_loss)
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()
            self.lr_scheduler_G.step()

            if (epoch + 1) % progress_interval == 0:
                self.save_progress(epoch, *images)

            tqdm.write(
                f'Epoch {epoch + 1}/{args.n_epochs}, \
                    Disc A loss: {np.mean(disc_A_losses):.3f}, \
                    Disc B loss: {np.mean(disc_B_losses):.3f}, \
                    Gen Loss: {np.mean(gen_AB_losses):.3f}'
            )
