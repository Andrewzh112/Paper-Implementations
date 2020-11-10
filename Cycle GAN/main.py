from cycle_gan.train import Trainer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-train', action="store_true", default=False, help='Train cycleGAN models')
parser.add_argument('-test',  action="store_true", default=False, help='Test cycleGAN')
parser.add_argument('--dim_A', type=int, default=3, help='Numer of channels for class A')
parser.add_argument('--dim_B', type=int, default=3, help='Numer of channels for class B')
parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of ResNet Blocks for generators')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimensions for model')
parser.add_argument('--starting_epoch', type=int, default=0, help='Starting epoch for resuming training')
parser.add_argument('--lr_G', type=float, default=0.0004, help='Learning rate for generators')
parser.add_argument('--lr_D', type=float, default=0.0002, help='Learning rate for discriminators')
parser.add_argument('--betas', type=tuple, default=(0.5, 0.999), help='Betas for Adam optimizer')
parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--decay_epoch', type=int, default=100, help='Starting epoch when learning rate will start decay')
parser.add_argument('--target_shape', type=int, default=224, help='Final image H or W')
parser.add_argument('--progress_interval', type=int, default=1, help='Save model and generated image every x epoch')
parser.add_argument('--sample_batches', type=int, default=25, help='How many generated images to sample')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--save_img_dir', type=str, default='save_images', help='Path to where generated images will be saved')
parser.add_argument('--data_root', type=str, default='horse2zebra', help='Path to where image data is located')
parser.add_argument('--checkpoint_dir', type=str, default='model_weights', help='Path to where model weights will be saved')
opt = parser.parse_args()

if __name__ == '__main__':
    if opt.train:
        trainer = Trainer(opt)
        trainer.train()
    # TODO
    if opt.test:
        pass
