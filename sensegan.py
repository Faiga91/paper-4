"""
SenseGAN module.
"""
import torch
from torch import nn


def get_generator_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation
          followed by a batch normalization and then a relu activation
    '''
    if not final_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    else: # Final Layer
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            nn.Tanh()
        )
    

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        in_dim: the dimension of the input
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, in_dim=1, hidden_dim=128):
        super().__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),

            nn.Linear(hidden_dim * 8, in_dim),
            nn.Sigmoid()

        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated time-series.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen

def get_noise(n_samples, z_dim, device='cuda'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device
    # argument to the function you use to generate the noise.
    # pylint: disable=E1101
    return  torch.randn(n_samples, z_dim, device = device)

def get_discriminator_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation
          followed by an nn.LeakyReLU activation with negative slope of 0.2
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    if not final_layer:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    else: # Final Layer
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride)
        )

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        in_dim: the dimension of the generated variable, fitted for the dataset used, a scalar
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, in_dim=1, hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(in_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input):
        '''
        Function for completing a forward pass of the discriminator: Given an input tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            input: a flattened input tensor with dimension (in_dim)
        '''
        return self.disc(input)

    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc

# pylint: disable=R0913
def get_disc_loss(gen, disc, criterion, real, num_inputs, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an input given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the inputs
               (e.g. fake = 0, real = 1)
        real: a batch of real inputs
        num_inputs: the number of inputs the generator should produce,
                which is also the length of the real inputs
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''

    fake = gen(get_noise(num_inputs, z_dim, device=device)).detach()
    pred_f = disc(fake)
    pred_r = disc(real)
    # pylint: disable=E1101
    ground_truth_f = torch.torch.zeros_like(pred_f)
    ground_truth_r = torch.torch.ones_like(pred_r)

    loss_f = criterion(pred_f, ground_truth_f)
    loss_r = criterion(pred_r, ground_truth_r)

    disc_loss = (loss_f + loss_r) / 2

    return disc_loss


def get_gen_loss(gen, disc, criterion, num_inputs, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an input given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the inputs
               (e.g. fake = 0, real = 1)
        num_inputs: the number of inputs the generator should produce,
                which is also the length of the real inputs
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    noise = get_noise(num_inputs, z_dim, device=device)
    fake = gen(noise)
    pred_f = disc(fake)
    # pylint: disable=E1101
    ground_truth_ = torch.torch.ones_like(pred_f)
    gen_loss = criterion(pred_f, ground_truth_)
    return gen_loss
