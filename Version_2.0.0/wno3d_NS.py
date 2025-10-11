#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet neural operator: a neural
   operator for parametric partial differential equations. arXiv preprint arXiv:2205.02191.

This code is for 2-D Navier-Stokes equation (2D time-dependent problem) using Time as third axis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv3d
from lightning_utils import BasicLightningRegressor, ToggleableGroupNorm

torch.manual_seed(0)
np.random.seed(0)

# %%
""" The forward operation """
class WNO3d(BasicLightningRegressor):
    def __init__(self, in_channels, out_channels, size, padding=0,
                 hidden_channels=32, n_layers=4, level=2, wavelet='db6',
                 hidden_norm_groups=0, out_norm_groups=0, ndims=3,
                 activation=nn.Mish, grid_range=[1.0, 1.0, 1.0]):
        super(WNO3d, self).__init__()

        """
        WNO3d adapted to match MOR_Operator interface.

        Input : (batch, channels, x, y, z) - automatically adds coordinate grid
        Output: (batch, out_channels, x, y, z)

        Parameters matching MOR_Operator:
        ---------------------------------
        in_channels: input channels (before adding coordinate grid)
        out_channels: output channels (configurable, was fixed to 1)
        hidden_channels: lifting dimension (was 'width')
        n_layers: number of wavelet layers (was 'layers')
        hidden_norm_groups: normalization groups for hidden layers (default 1 = LayerNorm)
        out_norm_groups: normalization groups for output (default 1 = LayerNorm)
        ndims: spatial dimensions (must be 3 for WNO3d)
        activation: activation function (default SiLU)

        WNO-specific parameters:
        ------------------------
        level: wavelet decomposition level (default 2)
        wavelet: wavelet filter (default 'db6')
        size: 3D volume size [x, y, z] (default [64, 64, 64])
        grid_range: coordinate grid range (default [1,1,1])
        padding: zero padding size - scalar (applied to all dims) or [pad_x, pad_y, pad_z] (default 0)
        """

        # Check ndims constraint
        if ndims != 3:
            raise NotImplementedError("WNO3d only supports ndims=3 (3D spatial dimensions)")

        # Map interface parameters
        self.n_layers = n_layers
        self.activation = activation()
        self.grid_range = grid_range

        # Handle flexible padding: scalar -> [pad, pad, pad], list/tuple -> [pad_x, pad_y, pad_z]
        if type(padding) is int: self.padding = [padding] * 3
        else: assert len(padding) == 3, "padding must be a scalar or a list/tuple of length 3"

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        # Input projection: in_channels + ndims (for coordinate grid)
        self.fc0 = nn.Linear(in_channels + ndims, hidden_channels)

        # Initialize wavelet layers
        for i in range(n_layers):
            self.conv.append(WaveConv3d(hidden_channels, hidden_channels,
                                        level, size, wavelet))
            self.w.append(nn.Conv3d(hidden_channels, hidden_channels, 1))

        # Use hidden_channels for projection layers (not hardcoded 128)
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

        # Add normalization layers
        self.hidden_norms = nn.ModuleList([ToggleableGroupNorm(hidden_norm_groups, hidden_channels) for _ in range(n_layers-1)])
        self.output_norm = ToggleableGroupNorm(out_norm_groups, out_channels)

    def forward(self, x):
        # Input: (batch, channels, x, y, z) -> convert to (batch, x, y, z, channels)
        x = x.permute(0, 2, 3, 4, 1)  # (batch, x, y, z, channels)

        # Add coordinate grid automatically
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # (batch, x, y, z, channels + ndims)

        # Input projection
        x = self.fc0(x)  # (batch, x, y, z, hidden_channels)

        # Convert to conv format: (batch, channels, x, y, z)
        x = x.permute(0, 4, 1, 2, 3)

        # Apply padding if required
        if any(p > 0 for p in self.padding):
            # F.pad format: [pad_x_left, pad_x_right, pad_y_top, pad_y_bottom, pad_z_front, pad_z_back]
            # padding=[pad_x, pad_y, pad_z] -> [0, pad_x, 0, pad_y, 0, pad_z]
            pad_list = [0, self.padding[0], 0, self.padding[1], 0, self.padding[2]]
            x = F.pad(x, pad_list)

        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x)
            if index != self.n_layers - 1:        # Final layer has no activation
                x = self.hidden_norms[index](x)   # Apply hidden normalization
                x = self.activation(x)            # Shape: Batch * Channel * x * y

        # Remove padding if required
        if any(p > 0 for p in self.padding):
            safe_slice = lambda p: slice(None, -p) if p > 0 else slice(None)
            x = x[..., safe_slice(self.padding[0]), safe_slice(self.padding[1]), safe_slice(self.padding[2])]

        # Convert back to (batch, x, y, z, channels) for output projection
        x = x.permute(0, 2, 3, 4, 1)

        # Output projection
        x = self.fc1(x)  # (batch, x, y, z, hidden_channels)
        x = self.activation(x)  # Apply configurable activation
        x = self.fc2(x)  # (batch, x, y, z, out_channels)

        # Convert back to (batch, out_channels, x, y, z) to match MOR_Operator interface
        x = x.permute(0, 4, 1, 2, 3)

        # Apply output normalization
        x = self.output_norm(x)

        return x

    def get_grid(self, shape, device):
        # shape is now (batch, x, y, z, channels) after permute
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, self.grid_range[2], size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

# %%
""" Model configurations """

PATH = 'data/ns_V1e-4_N10000_T30.mat'
ntrain = 1000
ntest = 100

batch_size = 10
learning_rate = 0.001

epochs = 500
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 2        # lavel of wavelet decomposition
width = 40       # uplifting dimension
layers = 4       # no of wavelet layers

sub = 1          # subsampling rate
h = 64           # total grid size divided by the subsampling rate
grid_range = [1, 1, 1]
in_channel = 13  # input channel is 12: (10 for a(x,t1-t10), 2 for x)

T_in = 10
T = 20           # No of prediction steps
step = 1         # Look-ahead step size

# %%
if __name__ == '__main__':
    """ Read data """
    reader = MatReader(PATH)
    data = reader.read_field('u')
    train_a = data[:ntrain,::sub,::sub,:T_in]
    train_u = data[:ntrain,::sub,::sub,T_in:T+T_in]

    test_a = data[-ntest:,::sub,::sub,:T_in]
    test_u = data[-ntest:,::sub,::sub,T_in:T+T_in]

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    train_a = train_a.reshape(ntrain,h,h,1,T_in).repeat([1,1,1,T,1])
    test_a = test_a.reshape(ntest,h,h,1,T_in).repeat([1,1,1,T,1])

    # %%
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u),
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u),
                                              batch_size=batch_size, shuffle=False)

    # %%
    """ The model definition """
    model = WNO3d(width, level, layers=layers, size=[T, h, h], wavelet=wavelet,
                  in_chanel=in_channel, grid_range=grid_range).to(device)
    print(count_params(model))

    """ Training and testing """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    epoch_loss = torch.zeros(epochs)
    myloss = LpLoss(size_average=False)
    y_normalizer.to(device)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x).squeeze(-1)

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss = myloss(out.view(out.shape[0],-1), y.view(y.shape[0],-1))

            loss.backward()
            optimizer.step()
            train_mse += mse.item()
            train_l2 += loss.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x).squeeze(-1)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out.view(out.shape[0],-1), y.view(y.shape[0],-1)).item()

        train_mse /= len(train_loader)
        train_l2/= ntrain
        epoch_loss[ep] = train_l2
        test_l2 /= ntest
        t2 = default_timer()
        print("Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}"
              .format(ep, t2-t1, train_mse, train_l2, test_l2))

    # %%
    """ Prediction """
    prediction = []
    test_e = []
    with torch.no_grad():
        index = 0
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.to(device), y.to(device)

            out = model(x).squeeze(-1)
            out = y_normalizer.decode(out)

            test_l2 = myloss(out.view(out.shape[0],-1), y.view(y.shape[0],-1)).item()

            test_e.append( test_l2/batch_size )
            prediction.append( out.cpu() )
            print("Batch-{}, Test-loss-{:0.6f}".format( index, test_l2/batch_size ))
            index += 1

    prediction = torch.cat(( prediction ))
    test_e = torch.tensor((test_e))
    print('Mean Error:', 100*torch.mean(test_e))

    # %%
    """ Plotting """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['font.size'] = 14

    figure1, ax = plt.subplots(nrows=4, ncols=10, figsize = (20, 10))
    plt.subplots_adjust(hspace=0.5)
    sample = 15
    index = 0
    for value in range(T):
        if value % 2 == 0:
            print(value)
            if index == 0:
                ax[0, index].imshow(test_a.numpy()[sample,:,:,0,0], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
                ax[0, index].set_title('t={}s'.format(value+10), color='b', fontsize=18, fontweight='bold')
                ax[0, index].set_ylabel('IC', rotation=90, color='r', fontsize=20)

                ax[1, index].imshow(test_u[sample,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
                ax[1, index].set_ylabel('Prediction', rotation=90, color='b', fontsize=20)

                ax[2, index].imshow(prediction[sample,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
                ax[2, index].set_ylabel('Truth', rotation=90, color='g', fontsize=20)

                ax[3, index].imshow(np.abs(test_u[sample,:,:,value]-prediction[sample,:,:,value]),
                                          vmin=0, vmax=0.5, cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
                ax[3, index].set_ylabel('Error', rotation=90, color='purple', fontsize=20)
            else:
                ax[0, index].imshow(test_a.numpy()[sample,:,:,0,0], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
                ax[0, index].set_title('t={}s'.format(value+10), color='b', fontsize=18, fontweight='bold')

                ax[1, index].imshow(test_u[sample,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')

                ax[2, index].imshow(prediction[sample,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')

                if index == 7:
                    im = ax[3, index].imshow(np.abs(test_u[sample,:,:,value]-prediction[sample,:,:,value]),
                                              vmin=0, vmax=0.5, cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
                    plt.colorbar(im, ax=ax[3, index], fraction=0.045)
                else:
                    ax[3, index].imshow(np.abs(test_u[sample,:,:,value]-prediction[sample,:,:,value]),
                                              vmin=0, vmax=0.5, cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
            index = index + 1

    # %%
    """
    For saving the trained model and prediction data
    """
    torch.save(model, 'model/WNO_cwt_navier_stokes_3D')
    scipy.io.savemat('results/wno_cwt_results_navier_stokes_3D.mat', mdict={'test_a':test_a.cpu().numpy(),
                                                        'test_u':test_u.cpu().numpy(),
                                                        'prediction':prediction.cpu().numpy(),
                                                        'test_e':test_e.cpu().numpy()})

