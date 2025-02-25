import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
class ModelFlow_stride(nn.Module):
    def __init__(self, in_channel, n_classes,start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        
        super(ModelFlow_stride, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=False)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=False)
        self.ec2 = self.encoder(self.start_channel, self.start_channel*2, stride=2, bias=False)
        self.ec3 = self.encoder(self.start_channel*2, self.start_channel*2, bias=False)
        self.ec4 = self.encoder(self.start_channel*2, self.start_channel*4, stride=2, bias=False)
        self.ec5 = self.encoder(self.start_channel*4, self.start_channel*4, bias=False)
        self.ec6 = self.encoder(self.start_channel*4, self.start_channel*8, stride=2, bias=False)
        self.ec7 = self.encoder(self.start_channel*8, self.start_channel*8, bias=False)
        self.ec8 = self.encoder(self.start_channel*8, self.start_channel*16, stride=2, bias=False)
        self.ec9 = self.encoder(self.start_channel*16, self.start_channel*8, bias=False)


        self.dc1 = self.encoder(self.start_channel*8+self.start_channel*8, self.start_channel*8, kernel_size=3, stride=1, bias=False)
        self.dc2 = self.encoder(self.start_channel*8, self.start_channel*4, kernel_size=3, stride=1, bias=False)          
        self.dc3 = self.encoder(self.start_channel*4+self.start_channel*4, self.start_channel*4, kernel_size=3, stride=1, bias=False)
        self.dc4 = self.encoder(self.start_channel*4, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc5 = self.encoder(self.start_channel*2+self.start_channel*2, self.start_channel*4, kernel_size=3, stride=1, bias=False)
        self.dc6 = self.encoder(self.start_channel*4, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc7 = self.encoder(self.start_channel*2+self.start_channel*1, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc8 = self.encoder(self.start_channel*2, self.start_channel*2, kernel_size=3, stride=1, bias=False)
        self.dc9 = self.outputs(self.start_channel*2, self.n_classes, kernel_size=1, stride=1,padding=0, bias=False)
        
        self.up1 = self.decoder(self.start_channel*8, self.start_channel*8)
        self.up2 = self.decoder(self.start_channel*4, self.start_channel*4)
        self.up3 = self.decoder(self.start_channel*2, self.start_channel*2)
        self.up4 = self.decoder(self.start_channel*2, self.start_channel*2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer


    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Tanh())
        return layer

    def forward(self, x,y):
        x_in=torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)
        
    
        e1 = self.ec2(e0)
        e1 = self.ec3(e1)


 
        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

  
        e3 = self.ec6(e2)
        e3 = self.ec7(e3)
        
        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)


        d0 = self.dc1(d0)
        d0 = self.dc2(d0)


        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)
        
        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)        
        
        d3 = self.dc9(d3)
        
        return d3


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, x,flow,sample_grid):
        sample_grid = sample_grid+flow
        size_tensor = sample_grid.size()
        sample_grid[0,:,:,:,0] = (sample_grid[0,:,:,:,0]-(size_tensor[3]/2))/size_tensor[3]*2
        sample_grid[0,:,:,:,1] = (sample_grid[0,:,:,:,1]-(size_tensor[2]/2))/size_tensor[2]*2
        sample_grid[0,:,:,:,2] = (sample_grid[0,:,:,:,2]-(size_tensor[1]/2))/size_tensor[1]*2  
        flow = torch.nn.functional.grid_sample(x, sample_grid,mode = 'bilinear')
        
        return flow    

def antifoldloss(y_pred):
    dy = y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :]-1
    dx = y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :]-1
    dz = y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:]-1

    dy = F.relu(dy) * torch.abs(dy*dy)
    dx = F.relu(dx) * torch.abs(dx*dx)
    dz = F.relu(dz) * torch.abs(dz*dz)
    return (torch.mean(dx)+torch.mean(dy)+torch.mean(dz))/3.0


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0

def mse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f-y_pred_f
    mse = torch.mul(diff,diff).mean()   
    return mse

def l1_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f-y_pred_f
    mse = torch.abs(diff).mean()
    return mse

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = int(np.prod(win))
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)