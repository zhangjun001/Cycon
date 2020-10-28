import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from torch.autograd import Variable
from Models import ModelFlow_stride, SpatialTransform, mse_loss, smoothloss, NCC, l1_loss
from Functions import Dataset, generate_grid
import torch.utils.data as Data

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=20001,
                    help="number of total iterations")
parser.add_argument("--lambda", type=float,
                    dest="lambda_", default=1.5,
                    help="Smooth：suggested range 0.01 to 10")
parser.add_argument("--alpha", type=float,
                    dest="alpha", default=1,
                    help="Cycle: suggested range 0.1 to 10")
parser.add_argument("--beta", type=float,
                    dest="beta", default=1,
                    help="Indentity: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath", default='../Dataset/',
                    help="data path for training images")
opt = parser.parse_args()

lr = opt.lr
iteration = opt.iteration
start_channel = opt.start_channel
alpha = opt.alpha
beta = opt.beta
lambda_ = opt.lambda_
n_checkpoint = opt.checkpoint
datapath = opt.datapath


def train():
    model = ModelFlow_stride(2, 3, start_channel).cuda()
    loss_similarity = NCC().loss
    loss_cycle = l1_loss
    loss_smooth = smoothloss
    transform = SpatialTransform().cuda()
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    names = glob.glob(datapath + '/*.gz')
    grid = generate_grid(imgshape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    print(grid.type())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    lossall = np.zeros((5, iteration))
    training_generator = Data.DataLoader(Dataset(names, iteration, False), batch_size=1,
                                         shuffle=False, num_workers=2)
    step = 0
    for X, Y in training_generator:

        X = X.cuda().float()
        Y = Y.cuda().float()
        F_xy = model(X, Y)
        F_yx = model(Y, X)

        X_Y = transform(X, F_xy.permute(0, 2, 3, 4, 1) * range_flow, grid)
        Y_X = transform(Y, F_yx.permute(0, 2, 3, 4, 1) * range_flow, grid)

        F_xy_ = model(Y_X, X_Y)
        F_yx_ = model(X_Y, Y_X)

        Y_X_Y = transform(Y_X, F_xy_.permute(0, 2, 3, 4, 1) * range_flow, grid)
        X_Y_X = transform(X_Y, F_yx_.permute(0, 2, 3, 4, 1) * range_flow, grid)

        F_xx = model(X, X)
        F_yy = model(Y, Y)
        X_X = transform(X, F_xx.permute(0, 2, 3, 4, 1) * range_flow, grid)
        Y_Y = transform(Y, F_yy.permute(0, 2, 3, 4, 1) * range_flow, grid)

        L_smooth = loss_smooth(F_xy * range_flow) + loss_smooth(F_yx * range_flow)
        L_regist = loss_similarity(Y, X_Y) + loss_similarity(X, Y_X) + \
                      lambda_ * L_smooth
        L_cycle = loss_cycle(X, X_Y_X) + loss_cycle(Y, Y_X_Y)
        L_identity = loss_similarity(X, X_X) + loss_similarity(Y, Y_Y)
        loss = L_regist + alpha * L_cycle + beta * L_identity

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        lossall[:, step] = np.array([loss.data[0], L_regist.data[0], L_cycle.data[0], L_identity.data[0], L_smooth.data[0]])
        sys.stdout.write(
            "\r" + 'step "{0}" -> training loss "{1:.4f}" - reg "{2:.4f}" - cyc "{3:.4f}" - ind "{4:.4f}" -smo "{5:.4f}" '.format(
                step, loss.data[0], L_regist.data[0], L_cycle.data[0], L_identity.data[0], L_smooth.data[0]))
        sys.stdout.flush()
        if (step % n_checkpoint == 0):
            modelname = model_dir + '/' + str(step) + '.pth'
            torch.save(model.state_dict(), modelname)
        step += 1
    np.save(model_dir + '/loss.npy', lossall)


if __name__ == '__main__':
    imgshape = (144, 192, 160)
    range_flow = 7
    train()
