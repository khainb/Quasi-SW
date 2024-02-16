import numpy as np
import torch
from torch.autograd import Variable
import ot
import random
import tqdm
from scipy.stats import norm
def compute_true_Wasserstein(X,Y,p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance


def SW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)



def transform_SW(src,target,src_label,origin,sw_type='sw',L=10,num_iter=1000):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    device='cpu'
    s = np.array(src).reshape(-1, 3)
    s = torch.from_numpy(s).float()
    s = torch.nn.parameter.Parameter(s)
    t = np.array(target).reshape(-1, 3)
    t = torch.from_numpy(t).float()
    opt = torch.optim.SGD([s], lr=1)
    if (sw_type == 'nqsw' or sw_type == 'rnqsw' or sw_type == 'rrnqsw'  ):
        soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=False)
        theta = soboleng.draw(L)
        theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
        theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
        theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
    elif(sw_type=='qsw' or sw_type=='rqsw' or sw_type=='rrqsw'):
        soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
        net = soboleng.draw(L)
        alpha = net[:, [0]]
        tau = net[:, [1]]
        theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                           2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(
            device)
    elif(sw_type=='sqsw' or sw_type=='rsqsw'):
        Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
        theta1 = torch.arccos(Z)
        theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        theta = torch.cat(
            [torch.sin(theta1) * torch.cos(theta2), torch.sin(theta1) * torch.sin(theta2), torch.cos(theta1)],
            dim=1)
        theta = theta.to(device)
    elif(sw_type=='odqsw' or sw_type=='rodqsw'):
        Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
        theta1 = np.arccos(Z)
        theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)],
                                axis=1)
        theta0 = torch.from_numpy(thetas)
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(100):
            loss = - torch.cdist(thetas, thetas, p=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        theta = thetas.to(device).float()
    elif (sw_type == 'ocqsw' or sw_type=='rocqsw'):
        Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
        theta1 = np.arccos(Z)
        theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
        thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)],
                                axis=1)
        theta0 = torch.from_numpy(thetas)
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(100):
            loss = (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        theta = thetas.to(device).float()
    for _ in tqdm.tqdm(range(num_iter)):
        opt.zero_grad()
        if (sw_type == 'sw'):
            g_loss = SW(s, t, L=L,p=2)
        elif(sw_type=='nqsw' or sw_type=='qsw' or sw_type=='sqsw' or sw_type=='ocqsw' or sw_type=='odqsw'):
            g_loss=one_dimensional_Wasserstein_prod(s,t,theta,p=2)
        elif(sw_type=='rnqsw'):
            soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
            theta = soboleng.draw(L)
            theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
            theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
            theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True)).to(device)
            g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)
        elif(sw_type=='rqsw'):
            soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            net = soboleng.draw(L)
            alpha = net[:, [0]]
            tau = net[:, [1]]
            theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                               2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(
                device)
            g_loss = one_dimensional_Wasserstein_prod(s, t, theta, p=2)
        elif (sw_type == 'rrnqsw' or sw_type == 'rrqsw' or sw_type == 'rsqsw' or sw_type == 'rocqsw' or sw_type == 'rodqsw'):
            U = torch.qr(torch.randn(3, 3))[0]
            thetaprime = torch.matmul(theta, U)
            g_loss = one_dimensional_Wasserstein_prod(s, t, thetaprime, p=2)
        g_loss =torch.sqrt(g_loss.mean())
        g_loss = g_loss*s.shape[0]
        opt.zero_grad()
        g_loss.backward()
        opt.step()
        s.data = torch.clamp(s, min=0)
    s = torch.clamp( s,min=0).cpu().detach().numpy()
    img_ot_transf = s[src_label].reshape(origin.shape)
    img_ot_transf = img_ot_transf / np.max(img_ot_transf) * 255
    img_ot_transf = img_ot_transf.astype("uint8")
    return s, img_ot_transf

