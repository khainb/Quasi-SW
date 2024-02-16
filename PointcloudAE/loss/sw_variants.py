import os.path as osp
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.stats import ortho_group
from scipy.stats import norm

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def minibatch_rand_projections(batchsize, dim, num_projections=1000, device='cuda', **kwargs):
    projections = torch.randn((batchsize, num_projections, dim), device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=2, keepdim=True))
    return projections


def proj_onto_unit_sphere(vectors, dim=2):
    """
    input: vectors: [batchsize, num_projs, dim]
    """
    return vectors / torch.sqrt(torch.sum(vectors ** 2, dim=dim, keepdim=True))


def _sample_minibatch_orthogonal_projections(batch_size, dim, num_projections, device='cuda'):
    projections = torch.zeros((batch_size, num_projections, dim), device=device)
    projections = torch.stack([torch.nn.init.orthogonal_(projections[i]) for i in range(projections.shape[0])], dim=0)
    return projections


def compute_practical_moments_sw(x, y, num_projections=30, device="cuda", degree=2.0, **kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    dim = x.size(2)
    batch_size = x.size(0)
    projections = minibatch_rand_projections(batch_size, dim, num_projections, device=device)
    # projs.shape: [batchsize, num_projs, dim]

    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = torch.pow(_sort_pow_p_get_sum.mean(dim=1),1./degree)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def compute_practical_moments_sw_with_predefined_projections(x, y, projections, device="cuda", degree=2.0, **kwargs):
    """
    x, y: [batch size, num points, dim]
    projections: [batch size, num projs, dim]
    """
    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = torch.pow(_sort_pow_p_get_sum.mean(dim=1),1./degree)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def _compute_practical_moments_sw_with_projected_data(xproj, yproj, device="cuda", degree=2.0, **kwargs):
    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def _circular(x, theta):
    """The circular defining function for generalized Radon transform
    Inputs
    X:  [batch size, num_points, d] - d: dim of 1 point
    theta: [batch size, L, d] that parameterizes for L projections
    """
    x_s = torch.stack([x for _ in range(theta.shape[1])], dim=2)
    theta_s = torch.stack([theta for _ in range(x.shape[1])], dim=1)
    z_s = x_s - theta_s
    return torch.sqrt(torch.sum(z_s ** 2, dim=3))


def _linear(x, theta):
    """
    x: [batch size, num_points, d] - d: dim of 1 point
    theta: [batch size, L, d] that parameterizes for L projections
    """
    xproj = x.bmm(theta.transpose(1, 2))
    return xproj


class SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        squared_sw_2, _ = compute_practical_moments_sw(x, y, num_projections=self.num_projs, device=self.device)
        return {"loss": squared_sw_2.mean(dim=0)}
    
    
    

class Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        soboleng = torch.quasirandom.SobolEngine(dimension=2,scramble=False)
        thetas = []
        net = soboleng.draw( self.num_projs ).to(self.device)
        alpha = net[:,[0]]
        tau = net[:,[1]]
        theta = torch.cat([2*torch.sqrt(tau-tau**2) *torch.cos(2*np.pi*alpha), 2*torch.sqrt(tau-tau**2) *torch.sin(2*np.pi*alpha),1-2*tau],dim=1)
        theta = theta.expand(x.shape[0],self.num_projs,3)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}

    
class Naive_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        soboleng = torch.quasirandom.SobolEngine(dimension=3,scramble=False)
        thetas = []
        net = soboleng.draw( self.num_projs )
        net=torch.clamp(net, min=1e-6, max=1e-6)
        net =  torch.from_numpy(norm.ppf(net)+1e-6).float()
        theta = net/torch.sqrt(torch.sum(net**2,dim=1,keepdim=True))
        theta = theta.expand(x.shape[0],self.num_projs,3).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}    

class Randomized_Naive_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        soboleng = torch.quasirandom.SobolEngine(dimension=3,scramble=True)
        thetas = []
        for _ in range(x.shape[0]):
            net = soboleng.draw( self.num_projs )
            net=torch.clamp(net, min=1e-6, max=1-1e-6)
            net =  torch.from_numpy(norm.ppf(net)+1e-6).float()
            theta = net/torch.sqrt(torch.sum(net**2,dim=1,keepdim=True))
            thetas.append(theta.to(self.device))
        theta = torch.stack(thetas,dim=0)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}        
class Randomized_Rotation_Naive_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        soboleng = torch.quasirandom.SobolEngine(dimension=3,scramble=False)
        thetas = []
        net = soboleng.draw( self.num_projs )
        net=torch.clamp(net, min=1e-6, max=1-1e-6)
        net =  torch.from_numpy(norm.ppf(net)+1e-6).float()
        theta = net/torch.sqrt(torch.sum(net**2,dim=1,keepdim=True))
        theta = theta
        for _ in range(x.shape[0]):
            U =torch.qr(torch.randn(3,3))[0]
            thetas.append(torch.matmul(theta,U))
        theta = torch.stack(thetas,dim=0).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}

class OptC_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        L = self.num_projs
        Z = (1 - (2*np.arange(1, L+1)-1) / L).reshape(-1, 1)
        theta1=np.arccos(Z)
        theta2=np.mod(1.8 * np.sqrt(L) * theta1,2*np.pi)
        theta = np.concatenate([np.sin(theta1)*np.cos(theta2),np.sin(theta1)*np.sin(theta2),np.cos(theta1)], axis=1)
        theta0 =torch.from_numpy(theta).to(self.device).float()
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        # thetas.data = thetas.data/torch.sqrt(torch.sum(thetas.data**2,dim=1,keepdim=True))
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(2000):
            loss = (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        self.theta =torch.randn(L, 3)
        self.theta.data = thetas.data
    def forward(self, x, y, **kwargs):
        theta = self.theta.expand(x.shape[0],self.num_projs,3).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}  
class Randomized_OptC_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        L = self.num_projs
        Z = (1 - (2*np.arange(1, L+1)-1) / L).reshape(-1, 1)
        theta1=np.arccos(Z)
        theta2=np.mod(1.8 * np.sqrt(L) * theta1,2*np.pi)
        theta = np.concatenate([np.sin(theta1)*np.cos(theta2),np.sin(theta1)*np.sin(theta2),np.cos(theta1)], axis=1)
        theta0 =torch.from_numpy(theta).float()
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        # thetas.data = thetas.data/torch.sqrt(torch.sum(thetas.data**2,dim=1,keepdim=True))
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(2000):
            loss = (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        self.theta =torch.randn(L, 3)
        self.theta.data = thetas.data
    def forward(self, x, y, **kwargs):
        thetas=[]
        for _ in range(x.shape[0]):
            U = torch.qr(torch.randn(3,3))[0]
            theta = torch.matmul(self.theta,U)
            thetas.append(theta)
        theta = torch.stack(thetas,dim=0).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}  
    
class OptD_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        L = self.num_projs
        Z = (1 - (2*np.arange(1, L+1)-1) / L).reshape(-1, 1)
        theta1=np.arccos(Z)
        theta2=np.mod(1.8 * np.sqrt(L) * theta1,2*np.pi)
        theta = np.concatenate([np.sin(theta1)*np.cos(theta2),np.sin(theta1)*np.sin(theta2),np.cos(theta1)], axis=1)
        theta0 =torch.from_numpy(theta).to(self.device).float()
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        # thetas.data = thetas.data/torch.sqrt(torch.sum(thetas.data**2,dim=1,keepdim=True))
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(2000):
            loss = -((torch.cdist(thetas, thetas, p=1))).mean()
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        self.theta =torch.randn(L, 3)
        self.theta.data = thetas.data
    def forward(self, x, y, **kwargs):
        theta = self.theta.expand(x.shape[0],self.num_projs,3).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}  
class Randomized_OptD_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        L = self.num_projs
        Z = (1 - (2*np.arange(1, L+1)-1) / L).reshape(-1, 1)
        theta1=np.arccos(Z)
        theta2=np.mod(1.8 * np.sqrt(L) * theta1,2*np.pi)
        theta = np.concatenate([np.sin(theta1)*np.cos(theta2),np.sin(theta1)*np.sin(theta2),np.cos(theta1)], axis=1)
        theta0 =torch.from_numpy(theta).float()
        thetas = torch.randn(L, 3, requires_grad=True)
        thetas.data = theta0
        # thetas.data = thetas.data/torch.sqrt(torch.sum(thetas.data**2,dim=1,keepdim=True))
        optimizer = torch.optim.SGD([thetas], lr=1)
        for _ in range(2000):
            loss = -((torch.cdist(thetas, thetas, p=1))).mean()
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
        self.theta =torch.randn(L, 3)
        self.theta.data = thetas.data
    def forward(self, x, y, **kwargs):
        thetas=[]
        for _ in range(x.shape[0]):
            U = torch.qr(torch.randn(3,3))[0]
            theta = torch.matmul(self.theta,U)
            thetas.append(theta)
        theta = torch.stack(thetas,dim=0).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}  
class Spiral_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        thetas = []
        L = self.num_projs
        Z = (1 - (2*np.arange(1, L+1)-1) / L).reshape(-1, 1)
        theta1=np.arccos(Z)
        theta2=np.mod(1.8 * np.sqrt(L) * theta1,2*np.pi)
        theta = np.concatenate([np.sin(theta1)*np.cos(theta2),np.sin(theta1)*np.sin(theta2),np.cos(theta1)], axis=1)
        theta =torch.from_numpy(theta).to(self.device).float()
        theta = theta.expand(x.shape[0],self.num_projs,3)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}

class Radomized_Spiral_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        thetas = []
        L = self.num_projs
        Z = (1 - (2*np.arange(1, L+1)-1) / L).reshape(-1, 1)
        theta1=np.arccos(Z)
        theta2=np.mod(1.8 * np.sqrt(L) * theta1,2*np.pi)
        theta = np.concatenate([np.sin(theta1)*np.cos(theta2),np.sin(theta1)*np.sin(theta2),np.cos(theta1)], axis=1)
        theta =torch.from_numpy(theta).float()
        for _ in range(x.shape[0]):
            U = torch.qr(torch.randn(3,3))[0]
            theta = torch.matmul(theta,U)
            thetas.append(theta)
        theta = torch.stack(thetas,dim=0).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}    
class Randomized_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        soboleng = torch.quasirandom.SobolEngine(dimension=2,scramble=True)
        thetas = []
        for _ in range(x.shape[0]):
            net = soboleng.draw( self.num_projs ).to(self.device)
            alpha = net[:,[0]]
            tau = net[:,[1]]
            theta = torch.cat([2*torch.sqrt(tau-tau**2) *torch.cos(2*np.pi*alpha), 2*torch.sqrt(tau-tau**2) *torch.sin(2*np.pi*alpha),1-2*tau],dim=1)
            thetas.append(theta)
        theta = torch.stack(thetas,dim=0)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}
    
class Randomized_Rotation_Quasi_SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        soboleng = torch.quasirandom.SobolEngine(dimension=2,scramble=False)
        net = soboleng.draw( self.num_projs )
        alpha = net[:,[0]]
        tau = net[:,[1]]
        theta = torch.cat([2*torch.sqrt(tau-tau**2) *torch.cos(2*np.pi*alpha), 2*torch.sqrt(tau-tau**2) *torch.sin(2*np.pi*alpha),1-2*tau],dim=1)
        thetas = []
        for _ in range(x.shape[0]):
            U =torch.qr(torch.randn(3,3))[0]
            thetas.append(torch.matmul(theta,U))
        theta = torch.stack(thetas,dim=0).to(self.device)
        distances = torch.sqrt(compute_projected_distances(x, y, theta,degree=2.0).mean(dim=1))
        return {"loss": distances.mean()}
def compute_projected_distances(x, y, projections, degree=2.0, **kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    # projs.shape: [batchsize, num_projs, dim]

    xproj = x.bmm(projections.transpose(1, 2))

    yproj = y.bmm(projections.transpose(1, 2))

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    return _sort_pow_p_get_sum



