"""
Gradient flows in 2D
====================

Let's showcase the properties of **kernel MMDs**, **Hausdorff**
and **Sinkhorn** divergences on a simple toy problem:
the registration of one blob onto another.
"""
##############################################
# Setup
# ---------------------
import time
from utils import *
import numpy as np
import torch
for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
A = np.load("reconstruct_random_100_shapenetcore55.npy")

ind1=0
ind2=98
device='cuda'
target=A[ind2]
source=torch.randn(target.shape) #A[ind1]
source = source/torch.sqrt(torch.sum(source**2,dim=1,keepdim=True))
learning_rate = 0.01
N_step=500
eps=0
Ls=[10,100]
print_steps = [0,99,199,299,399,499]
X = source#torch.from_numpy(source)
Y = torch.from_numpy(target).to(device)
N=target.shape[0]

Xdetach = X.detach()
Ydetach = Y.detach()


for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= SW(X,Y,L=L,p=2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/SW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/SW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/SW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=False)
    theta = soboleng.draw(L)
    theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
    theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
    theta = (theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))).to(device)
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,theta,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/NQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/NQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/NQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")


for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
            theta = soboleng.draw(L)
            theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
            theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
            theta = (theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))).to(device)
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,theta,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/RNQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/RNQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/RNQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")


for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)

for L in Ls:
    soboleng = torch.quasirandom.SobolEngine(dimension=3, scramble=False)
    theta = soboleng.draw(L)
    theta = torch.clamp(theta, min=1e-6, max=1 - 1e-6)
    theta = torch.from_numpy(norm.ppf(theta) + 1e-6).float()
    theta = (theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))).to(device)
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            U = torch.qr(torch.randn(3, 3,device=device))[0]
            thetaprime = torch.matmul(theta,U)
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,thetaprime,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/RRNQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/RRNQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/RRNQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")


for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)

for L in Ls:
    soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
    net = soboleng.draw(L)
    alpha = net[:, [0]]
    tau = net[:, [1]]
    theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                       2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(device)
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,theta,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/QSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/QSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/QSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")



for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            net = soboleng.draw(L)
            alpha = net[:, [0]]
            tau = net[:, [1]]
            theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                               2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(
                device)
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,theta,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/RQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/RQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/RQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    soboleng = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
    net = soboleng.draw(L)
    alpha = net[:, [0]]
    tau = net[:, [1]]
    theta = torch.cat([2 * torch.sqrt(tau - tau ** 2) * torch.cos(2 * np.pi * alpha),
                       2 * torch.sqrt(tau - tau ** 2) * torch.sin(2 * np.pi * alpha), 1 - 2 * tau], dim=1).to(device)
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            U = torch.qr(torch.randn(3, 3,device=device))[0]
            thetaprime = torch.matmul(theta, U)
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,thetaprime,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/RRQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/RRQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/RRQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
    theta1 = torch.arccos(Z)
    theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    theta = torch.cat([torch.sin(theta1) * torch.cos(theta2), torch.sin(theta1) * torch.sin(theta2), torch.cos(theta1)],
                      dim=1)
    theta = theta.to(device)
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,theta,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/SQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/SQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/SQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    Z = (1 - (2 * torch.arange(1, L + 1) - 1) / L).view(-1, 1)
    theta1 = torch.arccos(Z)
    theta2 = torch.remainder(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    theta = torch.cat([torch.sin(theta1) * torch.cos(theta2), torch.sin(theta1) * torch.sin(theta2), torch.cos(theta1)],
                      dim=1)
    theta = theta.to(device)
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            U = torch.qr(torch.randn(3, 3,device=device))[0]
            thetaprime = torch.matmul(theta, U)
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,thetaprime,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/RSQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/RSQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/RSQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
    theta1 = np.arccos(Z)
    theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)], axis=1)
    theta0 = torch.from_numpy(thetas)
    thetas = torch.randn(L, 3, requires_grad=True)
    thetas.data = theta0
    optimizer = torch.optim.SGD([thetas], lr=1)
    for _ in range(2000):
        loss = - torch.cdist(thetas, thetas, p=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
    theta = thetas.to(device).float()
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,theta,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/ODQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/ODQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/ODQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
    theta1 = np.arccos(Z)
    theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)], axis=1)
    theta0 = torch.from_numpy(thetas)
    thetas = torch.randn(L, 3, requires_grad=True)
    thetas.data = theta0
    optimizer = torch.optim.SGD([thetas], lr=1)
    for _ in range(2000):
        loss = - torch.cdist(thetas, thetas, p=1).mean()
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
    theta = thetas.to(device).float()
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            U = torch.qr(torch.randn(3, 3,device=device))[0]
            thetaprime = torch.matmul(theta, U)
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,thetaprime,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/RODQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/RODQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/RODQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")


for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
    theta1 = np.arccos(Z)
    theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)], axis=1)
    theta0 = torch.from_numpy(thetas)
    thetas = torch.randn(L, 3, requires_grad=True)
    thetas.data = theta0
    optimizer = torch.optim.SGD([thetas], lr=1)
    for _ in range(2000):
        loss = (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
    theta = thetas.to(device).float()
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,theta,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/OCQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/OCQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/OCQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")

for _ in range(1000):
    torch.randn(100)
    np.random.rand(100)
for L in Ls:
    Z = (1 - (2 * np.arange(1, L + 1) - 1) / L).reshape(-1, 1)
    theta1 = np.arccos(Z)
    theta2 = np.mod(1.8 * np.sqrt(L) * theta1, 2 * np.pi)
    thetas = np.concatenate([np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)], axis=1)
    theta0 = torch.from_numpy(thetas)
    thetas = torch.randn(L, 3, requires_grad=True)
    thetas.data = theta0
    optimizer = torch.optim.SGD([thetas], lr=1)
    for _ in range(2000):
        loss = (1 / (torch.cdist(thetas, thetas, p=1) + 1e-6)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        thetas.data = thetas.data / torch.sqrt(torch.sum(thetas.data ** 2, dim=1, keepdim=True))
    theta = thetas.to(device).float()
    for seed in [1,2,3]:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        X=torch.tensor(source, requires_grad=True,device=device)
        optimizer = torch.optim.SGD([X], lr=learning_rate)
        points=[]
        caltimes=[]
        distances=[]
        start = time.time()
        for i in range(N_step):
            if (i in print_steps):
                distance,cal_time=compute_true_Wasserstein(X, Y), time.time() - start
                print("W {}:{} ({}s)".format(i + 1, distance,np.round(cal_time,2)))
                points.append(X.clone().cpu().data.numpy())
                caltimes.append(cal_time)
                distances.append(distance)
            optimizer.zero_grad()
            U = torch.qr(torch.randn(3, 3,device=device))[0]
            thetaprime = torch.matmul(theta, U)
            sw= torch.pow(one_dimensional_Wasserstein_prod(X,Y,thetaprime,p=2).mean(),1./2)
            loss= N*sw
            loss.backward()
            optimizer.step()
        points.append(Y.clone().cpu().data.numpy())
        np.save("saved/ROCQSW_L{}_{}_{}_points_seed{}.npy".format(L,ind1,ind2,seed),np.stack(points))
        np.savetxt("saved/ROCQSW_L{}_{}_{}_distances_seed{}.txt".format(L,ind1,ind2,seed), np.array(distances), delimiter=",")
        np.savetxt("saved/ROCQSW_L{}_{}_{}_times_seed{}.txt".format(L,ind1,ind2,seed), np.array(caltimes), delimiter=",")