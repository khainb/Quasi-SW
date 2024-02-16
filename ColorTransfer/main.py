import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.transform import resize
import os
import random
from sklearn import cluster
from tqdm import tqdm
import sys
import torch
import time
import argparse
from utils import *
import ot
np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='CT')
parser.add_argument('--L', type=int, default=3, metavar='N',
                    help='input batch size for training (default: 100)')
# parser.add_argument('--k', type=int, default=3, metavar='N',
#                     help='input num batch for training (default: 200)')
parser.add_argument('--num_iter', type=int, default=10000, metavar='N',
                    help='Num Interations')
# parser.add_argument('--T', type=int, default=1, metavar='N',
#                     help='Num Interations')
# parser.add_argument('--N', type=int, default=1, metavar='N',
#                     help='Num Interations')
# parser.add_argument('--M', type=int, default=0, metavar='N',
#                     help='Num Interations')
parser.add_argument('--source', type=str, metavar='N',
                    help='Source')
parser.add_argument('--target', type=str, metavar='N',
                    help='Target')
parser.add_argument('--cluster',  action='store_true',
                    help='Use clustering')
parser.add_argument('--load',  action='store_true',
                    help='Load precomputed')
parser.add_argument('--palette',  action='store_true',
                    help='Show color palette')
# parser.add_argument('--sw_type', type=str, metavar='N',
#                     help='Target')


args = parser.parse_args()


n_clusters = 3000
name1=args.source#path to images 1
name2=args.target#path to images 2
source = img_as_ubyte(io.imread(name1))
target = img_as_ubyte(io.imread(name2))
reshaped_target = img_as_ubyte(resize(target, source.shape[:2]))
name1=name1.replace('/', '')
name2=name2.replace('/', '')
if(args.cluster):
    X = source.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    source_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    source_k_means.fit(X)
    source_values = source_k_means.cluster_centers_.squeeze()
    source_labels = source_k_means.labels_

    # create an array from labels and values
    #source_compressed = np.choose(labels, values)
    source_compressed = source_values[source_labels]
    source_compressed.shape = source.shape

    vmin = source.min()
    vmax = source.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Source")
    plt.imshow(source,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Source")
    plt.imshow(source_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)
    os.makedirs('npzfiles', exist_ok=True)
    with open('npzfiles/'+name1+'source_compressed.npy', 'wb') as f:
        np.save(f, source_compressed)
    with open('npzfiles/'+name1+'source_values.npy', 'wb') as f:
        np.save(f, source_values)
    with open('npzfiles/'+name1+'source_labels.npy', 'wb') as f:
        np.save(f, source_labels)
    np.random.seed(0)

    X = target.reshape((-1, 3))  # We need an (n_sample, n_feature) array
    target_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init=4, batch_size=100)
    target_k_means.fit(X)
    target_values = target_k_means.cluster_centers_.squeeze()
    target_labels = target_k_means.labels_

    # create an array from labels and values
    target_compressed = target_values[target_labels]
    target_compressed.shape = target.shape

    vmin = target.min()
    vmax = target.max()

    # original image
    plt.figure(1, figsize=(5, 5))
    plt.title("Original Target")
    plt.imshow(target,  vmin=vmin, vmax=256)

    # compressed image
    plt.figure(2, figsize=(5, 5))
    plt.title("Compressed Target")
    plt.imshow(target_compressed.astype('uint8'),  vmin=vmin, vmax=vmax)

    with open('npzfiles/'+name2+'target_compressed.npy', 'wb') as f:
        np.save(f, target_compressed)
    with open('npzfiles/'+name2+'target_values.npy', 'wb') as f:
        np.save(f, target_values)
    with open('npzfiles/'+name2+'target_labels.npy', 'wb') as f:
        np.save(f, target_labels)
else:
    with open('npzfiles/'+name1+'source_compressed.npy', 'rb') as f:
        source_compressed = np.load(f)
    with open('npzfiles/'+name2+'target_compressed.npy', 'rb') as f:
        target_compressed = np.load(f)
    with open('npzfiles/'+name1+'source_values.npy', 'rb') as f:
        source_values = np.load(f)
    with open('npzfiles/'+name2+'target_values.npy', 'rb') as f:
        target_values = np.load(f)
    with open('npzfiles/'+name1+'source_labels.npy', 'rb') as f:
        source_labels = np.load(f)



for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
SWcluster,SW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='sw',num_iter=args.num_iter)
SWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
NQSWcluster,NQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='nqsw',num_iter=args.num_iter)
NQSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
RNQSWcluster,RNQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='rnqsw',num_iter=args.num_iter)
RNQSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
RRNQSWcluster,RRNQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='rrnqsw',num_iter=args.num_iter)
RRNQSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
QSWcluster,QSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='qsw',num_iter=args.num_iter)
QSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
RQSWcluster,RQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='rqsw',num_iter=args.num_iter)
RQSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
RRQSWcluster,RRQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='rrqsw',num_iter=args.num_iter)
RRQSWtime = np.round(time.time() - start,2)


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
SQSWcluster,SQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='sqsw',num_iter=args.num_iter)
SQSWtime = np.round(time.time() - start,2)


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
RSQSWcluster,RSQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='rsqsw',num_iter=args.num_iter)
RSQSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
ODQSWcluster,ODQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='odqsw',num_iter=args.num_iter)
ODQSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
RODQSWcluster,RODQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='rodqsw',num_iter=args.num_iter)
RODQSWtime = np.round(time.time() - start,2)


for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
OCQSWcluster,OCQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='ocqsw',num_iter=args.num_iter)
OCQSWtime = np.round(time.time() - start,2)

for _ in range(1000):
    a = np.random.randn(100)
    a = torch.randn(100)

start = time.time()
ROCQSWcluster,ROCQSW = transform_SW(source_values,target_values,source_labels,source,L=args.L,sw_type='rocqsw',num_iter=args.num_iter)
ROCQSWtime = np.round(time.time() - start,2)

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
source3=source_values.reshape(-1,3)
reshaped_target3=target_values.reshape(-1,3)

SWcluster=SWcluster/np.max(SWcluster)*255
NQSWcluster=NQSWcluster/np.max(NQSWcluster)*255
RNQSWcluster=RNQSWcluster/np.max(RNQSWcluster)*255
RRNQSWcluster=RRNQSWcluster/np.max(RRNQSWcluster)*255
QSWcluster=QSWcluster/np.max(QSWcluster)*255
RQSWcluster=RQSWcluster/np.max(RQSWcluster)*255
RRQSWcluster=RRQSWcluster/np.max(RRQSWcluster)*255
SQSWcluster=SQSWcluster/np.max(SQSWcluster)*255
RSQSWcluster=RSQSWcluster/np.max(RSQSWcluster)*255
ODQSWcluster=ODQSWcluster/np.max(ODQSWcluster)*255
RODQSWcluster=RODQSWcluster/np.max(RODQSWcluster)*255
OCQSWcluster=OCQSWcluster/np.max(OCQSWcluster)*255
ROCQSWcluster=ROCQSWcluster/np.max(ROCQSWcluster)*255

# f.suptitle("L={}, k={}, T={}".format(L, k, iter), fontsize=20)
C_SW = ot.dist(SWcluster,reshaped_target3)
C_NQSW = ot.dist(NQSWcluster,reshaped_target3)
C_RNQSW = ot.dist(RNQSWcluster,reshaped_target3)
C_RRNQSW = ot.dist(RNQSWcluster,reshaped_target3)
C_QSW = ot.dist(QSWcluster,reshaped_target3)
C_RQSW = ot.dist(RQSWcluster,reshaped_target3)
C_RRQSW = ot.dist(RRQSWcluster,reshaped_target3)
C_SQSW = ot.dist(SQSWcluster,reshaped_target3)
C_RSQSW = ot.dist(RSQSWcluster,reshaped_target3)
C_ODQSW = ot.dist(ODQSWcluster,reshaped_target3)
C_RODQSW = ot.dist(RODQSWcluster,reshaped_target3)
C_OCQSW = ot.dist(OCQSWcluster,reshaped_target3)
C_ROCQSW = ot.dist(ROCQSWcluster,reshaped_target3)

W_SW = np.round(ot.emd2([],[],C_SW),2)
W_NQSW = np.round(ot.emd2([],[],C_NQSW),2)
W_RNQSW = np.round(ot.emd2([],[],C_RNQSW),2)
W_RRNQSW = np.round(ot.emd2([],[],C_RRNQSW),2)
W_QSW = np.round(ot.emd2([],[],C_QSW),2)
W_RQSW = np.round(ot.emd2([],[],C_RQSW),2)
W_RRQSW = np.round(ot.emd2([],[],C_RRQSW),2)
W_SQSW = np.round(ot.emd2([],[],C_SQSW),2)
W_RSQSW = np.round(ot.emd2([],[],C_RSQSW),2)
W_ODQSW = np.round(ot.emd2([],[],C_ODQSW),2)
W_RODQSW = np.round(ot.emd2([],[],C_RODQSW),2)
W_OCQSW = np.round(ot.emd2([],[],C_OCQSW),2)
W_ROCQSW = np.round(ot.emd2([],[],C_ROCQSW),2)
# SW_SW = np.round(SW(SWcluster.float(),torch.from_numpy(reshaped_target3).float(),L=100000))




f, ax = plt.subplots(3, 5, figsize=(12, 5))
ax[0,0].set_title('Source', fontsize=14)
ax[0,0].imshow(source)


ax[0,1].set_title('SW, $W_2={}$'.format(W_SW), fontsize=12)
ax[0,1].imshow(SW)


ax[0,2].set_title('GQSW, $W_2={}$'.format(W_NQSW), fontsize=12)
ax[0,2].imshow(NQSW)

ax[0,3].set_title('RGQSW, $W_2={}$'.format(W_RNQSW), fontsize=12)
ax[0,3].imshow(RNQSW)

ax[0,4].set_title('RRGQSW, $W_2={}$'.format(W_RRNQSW), fontsize=12)
ax[0,4].imshow(RRNQSW)

ax[1,0].set_title('EQSW, $W_2={}$'.format(W_QSW), fontsize=12)
ax[1,0].imshow(QSW)

ax[1,1].set_title('REQSW, $W_2={}$'.format(W_RQSW), fontsize=12)
ax[1,1].imshow(RQSW)

ax[1,2].set_title('RREQSW, $W_2={}$'.format(W_RQSW), fontsize=12)
ax[1,2].imshow(RQSW)

ax[1,3].set_title('SQSW, $W_2={}$'.format(W_SQSW), fontsize=12)
ax[1,3].imshow(SQSW)

ax[1,4].set_title('RSQSW, $W_2={}$'.format(W_RSQSW), fontsize=12)
ax[1,4].imshow(RSQSW)

ax[2,0].set_title('DQSW, $W_2={}$'.format(W_ODQSW), fontsize=12)
ax[2,0].imshow(ODQSW)

ax[2,1].set_title('RDQSW, $W_2={}$'.format(W_RODQSW), fontsize=12)
ax[2,1].imshow(RODQSW)

ax[2,2].set_title('CQSW, $W_2={}$'.format(W_OCQSW), fontsize=12)
ax[2,2].imshow(OCQSW)


ax[2,3].set_title('RCQSW, $W_2={}$'.format(W_ROCQSW), fontsize=12)
ax[2,3].imshow(ROCQSW)

ax[2,4].set_title('Target', fontsize=14)
ax[2,4].imshow(reshaped_target)
# ax[3,3].scatter(reshaped_target3[:, 0], reshaped_target3[:, 1], reshaped_target3[:, 2], c=reshaped_target3 / 255)

for i in range(3):
    for j in range(5):
        ax[i,j].get_yaxis().set_visible(False)
        ax[i,j].get_xaxis().set_visible(False)

plt.tight_layout()
plt.subplots_adjust(left=0, right=1, top=0.88, bottom=0.01, wspace=0, hspace=0.145)
plt.show()


