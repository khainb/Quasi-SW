# Quasi-SW
Official PyTorch implementation for paper:  [Quasi-Monte Carlo for 3D Sliced Wasserstein](https://arxiv.org/abs/2309.11713)


Details of the model architecture and experimental results can be found in our papers.

```
@article{nguyen2024quasi,
  title={Quasi-Monte Carlo for 3D Sliced Wasserstein},
  author={Khai Nguyen and Nicola Bariletto and Nhat Ho},
  booktitle={International Conference on Learning Representations},
  year={2024},
  pdf={https://arxiv.org/pdf/2309.11713.pdf}
}
```
Please CITE our paper whenever this repository is used to help produce published results or incorporated into other software.

This implementation is made by [Khai Nguyen](https://khainb.github.io).

## Requirements
To install the required python packages, run
```
pip install -r requirements.txt
```

## What is included?
* Point-Cloud Gradient flow 
* Color Transfer
* Deep Point-Cloud Reconstruction


## Point-Cloud Gradient flow 
```
cd GradientFlow
python main_point.py
```

## Color Transfer

```
cd ColorTransfer
python main.py --source [source image] --target [target image] --num_iter 1000 --cluster

```

## Deep Point-cloud Reconstruction
Please read the README file in the PointcloudAE folder.

## Acknowledgment
The structure of this repo is largely based on [PointSWD](https://github.com/VinAIResearch/PointSWD). The structure of folder `render` is largely based on [Mitsuba2PointCloudRenderer](https://github.com/tolgabirdal/Mitsuba2PointCloudRenderer).