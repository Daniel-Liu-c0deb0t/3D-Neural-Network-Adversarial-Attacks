# 3D Point Cloud Adversarial Attacks and Defenses
Adversarial attacks and defenses on neural networks that process 3D point cloud data, namely PointNet and PointNet++. The preprint paper is available on Arxiv [**here**](https://arxiv.org/abs/1901.03006). A shortened version is accepted at the 2019 IEEE ICIP. If you use this code, please cite

```
@article{liu2019extending,
  title={Extending Adversarial Attacks and Defenses to Deep 3D Point Cloud Classifiers},
  author={Liu, Daniel and Yu, Ronald and Su, Hao},
  journal={arXiv preprint arXiv:1901.03006},
  year={2019}
}
```

A continuation of this research is in [this paper](https://arxiv.org/abs/1908.06062), and its code is available [here](https://github.com/Daniel-Liu-c0deb0t/Adversarial-point-perturbations-on-3D-objects).

Note that files modified from the PointNet and PointNet++ source codes are included. Some files may need to be moved to the correct location before running experiments. Clone the PointNet and PointNet++ repositories, train the models, and place the code in `src/` into each of those repositories. The files `src/adversarial_pointnet.py` and `src/adversarial_pointnet2.py` should be ran for attacks and defenses. Get the `.npz` file required for gradient projection attacks from [here](https://github.com/Daniel-Liu-c0deb0t/Adversarial-point-perturbations-on-3D-objects/releases/download/Data/point_clouds.npz).

## Highlights
### Attacks
- Fast/iterative gradient sign
- Jacobian-based saliency map attack
- Gradient projection
- Clipping L2 norms
### Defenses
- Adversarial training
- Outlier removal
- Salient point removal
### Conclusions
- Adversarial attacks are effective against deep 3D point cloud classifiers
- It is more easy to defend point cloud classifiers than 2D image classifiers
