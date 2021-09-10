###  Multi-granularity for knowledge distillation ![]( https://visitor-badge.glitch.me/badge?page_id=multi_granularity_distillation)
🎉🎉🎉**Our paper has been accepted by IMAVIS!!!** [paper](https://www.sciencedirect.com/science/article/abs/pii/S0262885621001918)

#### Dependencies

- python3.6
- pytorch1.7
- tensorboard2.4

#### Training on CIFAR100

- First, train a teacher network

```shell
python teacher.py --arch [teacher]
```

- Then, construct multi-granularity knowledge

```shell
python train_teacher_wrapper.py --t-arch [teacher] --t-path [teacher-weight-path]
```

- Distill

Granularity-wise distillation

```shell
python student.py --kd_func [kd-function] --s-arch [student] --t-arch [teacher] --t-path [teacher-weight-path] 
```

Stable excitation distillation

```shell
python student_stable.py --kd_func [kd-function] --s-arch [student] --t-arch [teacher] --t-path [teacher-weight-path] 
```

#### Performance

- CIFAR100

|                | WRN-40-2/WRN-16-2 | WRN-40-2/WRN-40-1 | res56/res20  | res110/res20 | res110/res32 | resnet32x4/resnet8x4 |  vgg13/vgg8  |
| :------------: | :---------------: | :---------------: | :----------: | :----------: | :----------: | :------------------: | :----------: |
|      T/S       |    75.61/73.26    |    75.61/71.98    | 72.34/69.06  | 74.31/69.06  | 74.31/71.14  |     79.42/72.50      | 74.64/70.36  |
|       KD       |       73.59       |       73.58       |    71.05     |    70.90     |    73.34     |        73.27         |    73.18     |
|   **MAG+KD**   |       75.09       |       74.10       |    71.43     |    71.53     |    73.55     |        73.82         |    73.63     |
|   **MAS+KD**   |       75.35       |       74.42       |    71.08     |    71.03     |    73.54     |        74.20         | <u>74.18</u> |
|     FitNet     |       73.82       |       72.32       |    69.33     |    68.96     |    71.07     |        73.62         |    71.14     |
| **MAG+FitNet** |       75.30       |       73.99       |    70.29     |    70.31     |    72.73     |        74.88         |    73.06     |
| **MAS+FitNet** |       75.17       |       74.43       |    71.08     |    70.69     |    73.18     |     <u>75.76</u>     |    73.59     |
|       AT       |       74.39       |       72.82       |    70.39     |    70.36     |    72.60     |        73.53         |    71.41     |
|   **MAG+AT**   |       75.28       |       73.81       |    70.99     |    70.57     |    73.56     |        74.56         |    72.11     |
|   **MAS+AT**   |       75.98       |     **74.90**     | <u>71.78</u> |    71.34     |    73.29     |        74.92         |    73.38     |
|       SP       |       74.01       |       73.00       |    70.28     |    70.29     |    72.74     |        73.28         |    72.94     |
|   **MAG+SP**   |       74.30       |       73.71       |    71.13     |    70.79     |    73.44     |        73.58         |    73.20     |
|   **MAS+SP**   |       75.37       |       73.79       |    70.97     | <u>71.78</u> |    73.66     |        74.26         |    73.64     |
|      VID       |       74.19       |       73.23       |    70.53     |    70.68     |    72.67     |        73.24         |    71.41     |
|  **MAG+VID**   |       74.84       |       73.35       |    71.14     |    70.69     |    73.00     |        74.73         |    72.92     |
|  **MAS+VID**   |       75.63       |       74.49       |    71.28     |    71.61     |    73.32     |        74.86         |    73.56     |
|      RKD       |       73.37       |       72.10       |    69.67     |    69.44     |    72.24     |        72.03         |    71.35     |
|  **MAG+RKD**   |       75.73       |       73.59       |    71.51     |    71.11     |    73.71     |        74.23         |    73.44     |
|  **MAS+RKD**   |       75.31       |       74.30       |  **71.91**   |    71.06     |    73.17     |        74.39         |    73.06     |
|      CRD       |       75.52       |       74.24       |    71.38     |    71.34     |    73.55     |        75.32         |     73.9     |
|  **MAG+CRD**   |   <u>75.84</u>    |       74.53       |    71.77     |  **71.91**   | <u>74.00</u> |      **75.89**       |  **74.29**   |
|  **MAS+CRD**   |     **75.87**     |   <u>74.80</u>    |    71.52     |    71.52     |  **74.06**   |        75.41         |    74.06     |
|      AFD       |       75.41       |       73.66       |    71.32     |    71.20     |    73.46     |        74.72         |    73.57     |
|  **MAG+AFD**   |       75.53       |       74.53       |    71.62     |    71.40     |    73.57     |        74.75         |    73.89     |
|  **MAS+AFD**   |       75.55       |       74.12       |    71.49     |    71.22     | <u>74.00</u> |        75.03         |    73.62     |

- Market1501

| Setting |    Method    |  Backbone   | Rank1 |  mAP  |
| :-----: | :----------: | :---------: | :---: | :---: |
|    1    |   Vanilla    |  ResNet50   | 88.84 | 71.59 |
|    2    |   Vanilla    | DenseNet121 | 90.17 | 74.02 |
|    3    | Circle loss  | DenseNet121 | 91.00 | 76.54 |
|    4    |    HA-CNN    |  Inception  | 90.90 | 75.60 |
|    5    |     MLFN     |   ResNeXt   | 90.10 | 74.30 |
|    6    |     PCB      |  ResNet50   | 92.64 | 77.47 |
|    7    |  OSNetx0.75  |    OSNet    | 93.60 | 82.50 |
|    8    |  OSNetx1.0   |    OSNet    | 94.10 | 82.90 |
|    9    | MAS_RKD(T:2) |  ResNet50   | 91.09 | 79.43 |
|   10    | MAS_RKD(T:8) | OSNetx0.75  | 94.50 | 84.30 |

#### BibTex Cite
```
@article{shao2021multi,
  title={Multi-granularity for knowledge distillation},
  author={Shao, Baitan and Chen, Ying},
  journal={Image and Vision Computing},
  pages={104286},
  year={2021},
  publisher={Elsevier}
}
```

####  Acknowledgements

This repo is partly based on the following repos, thank the authors a lot.

- [HobbitLong/RepDistiller](https://github.com/HobbitLong/RepDistiller)
- [AberHu/Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)
- [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
