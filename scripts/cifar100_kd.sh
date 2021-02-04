# python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch wrn_16_2 --t-arch wrn_40_2 --t-path ./experiments/cifar100/teacher_wrn_40_2_seed0 --seed 1 --gpu-id 5 --print_freq 100

# python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch wrn_40_1 --t-arch wrn_40_2 --t-path ./experiments/cifar100/teacher_wrn_40_2_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch resnet20 --t-arch resnet56 --t-path ./experiments/cifar100/teacher_resnet56_seed0 --seed 1 --gpu-id 5 --print_freq 100


# python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch resnet20 --t-arch resnet110 --t-path ./experiments/cifar100/teacher_resnet110_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch resnet32 --t-arch resnet110 --t-path ./experiments/cifar100/teacher_resnet110_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch resnet8x4 --t-arch resnet32x4 --t-path ./experiments/cifar100/teacher_resnet32x4_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch vgg8 --t-arch vgg13 --t-path ./experiments/cifar100/teacher_vgg13_seed0 --seed 1 --gpu-id 5 --print_freq 100

# python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch MobileNetV2 --t-arch vgg13 --t-path ./experiments/cifar100/teacher_vgg13_seed0 --seed 1 --gpu-id 5 --print_freq 100

# python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch MobileNetV2 --t-arch ResNet50 --t-path ./experiments/cifar100/teacher_ResNet50_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch vgg8 --t-arch ResNet50 --t-path ./experiments/cifar100/teacher_ResNet50_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch ShuffleV1 --t-arch resnet32x4 --t-path ./experiments/cifar100/teacher_resnet32x4_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch ShuffleV2 --t-arch resnet32x4 --t-path ./experiments/cifar100/teacher_resnet32x4_seed0 --seed 1 --gpu-id 5 --print_freq 100

python student.py --kd_func kd --div_weight 0.8 --ce_weight 0.2 --s-arch ShuffleV2 --t-arch wrn_40_2 --t-path ./experiments/cifar100/teacher_wrn_40_2_seed0 --seed 1 --gpu-id 5 --print_freq 100

