# MAPE: Defending Against Transfer-Based Black-Box Attacks Using Multi-Source Adversarial Perturbations Elimination

[//]: # ([Paper]&#40;&#41; )

> **Abstract:** *Neural networks are susceptible to carefully crafted adversarial examples, leading to high-confidence 
> incorrect judgments. Due to the stealthiness and difficulty in detection, transfer-based black-box attacks have become
> a focus of adversarial defense. Previous methods for high-order denoising were limited to specific target models, 
> exhibiting poor generalization when countering adversarial attacks from unseen substitute models. Herein, we propose 
> a deep learning training framework called multi-source adversarial perturbations elimination (MAPE) to defend against 
> transfer-based attacks. MAPE consists of multiple parallel single-source adversarial perturbation elimination 
> mechanisms (SAPE), creating adversarial examples for training in a randomized and multimodal manner. Using a 
> meticulously designed self-attention U-Net (SAU-Net) as the extractor of perturbations elimination, MAPE enhances 
> the generalization of SAU-Net through diverse updates of label loss differences. Evaluations on CIFAR and 
> Mini-ImageNet demonstrate that MAPE showcases high generalization and robustness. The MAPE-based defense system 
> constructed with SAU-Net and the target model can extract and eliminate adversarial perturbations from various 
> adversarial examples, effectively defending against unseen adversarial attacks from different substitute models.*
>

## Installation

```
conda create -n MAPE python=3.11.9
conda activate MAPE
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
PyTorch is not version-sensitive. The project can typically run on other versions of PyTorch as well. 
Furthermore, allow the system to automatically select the version when installing any other missing libraries.

## Restoring Files Structure

You can restore the files structure with `restore_file_structure.py`.

  `python restore_file_structure.py`

## Pre-Training Classifiers

You can train various classifiers with `train_classifier.py`.

* For GoogLeNet on CIFAR-10
  `python train_classifier.py --classifier_name GoogLeNet --dataset_name cifar10`
* For ResNet34 on CIFAR-10
  `python train_classifier.py --classifier_name ResNet34 --dataset_name cifar10`
* For VGG19 on CIFAR-10
  `python train_classifier.py --classifier_name VGG19 --dataset_name cifar10`
* For MobileNetV2 on CIFAR-10
  `python train_classifier.py --classifier_name MobileNetV2 --dataset_name cifar10`


## Generating Adversarial Examples

* For MobileNetV2 on CIFAR-10
  `python generate_adversarial_example.py --classifier_name MobileNetV2 --classifier_path Classifier/cifar10/MobileNetV2_0.9372.pt --dataset_name cifar10`


## Training SAPE

* For SAU-Net-5 on CIFAR-10. After modifying utils/classifiers_path.py, run the following command.
  `python train_sape.py`


## Training MAPE

* For SAU-Net-5 on CIFAR-10. After modifying utils/classifiers_path.py, run the following command.
  `python train_mape.py`


## Evaluation

* Unzip the pre-trained SAU-Net-5.
`python unzip.py`
* Defending FGSM and BIM for MobileNetV2 with MAPE on CIFAR-10.
`python evaluate.py --attack_methods fgsm_pgd --sau_net_load_path SAU_Net/cifar10/SAU_Net_MAPE_0.9594.pt --dataset_name cifar10 --classifier_name MobileNetV2`
