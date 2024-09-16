# MAPE: Defending Against Transfer-Based Black-Box Attacks Using Multi-Source Adversarial Perturbations Elimination

[//]: # ([Paper]&#40;&#41; )

> **Abstract:** *Neural networks are vulnerable to meticulously crafted adversarial examples, resulting in high-confidence misclassifications in image classification tasks. Due to their stealthiness and difficulty in detection, transfer-based black-box attacks have become a significant focus of defense. In this work, we propose a deep learning training framework called multi-source adversarial perturbations elimination (MAPE) to defend against diverse transfer-based attacks. MAPE consists of the single-source adversarial perturbation elimination (SAPE) training mechanism and the pre-trained models probabilistic scheduling algorithm (PPSA). SAPE employs a thoughtfully designed channel-attention U-Net as an extractor and generates adversarial examples using a pre-trained model (e.g., ResNet). These adversarial examples are used to train the extractor, enhancing its ability to extract and eliminate adversarial perturbations. PPSA extends the number of pre-trained models from one to several, introducing model difference probability and negative momentum probability based on their output scores and scheduling records, respectively. By combining these two probabilities, PPSA can dynamically schedule the pre-trained models utilized in SAPE to maximize the differences between them across adjacent training cycles, thereby enhancing the extractor's generalization in addressing adversarial perturbations. The MAPE-based defense system effectively eliminates adversarial perturbations from various adversarial examples, providing robust defense against unseen adversarial attacks from different substitute models. In a black-box attack scenario, utilizing ResNet-34 as the target model, our approach achieves average defense rates exceeding 95% on the CIFAR-10 dataset and over 70% on the Mini-ImageNet dataset, demonstrating state-of-the-art performance.*
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
