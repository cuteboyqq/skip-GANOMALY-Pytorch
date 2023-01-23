### skip-GANomaly-Pytorch-CIFAR10-MNIST-CUSTOM
[(Back to menu)](#table-of-contents)

Use skip-connection  to skip Encoder layer to Decoder layer by concatenation, the framework is based on gamonaly.
Impement skip-ganomaly and skip-attention-ganomaly, here use CBAM attention before skip Encoder layer to Decoder 

### updated
[(Back to menu)](#table-of-contents)
1. Able to train CIFAR10, MNIST datasets [2023-01-23 updated]


Generator +  Discriminator model 


### Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [skip-GANomaly-Pytorch-CIFAR10-MNIST-CUSTOM](#skip-GANomaly-Pytorch-CIFAR10-MNIST-CUSTOM)
- [updated](#updated)
- [Requirement](#Requirement)
- [implement](#implement)
   - [Unet-Network](#Unet-Network)
   - [Unet-CBAM-Network](#Unet-CBAM-Network)
- [Train-on-custom-dataset](#Train-on-custom-dataset)
- [Train](#Train)
- [Test](#Test)
- [Lose-value-distribution](#Lose-value-distribution)
- [Reference](#Reference)
   
### Requirement
```
pip install -r requirements.txt
```

### implement 
[(Back to top)](#table-of-contents)

1. Encoder-Decoder use Unet  (image reference from paper "Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection")


![Ganomaly](https://user-images.githubusercontent.com/58428559/210389653-27f8b7dd-bd35-470b-908c-ebf7bd92b7ca.png)

### Unet-Network
[(Back to top)](#table-of-contents)

![Unet](https://user-images.githubusercontent.com/58428559/210389166-bee0d5e5-1810-41af-8628-3fd4907e3aa8.png)



### Unet-CBAM-Network
[(Back to top)](#table-of-contents) (Ref from:U-Net: Convolutional Networks for Biomedical Image Segmentation )

![CBAM](https://user-images.githubusercontent.com/58428559/210389295-6d2eb925-396e-4706-8ae0-dcd75de82531.png)


### Train-on-custom-dataset
[(Back to top)](#table-of-contents)

```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png


```

### Train
[(Back to top)](#table-of-contents)
```
python train.py --img-dir "[train dataset dir] or cifar10 or mnist" 
                  --batch-size 64 
                  --img-size 32 
                  --epoch 20 
                  --model "ganomaly or skip-ganomaly or skip-attention-ganomly" 
                  --abnormal-class "airplane" 
```

### Test
[(Back to top)](#table-of-contents)
```
python test.py --nomal-dir "[test normal dataset dir]" 
               --abnormal-dir "[test abnormal dataset dir]" 
               --view-img 
               --img-size 32
```




### Reference 
[(Back to top)](#table-of-contents)

GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training

https://arxiv.org/abs/1805.06725

Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection

https://arxiv.org/pdf/1901.08954.pdf

