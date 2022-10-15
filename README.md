### skip-GANomaly-Pytorch
[(Back to top)](#table-of-contents)

Generator +  Discriminator model 


### Table of contents

<!-- After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README. -->

- [skip-GANomaly-Pytorch](#skip-GANomaly-Pytorch)
- [Requirement](#Requirement)
- [implement](#implement)
- [Unet-Network](#Unet-Network)
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

1. Encoder-Decoder use Unet

![image](https://user-images.githubusercontent.com/58428559/195968483-e7b102f1-6071-4e70-8f18-0c4b749eda30.png)


### Unet-Network
[(Back to top)](#table-of-contents)
![image](https://user-images.githubusercontent.com/58428559/195968671-a287ecae-67b0-41e2-9bfc-7283014c8c3b.png)

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
python train.py --img-dir "[train dataset dir]" --batch-size 64 --img-size 32 --epoch 20
```
### Test
[(Back to top)](#table-of-contents)
```
python test.py --nomal-dir "[test normal dataset dir]" --abnormal-dir "[test abnormal dataset dir]" --view-img --img-size 32
```
Example :
Train dataset : factory line only

dataset :factory line , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/195970559-8acd1c96-fc0b-4ba8-81a4-6faf56bff6d2.png)


dataset :factory noline , top: input images, bottom: reconstruct images
![image](https://user-images.githubusercontent.com/58428559/195970601-3a851ecf-5169-4992-8f80-d78c76a618c7.png)


### Lose-value-distribution
[(Back to top)](#table-of-contents)

Blue : normal dataset

Orange : abnormal dataset


![image](https://user-images.githubusercontent.com/58428559/195972746-dfd5328f-58f8-4c48-81d4-7510011f9528.png)


### Reference 
[(Back to top)](#table-of-contents)

https://arxiv.org/abs/1805.06725

https://arxiv.org/pdf/1901.08954.pdf

