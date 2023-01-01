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
![infer_normal_loss_0 95653355_2](https://user-images.githubusercontent.com/58428559/210168376-8a2c1f61-85c4-4372-b93a-2506e3e593be.jpg)


dataset :factory noline , top: input images, bottom: reconstruct images

![infer_abnormal_loss_6 2653203_4](https://user-images.githubusercontent.com/58428559/210168379-a5db38ca-5d5d-4f74-973e-910365c40ab3.jpg)



### Lose-value-distribution
[(Back to top)](#table-of-contents)

Blue : normal dataset

Orange : abnormal dataset
![loss_distribution](https://user-images.githubusercontent.com/58428559/210168369-66b80327-7dbd-44a2-a393-b267a289ca09.jpg)



![2023-01-01-skip-ganomaly-histogram](https://user-images.githubusercontent.com/58428559/210168360-f15e1bd3-8f0f-452a-9f96-07406e2c4785.jpg)





### Reference 
[(Back to top)](#table-of-contents)

https://arxiv.org/abs/1805.06725

https://arxiv.org/pdf/1901.08954.pdf

