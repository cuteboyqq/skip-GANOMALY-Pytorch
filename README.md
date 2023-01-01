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

![image](https://user-images.githubusercontent.com/58428559/210168476-2cb1d156-d373-4bcc-84f4-89ef64679728.png)



### Test
[(Back to top)](#table-of-contents)
```
python test.py --nomal-dir "[test normal dataset dir]" --abnormal-dir "[test abnormal dataset dir]" --view-img --img-size 32
```
Example :
Train dataset : factory line only


dataset :factory line , top: input images, bottom: reconstruct images, avg_batch_normal_loss=0.99565

![infer_normal_loss_0 95653355_2](https://user-images.githubusercontent.com/58428559/210168376-8a2c1f61-85c4-4372-b93a-2506e3e593be.jpg)


dataset :factory noline , top: input images, bottom: reconstruct images, avg_batch_anomaly_loss = 6.2653

![infer_abnormal_loss_6 2653203_4](https://user-images.githubusercontent.com/58428559/210168379-a5db38ca-5d5d-4f74-973e-910365c40ab3.jpg)



### Lose-value-distribution
[(Back to top)](#table-of-contents)

Blue : normal dataset

Orange : abnormal dataset


![image](https://user-images.githubusercontent.com/58428559/210168535-682bd748-df50-4935-a2ef-f67ad9e3a313.png)

![image](https://user-images.githubusercontent.com/58428559/210168526-8c657772-35b3-4d9d-a8ee-cf81aadc919d.png)

![image](https://user-images.githubusercontent.com/58428559/210168496-76ade09d-28a1-4900-b68b-65be4d80496e.png)




### Reference 
[(Back to top)](#table-of-contents)

https://arxiv.org/abs/1805.06725

Skip-GANomaly: Skip Connected and Adversarially Trained Encoder-Decoder Anomaly Detection

https://arxiv.org/pdf/1901.08954.pdf

