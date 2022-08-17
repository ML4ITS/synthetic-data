# Models


## CGAN
```python
Layer (type:depth-idx)                   Output Shape              Param #
---------------------------------------------------------------------------

Generator                                [128, 1024]               --
├─Embedding: 1-1                         [128, 10]                 100
├─Sequential: 1-2                        [128, 1024]               --
│    └─Linear: 2-1                       [128, 128]                14,208
│    └─LeakyReLU: 2-2                    [128, 128]                --
│    └─Linear: 2-3                       [128, 256]                32,768
│    └─BatchNorm1d: 2-4                  [128, 256]                512
│    └─LeakyReLU: 2-5                    [128, 256]                --
│    └─Linear: 2-6                       [128, 512]                131,072
│    └─BatchNorm1d: 2-7                  [128, 512]                1,024
│    └─LeakyReLU: 2-8                    [128, 512]                --
│    └─Linear: 2-9                       [128, 1024]               525,312
Total params: 704,996
Trainable params: 704,996
Non-trainable params: 0
Total mult-adds (M): 90.24

---------------------------------------------------------------------------
Discriminator                            [128]                     --
├─Embedding: 1-1                         [128, 10]                 100
├─Sequential: 1-2                        [128, 1]                  --
│    └─Linear: 2-1                       [128, 512]                529,920
│    └─LeakyReLU: 2-2                    [128, 512]                --
│    └─Linear: 2-3                       [128, 512]                262,656
│    └─Dropout: 2-4                      [128, 512]                --
│    └─LeakyReLU: 2-5                    [128, 512]                --
│    └─Linear: 2-6                       [128, 512]                262,656
│    └─Dropout: 2-7                      [128, 512]                --
│    └─LeakyReLU: 2-8                    [128, 512]                --
│    └─Linear: 2-9                       [128, 1]                  513
Total params: 1,055,845
Trainable params: 1,055,845
Non-trainable params: 0
Total mult-adds (M): 135.15
```


## WGAN-GP
```python
Layer (type:depth-idx)                   Output Shape              Param #
---------------------------------------------------------------------------

Generator                                [128, 1024]               --****
├─Sequential: 1-1                        [128, 1024]               --
│    └─Linear: 2-1                       [128, 128]                12,928
│    └─LeakyReLU: 2-2                    [128, 128]                --
│    └─Linear: 2-3                       [128, 256]                32,768
│    └─BatchNorm1d: 2-4                  [128, 256]                512
│    └─LeakyReLU: 2-5                    [128, 256]                --
│    └─Linear: 2-6                       [128, 512]                131,072
│    └─BatchNorm1d: 2-7                  [128, 512]                1,024
│    └─LeakyReLU: 2-8                    [128, 512]                --
│    └─Linear: 2-9                       [128, 1024]               524,288
│    └─BatchNorm1d: 2-10                 [128, 1024]               2,048
│    └─LeakyReLU: 2-11                   [128, 1024]               --
│    └─Linear: 2-12                      [128, 1024]               1,049,600
Total params: 1,754,240
Trainable params: 1,754,240
Non-trainable params: 0
Total mult-adds (M): 224.54

---------------------------------------------------------------------------
Discriminator                            [128, 1]                  --
├─Sequential: 1-1                        [128, 1]                  --
│    └─Linear: 2-1                       [128, 512]                524,800
│    └─LeakyReLU: 2-2                    [128, 512]                --
│    └─Linear: 2-3                       [128, 256]                131,328
│    └─LeakyReLU: 2-4                    [128, 256]                --
│    └─Linear: 2-5                       [128, 1]                  257
Total params: 656,385
Trainable params: 656,385
Non-trainable params: 0
Total mult-adds (M): 84.02
```