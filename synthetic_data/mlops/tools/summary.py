from typing import Callable, Type

import torch
from torch import nn, randint, randn
from torchinfo import summary


def summarize_gan(cls_gen: Type[nn.Module], cls_dis: Type[nn.Module]) -> None:

    BS = 128
    Z_DIM = 100
    SEQ_LENGTH = 1024

    gen = cls_gen(SEQ_LENGTH, Z_DIM)
    dis = cls_dis(SEQ_LENGTH)

    input_sizeG = BS, Z_DIM
    input_sizeD = BS, SEQ_LENGTH

    summary(gen, input_size=input_sizeG)
    summary(dis, input_size=input_sizeD)


def summarize_conditional_gan(
    cls_gen: Type[nn.Module], cls_dis: Type[nn.Module], n_classes: int
) -> None:

    BS = 128
    Z_DIM = 100
    SEQ_LENGTH = 1024

    gen = cls_gen(SEQ_LENGTH, n_classes, Z_DIM)
    dis = cls_dis(SEQ_LENGTH, n_classes)

    dataG = randn(BS, Z_DIM)
    dataD = randn(BS, SEQ_LENGTH)

    input_dataG = dataG, randint(0, n_classes, (BS,))
    input_dataD = dataD, randint(0, n_classes, (BS,))

    summary(gen, input_data=input_dataG)  # , input_data=input_data)
    summary(dis, input_data=input_dataD)  # , input_data=input_data)


def summarize_lstm(cls_lstm: Type[nn.Module]) -> None:

    BS = 1
    SEQ_LENGTH = 1024
    LAYERS = 64

    lstm = cls_lstm(LAYERS).double()
    input_size = BS, SEQ_LENGTH

    summary(lstm, input_size=input_size, dtypes=[torch.double])
