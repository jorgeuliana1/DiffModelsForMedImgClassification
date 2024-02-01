# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Function to load the CNN models
"""

import timm

# from effnet import MyEffnet
# from densenet import MyDensenet
from mobilenet import MyMobilenet

# from resnet import MyResnet
# from vggnet import MyVGGNet
from torchvision import models

# from efficientnet_pytorch import EfficientNet


_MODELS = [
    "resnet-50",
    "resnet-101",
    "densenet-121",
    "vgg-13",
    "vgg-16",
    "vgg-19",
    "mobilenet",
    "efficientnet-b4",
]

_NORM_AND_SIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [224, 224]]


def get_norm_and_size(model_name):
    if model_name == "inceptionv4":
        return [229, 229], ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif model_name in ["beit", "vit"]:
        return [224, 224], ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif model_name in [
        "resnet-50",
        "resnet-101",
        "densenet-121",
        "googlenet",
        "vgg-13",
        "vgg-16",
        "vgg-19",
        "mobilenet",
        "efficientnet-b4",
        "senet",
        "vit",
        "deit",
    ]:
        return [224, 224], ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        # Load pre defined parameters for timm models
        pre_load_model = timm.create_model(model_name, pretrained=True, num_classes=0)
        size = list(pre_load_model.default_cfg["input_size"][1:])
        mean_std = (
            list(pre_load_model.default_cfg["mean"]),
            list(pre_load_model.default_cfg["std"]),
        )
        del pre_load_model
        return size, mean_std
        # return [384, 384], ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def set_model(
    model_name,
    num_class,
    neurons_reducer_block=0,
    comb_method=None,
    comb_config=None,
    pretrained=True,
    freeze_conv=False,
):

    if pretrained:
        pre_torch = True
    else:
        pre_torch = False

    if model_name not in _MODELS:
        raise Exception("The model {} is not available!".format(model_name))

    model = None
    # if model_name == 'resnet-50':
    #     model = MyResnet(models.resnet50(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
    #                      comb_method=comb_method, comb_config=comb_config)

    # elif model_name == 'resnet-101':
    #     model = MyResnet(models.resnet101(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
    #                      comb_method=comb_method, comb_config=comb_config)

    # elif model_name == 'densenet-121':
    #     model = MyDensenet(models.densenet121(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
    #                      comb_method=comb_method, comb_config=comb_config)

    # elif model_name == 'vgg-13':
    #     model = MyVGGNet(models.vgg13_bn(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
    #                      comb_method=comb_method, comb_config=comb_config)

    # elif model_name == 'vgg-16':
    #     model = MyVGGNet(models.vgg16_bn(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
    #                      comb_method=comb_method, comb_config=comb_config)

    # elif model_name == 'vgg-19':
    #     model = MyVGGNet(models.vgg19_bn(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
    #                      comb_method=comb_method, comb_config=comb_config)

    # elif model_name == 'mobilenet':
    if model_name == "mobilenet":
        model = MyMobilenet(
            models.mobilenet_v2(pretrained=pre_torch),
            num_class,
            neurons_reducer_block,
            freeze_conv,
            comb_method=comb_method,
            comb_config=comb_config,
        )

    # elif model_name == 'efficientnet-b4':
    #     if pretrained:
    #         model = MyEffnet(EfficientNet.from_pretrained(model_name), num_class, neurons_reducer_block, freeze_conv,
    #                          comb_method=comb_method, comb_config=comb_config)
    #     else:
    #         model = MyEffnet(EfficientNet.from_name(model_name), num_class, neurons_reducer_block, freeze_conv,
    #                          comb_method=comb_method, comb_config=comb_config)

    return model
