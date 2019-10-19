import torchvisions.models as pyModels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_vggModel(vggConfig='A', batch_normalization=True)

    if vggConfig == 'A':
        if batch_normalization:
            return pyModels.vgg11_bn()
        else:
            return pyModels.vgg11()
    elif vggConfig = 'B':
        if batch_normalization:
            return pyModels.vgg13_bn()
        else:
            return pyModels.vgg13()
    elif vggConfig == 'D':
        if batch_normalization:
            return pyModels.vgg16_bn()
        else:
            return pyModels.vgg16()
    elif vggConfig == 'E':
        if batch_normalization:
            return pyModels.vgg19_bn()
        else:
            return pyModels.vgg19()

    return None


def get_resnet(depth=18):

    if depth == 18:
        return pyModels.resnet18()
    elif depth == 34:
        return pyModels.resnet34()

    return None













        
