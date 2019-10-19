import torch.optim as optimizers


def get_SGD(lr=0.01, momentum=0, dampening=0, weight_decay=0):
    
    return optimizers.SGD(lr=lr, momentum=0, dampening=0, weight_decay=0)

