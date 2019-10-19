import torch.optim as optimizers


def get_SGD():
    
    return optimizers.SGD(lr=lr, momentum=0, dampening=0, weight_decay=0)


def get_adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):

    return optimizers.Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

