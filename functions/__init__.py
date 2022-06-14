import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def get_g_optimizer(config, parameters):
    if config.optim_g.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim_g.lr,
                          betas=[0,0.99],
                          eps=config.optim_g.eps)
    elif config.optim_g.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim_g.lr, weight_decay=config.optim_g.weight_decay)
    elif config.optim_g.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim_g.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim_g.optimizer))

def get_d_optimizer(config, parameters):
    if config.optim_d.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim_d.lr,
                          betas=[0,0.99],
                          eps=config.optim_d.eps)
    elif config.optim_d.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim_d.lr, weight_decay=config.optim_d.weight_decay)
    elif config.optim_d.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim_d.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim_d.optimizer))