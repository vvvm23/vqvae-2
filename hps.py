from types import SimpleNamespace

_common = {
    'checkpoint_frequency':         10,
    'image_frequency':              1,
    'test_size':                    0.1,
    'nb_workers':                   8,
}

_ffhq1024 = {
    'display_name':             'FFHQ1024',

    'in_channels':              3,
    'hidden_channels':          128,
    'res_channels':             32,
    'nb_res_layers':            2,
    'embed_dim':                64,
    'nb_entries':               512,
    'nb_levels':                3,
    'scaling_rates':            [8, 2, 2],

    'learning_rate':            1e-4,
    'beta':                     0.25,
    'batch_size':               4,
    'max_epochs':               100,
}

_cifar10 = {
    'display_name':             'CIFAR10',

    'in_channels':              3,
    'hidden_channels':          128,
    'res_channels':             32,
    'nb_res_layers':            2,
    'embed_dim':                64,
    'nb_entries':               512,
    'nb_levels':                2,
    'scaling_rates':            [4, 2],

    'learning_rate':            1e-4,
    'beta':                     0.25,
    'batch_size':               128,
    'max_epochs':               100,
}

_mnist = {
    'display_name':             'MNIST',

    'in_channels':              1,
    'hidden_channels':          128,
    'res_channels':             32,
    'nb_res_layers':            2,
    'embed_dim':                32,
    'nb_entries':               128,
    'nb_levels':                2,
    'scaling_rates':            [2, 2],

    'learning_rate':            1e-4,
    'beta':                     0.25,
    'batch_size':               32,
    'max_epochs':               100,
}

HPS = {
    'ffhq1024': SimpleNamespace(**(_common | _ffhq1024)),
    'cifar10': SimpleNamespace(**(_common | _cifar10)),
    'mnist': SimpleNamespace(**(_common | _mnist)),
}
