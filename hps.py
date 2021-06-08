from types import SimpleNamespace

"""
    -- VQ-VAE-2 Hyperparameters --
"""
_common = {
    'checkpoint_frequency':         5,
    'image_frequency':              1,
    'test_size':                    0.1,
    'nb_workers':                   8,
}

_ffhq1024 = {
    'display_name':             'FFHQ1024',
    'image_shape':              (3, 1024, 1024),

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
    'batch_size':               8,
    'mini_batch_size':          8,
    'max_epochs':               100,
}

_ffhq1024_large = {
    'display_name':             'FFHQ1024 VQ-VAE++',
    'image_shape':              (3, 1024, 1024),

    'in_channels':              3,
    'hidden_channels':          128,
    'res_channels':             32,
    'nb_res_layers':            2,
    'embed_dim':                64,
    'nb_entries':               512,
    'nb_levels':                5,
    'scaling_rates':            [4, 2, 2, 2, 2],

    'learning_rate':            1e-4,
    'beta':                     0.25,
    'batch_size':               16,
    'mini_batch_size':          8,
    'max_epochs':               100,
}

_ffhq256 = {
    'display_name':             'FFHQ256',
    'image_shape':              (3, 256, 256),

    'in_channels':              3,
    'hidden_channels':          128,
    'res_channels':             32,
    'nb_res_layers':            2,
    'embed_dim':                64,
    'nb_entries':               512,
    'nb_levels':                2,
    'scaling_rates':            [4, 2],

    'learning_rate':            3e-3,
    'beta':                     0.25,
    'batch_size':               4096,
    'mini_batch_size':          32,
    'max_epochs':               100,
}
_ffhq128 = {
    'display_name':             'FFHQ128',
    'image_shape':              (3, 128, 128),

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
    'mini_batch_size':          128,
    'max_epochs':               100,
}

_cifar10 = {
    'display_name':             'CIFAR10',
    'image_shape':              (3, 32, 32),

    'in_channels':              3,
    'hidden_channels':          128,
    'res_channels':             32,
    'nb_res_layers':            2,
    'embed_dim':                64,
    'nb_entries':               512,
    'nb_levels':                2,
    'scaling_rates':            [2, 2],

    'learning_rate':            1e-3,
    'beta':                     0.25,
    'batch_size':               128,
    'mini_batch_size':          128,
    'max_epochs':               100,
}

_mnist = {
    'display_name':             'MNIST',
    'image_shape':              (1, 28, 28),

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
    'mini_batch_size':          32,
    'max_epochs':               100,
}

_kmnist = {
    'display_name':             'Kuzushiji-MNIST',
    'image_shape':              (1, 28, 28),

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
    'mini_batch_size':          32,
    'max_epochs':               100,
}

HPS_VQVAE = {
    'ffhq1024':             SimpleNamespace(**(_common | _ffhq1024)),
    'ffhq1024-large':       SimpleNamespace(**(_common | _ffhq1024_large)),
    'ffhq256':              SimpleNamespace(**(_common | _ffhq256)),
    'ffhq128':              SimpleNamespace(**(_common | _ffhq128)),
    'cifar10':              SimpleNamespace(**(_common | _cifar10)),
    'mnist':                SimpleNamespace(**(_common | _mnist)),
    'kmnist':               SimpleNamespace(**(_common | _kmnist)),
}

"""
    -- PixelSnail Hyperparameters --
"""

_common = {
    'checkpoint_frequency':     5,
    'image_frequency':          1,
    'nb_workers':               8,
}

_cifar10 = {
    'display_name':                 'CIFAR10',

    'batch_size':                   512,
    'mini_batch_size':              128,
    'learning_rate':                1e-4,
    'max_epochs':                   100,

    'scaling_rates':                [2, 2],
    'nb_entries':                   512,

    'level': [
        SimpleNamespace(**{
            'channel':              256,
            'kernel_size':          5,
            'nb_block':             4,
            'nb_res_block':         4,
            'nb_res_channel':       256,
            'attention':            True,
            'dropout':              0.1,
            
            'nb_cond_res_block':    3,
            'nb_cond_res_channel':  256,

            'nb_out_res_block':     0,
        }),
        SimpleNamespace(**{
            'channel':              256,
            'kernel_size':          5,
            'nb_block':             4,
            'nb_res_block':         4,
            'nb_res_channel':       256,
            'attention':            True,
            'dropout':              0.1,

            'nb_out_res_block':     0,
        }),
    ]
}

_ffhq256 = {
    'display_name':                 'FFHQ256',

    'batch_size':                   256,
    'mini_batch_size':              64,
    'learning_rate':                1e-4,
    'max_epochs':                   100,

    'scaling_rates':                [2, 2],
    'nb_entries':                   512,

    'level': [
        SimpleNamespace(**{
            'channel':              256,
            'kernel_size':          5,
            'nb_block':             4,
            'nb_res_block':         4,
            'nb_res_channel':       256,
            'attention':            True,
            'dropout':              0.1,
            
            'nb_cond_res_block':    3,
            'nb_cond_res_channel':  256,

            'nb_out_res_block':     0,
        }),
        SimpleNamespace(**{
            'channel':              256,
            'kernel_size':          5,
            'nb_block':             4,
            'nb_res_block':         4,
            'nb_res_channel':       256,
            'attention':            True,
            'dropout':              0.1,

            'nb_out_res_block':     0,
        }),
    ]
}

_ffhq1024 = {
    'display_name':                 'FFHQ1024',

    'batch_size':                   32,
    'mini_batch_size':              4,
    'learning_rate':                1e-4,
    'max_epochs':                   100,

    'scaling_rates':                [8, 2, 2],
    'nb_entries':                   512,

    'level': [
        SimpleNamespace(**{
            'channel':              256,
            'kernel_size':          5,
            'nb_block':             4,
            'nb_res_block':         4,
            'nb_res_channel':       256,
            'attention':            False, 
            'dropout':              0.1,
            
            'nb_cond_res_block':    3,
            'nb_cond_res_channel':  256,

            'nb_out_res_block':     0,
        }),
        SimpleNamespace(**{
            'channel':              256,
            'kernel_size':          5,
            'nb_block':             4,
            'nb_res_block':         4,
            'nb_res_channel':       256,
            'attention':            False,
            'dropout':              0.1,

            'nb_cond_res_block':    3,
            'nb_cond_res_channel':  256,

            'nb_out_res_block':     0,
        }),
        SimpleNamespace(**{
            'channel':              256,
            'kernel_size':          5,
            'nb_block':             4,
            'nb_res_block':         4,
            'nb_res_channel':       256,
            'attention':            True,
            'dropout':              0.1,

            'nb_out_res_block':     0,
        }),
    ]
}

HPS_PIXEL = {
    'cifar10':          SimpleNamespace(**(_common | _cifar10)),
    'ffhq256':          SimpleNamespace(**(_common | _ffhq256)),
    'ffhq1024':         SimpleNamespace(**(_common | _ffhq1024)),
}
