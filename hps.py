from types import SimpleNamespace

HPS = {
    'ffhq1024': SimpleNamespace(**{
        'in_channels':              3,
        'hidden_channels':          128,
        'res_channels':             32,
        'nb_res_layers':            2,
        'nb_levels':                3,
        'embed_dim':                64,
        'nb_entries':               512,
        'scaling_rates':            [4, 4, 2],

        'learning_rate':            1e-4,
        'beta':                     0.25,
        'batch_size':               4,
        'max_epochs':               100,
        'checkpoint_frequency':     10,
    })
}
