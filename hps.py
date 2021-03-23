from types import SimpleNamespace

HPS = {
    'ffhq1024': SimpleNamespace(**{
        'learning_rate':            1e-4,
        'beta':                     0.25,
        'batch_size':               4,
        'max_epochs':               100,
        'checkpoint_frequency':     10,
    })
}
