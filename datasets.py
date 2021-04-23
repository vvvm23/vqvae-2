import torch
import torchvision

def get_dataset(task: str, cfg):
    if task == 'ffhq1024':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        nb_test = int(len(dataset) * cfg.test_size)
        nb_train = len(dataset) - nb_test
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [nb_train, nb_test])
    elif task == 'ffhq256':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        nb_test = int(len(dataset) * cfg.test_size)
        nb_train = len(dataset) - nb_test
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [nb_train, nb_test])
    elif task == 'ffhq128':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.ImageFolder('data/ffhq1024', transform=transforms)
        nb_test = int(len(dataset) * cfg.test_size)
        nb_train = len(dataset) - nb_test
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [nb_train, nb_test])
    elif task == 'cifar10':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transforms, download=True)
    elif task == 'mnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.MNIST('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.MNIST('data', train=False, transform=transforms, download=True)
    elif task == 'kmnist':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.KMNIST('data', train=True, transform=transforms, download=True)
        test_dataset = torchvision.datasets.KMNIST('data', train=False, transform=transforms, download=True)
    else:
        print("> Unknown dataset. Terminating")
        exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.nb_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.nb_workers, shuffle=False)

    return train_loader, test_loader
