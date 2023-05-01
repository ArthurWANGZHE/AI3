import torch
from torch import nn
from torchvision import models
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
from datetime import datetime

def main():
    start_time = datetime.now()
    model = models.resnet18(pretrained=True, )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 100)

    classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale',
               'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
               'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
               'bottles', 'bowls', 'cans', 'cups', 'plates',
               'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
               'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
               'bed', 'chair', 'couch', 'table', 'wardrobe',
               'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
               'bear', 'leopard', 'lion', 'tiger', 'wolf',
               'bridge', 'castle', 'house', 'road', 'skyscraper',
               'cloud', 'forest', 'mountain', 'plain', 'sea',
               'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
               'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
               'crab', 'lobster', 'snail', 'spider', 'worm',
               'baby', 'boy', 'girl', 'man', 'woman',
               'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
               'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
               'maple', 'oak', 'palm', 'pine', 'willow',
               'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
               'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')

    def get_mean_and_std(dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for inputs, targets in tqdm(dataloader):
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        return mean, std
    num_classes = 100
    batch_size = 64
    num_epochs = 5
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = datasets.CIFAR100(root='../data/',
                                      download=True,
                                      train=True,
                                      transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR100(root='../data/',
                                     download=True,
                                     train=False,
                                     transform=transforms.ToTensor())
    get_mean_and_std(train_dataset)
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.4), std=(0.2, 0.2, 0.2))
    ])
    train_dataset = datasets.CIFAR100(root='../data/',
                                      download=True,
                                      train=True,
                                      transform=transform)
    test_dataset = datasets.CIFAR100(root='../data/',
                                     download=True,
                                     train=False,
                                     transform=transform)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  shuffle=False,
                                                  batch_size=batch_size)
    images, labels = next(iter(train_dataloader))
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 100)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    total_batch = len(train_dataloader)
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            out = model(images)
            loss = criterion(out, labels)

            n_corrects = (out.argmax(axis=1) == labels).sum().item()
            acc = n_corrects / labels.size(0)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                    f' {epoch + 1}/{num_epochs}, {batch_idx + 1}/{total_batch}: {loss.item():.4f}, acc: {acc}')

    class UnNormalize(object):

        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return tensor

    unnormalize = UnNormalize((0.5, 0.5, 0.4), (0.2, 0.2, 0.2))
    total = 0
    correct = 0


    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        preds = torch.argmax(out, dim=1)
        total += images.size(0)
        correct += (preds == labels).sum().item()

    print(f'{correct}/{total}={correct / total}')
    finish_time = datetime.now()
    print(finish_time - start_time)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
