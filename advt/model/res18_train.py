import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18

PATH_PARAMETERS = '../../tests/res18_model.pth'

def main():
    # initialize dataset
    transform = transforms.Compose([transforms.ToTensor()])
    t = transforms.Compose([transforms.ToPILImage()])
    train_dataset = datasets.CIFAR10(root='/data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='/data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load victim model
    net = resnet18()
    net.fc = nn.Linear(512, 10)
    if os.path.exists(PATH_PARAMETERS):
        net.load_state_dict(torch.load(PATH_PARAMETERS))
    print(net)
    net = net.to(device)

    # initialize optimizer, loss
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # initialize parameters
    epoches = 15

    # start attack
    for epoch in range(epoches):
        # initialize indicators
        total_num = 0
        succ = 0

        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            output = net(imgs)
            _, pred_indices = output.max(1)
            optimizer.zero_grad()
            loss_ = loss(output, lbls)
            loss_.backward()

            total_num += len(lbls)
            succ += (pred_indices == lbls).sum().item()

            optimizer.step()

        print('epoch {}:'.format(epoch + 1),
              'total train number: {}, correct number: {}'.format(total_num, succ))

    torch.save(net.state_dict(), PATH_PARAMETERS)

if __name__ == '__main__':
    # epoch 20: total train number: 50000, correct number: 48284
    main()