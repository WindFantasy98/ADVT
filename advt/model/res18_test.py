import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from advt.attack import DIM

PATH_PARAMETERS = '../../tests/res18_model.pth'
PATH_ADVIMGS = 'data/dim'

def main():
    # initialize dataset
    transform = transforms.Compose([transforms.ToTensor()])
    t = transforms.Compose([transforms.ToPILImage()])
    test_dataset = datasets.CIFAR10(root='/data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load victim model
    net = resnet18()
    net.fc = torch.nn.Linear(in_features=512, out_features=10)
    net.load_state_dict(torch.load(PATH_PARAMETERS))
    net.eval()
    net = net.to(device)

    # initialize indicator
    succ = 0
    total_num = 0
    id = 0

    # start attack
    for i, (img, lbl) in enumerate(test_loader):
        img, lbl = img.to(device), lbl.to(device)

        output = net(img)
        _, pred_indice = output.max(1)

        total_num += len(lbl)
        succ += (pred_indice == lbl).sum().item()
        if (i + 1) % 20 == 0:
            print('batch {}:'.format((i + 1) // 20),
                  'total tested number: {}, correct number: {}'.format(total_num, succ))

    # result paddle
    print('----------------------Summary----------------------')
    print('res18 result:')
    print('model predict correct rate:', succ / total_num)
    print('------------------------End------------------------')

if __name__ == '__main__':
    main()