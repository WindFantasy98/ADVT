import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from advt.model.cnn import CNN
from advt.attack import BIM

PATH_PARAMETERS = 'tests/cnn_model.pth'

def main():
    # initialize dataset
    transform = transforms.Compose([transforms.ToTensor()])
    t = transforms.Compose([transforms.ToPILImage()])
    test_dataset = datasets.CIFAR10(root='/data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load victim model
    net = CNN()
    net.load_state_dict(torch.load(PATH_PARAMETERS))
    net = net.to(device)

    # initialize attack method
    bim = BIM(net, device)

    # initialize indicator
    attack_succ = 0
    total_num = 0

    # start attack
    for i, (img, lbl) in enumerate(test_loader):
        img, lbl = img.to(device), lbl.to(device)
        adv_img = bim.attack(img, lbl)

        output = net(adv_img)
        _, pred_indice = output.max(1)

        total_num += len(lbl)
        attack_succ += (pred_indice == lbl).sum().item()
        if (i + 1) % 20 == 0:
            print('batch {}:'.format((i + 1) // 20),
                  'total tested number: {}, correct number: {}'.format(total_num, attack_succ))

    # result paddle
    print('----------------------Summary----------------------')
    print('BIM attack result:')
    print('parameters:','epsilon:{} max_iter:{}'.format(bim.eps, bim.itr_numbers))
    print('model predict correct rate:', attack_succ / total_num)
    print('l2 distance of average perturbation:', bim.total_distance / bim.total_num)
    print('------------------------End------------------------')

if __name__ == '__main__':
    main()