import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from advt.attack import MIM

PATH_PARAMETERS = 'tests/res18_model.pth'  # config your model.pth path
PATH_ADVIMGS = 'data/dim_res18'  # config your test imgs path

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

    # initialize attack method
    mim = MIM(net, device)

    # initialize indicator
    attack_succ = 0
    total_num = 0
    id = 0

    # start attacking
    for i, (img, lbl) in enumerate(test_loader):
        img, lbl = img.to(device), lbl.to(device)
        adv_img = mim.attack(img, lbl)

        output = net(adv_img)
        _, pred_indice = output.max(1)
        # mim.display(adv_img[0])
        # dim.save_to_path(PATH_ADVIMGS, id, adv_img, lbl)
        # id += len(lbl)

        total_num += len(lbl)
        attack_succ += (pred_indice == lbl).sum().item()
        if (i + 1) % 20 == 0:
            print('batch {}:'.format((i + 1) // 20),
                  'total tested number: {}, correct number: {}'.format(total_num, attack_succ))

    # result paddle
    print('----------------------Summary----------------------')
    print('MIM attack result:')
    print('parameters:','eps:{}, max_iters:{}, decay_factor:{}'.format(mim.eps, mim.itr_numbers, mim.decay_factor))
    print('model predict correct rate:', attack_succ / total_num)
    print('l2 distance of average perturbation:', mim.total_distance / mim.total_num)
    print('------------------------End------------------------')

if __name__ == '__main__':
    main()