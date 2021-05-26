import os
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.transforms import ToPILImage
from advt.model.cnn_rnn import *
from advt.attack import attack
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from advt.attack import FGSM
from advt.utils import singlevideo_save_to_path

PATH_ADV_FRAMES = 'D:\\data\\ucf101_jpegs_256\\result_show'

def main():
    # path setting
    # data_path = "D:\\data\\ucf101_jpegs_256\\jpegs_256"  # define UCF-101 RGB train data path
    data_path = "D:\\data\\ucf101_jpegs_256\\test"  # test data path
    action_name_path = "D:\\data\\ucf101_jpegs_256\\UCF101actions.pkl" # test action name path
    save_model_path = "D:\\data\\ucf101_jpegs_256\\CRNN_ckpt/" # model save path

    # use same encoder CNN saved!
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
    CNN_embed_dim = 512  # latent dim extracted by 2D CNN
    img_x, img_y = 256, 342  # resize video 2d frame size
    dropout_p = 0.0  # dropout probability

    # use same decoder RNN saved!
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_FC_dim = 256

    # training parameters
    k = 101  # number of target category
    batch_size = 1
    begin_frame, end_frame, skip_frame = 1, 29, 1 # Select which frame to begin & end in videos

    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)  # load UCF101 actions names

    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # show how many classes there are
    list(le.classes_)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    actions = []
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        actions.append(f[(loc1 + 2): loc2])

        all_names.append(f)

    # list all data files
    all_X_list = all_names  # all video file names
    all_y_list = labels2cat(le, actions)  # all video labels

    # data loading parameters
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    # reset data loader
    all_data_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4,
                       'pin_memory': True} if use_cuda else {}
    all_data_loader = data.DataLoader(
        Dataset_CRNN(data_path, all_X_list, all_y_list, selected_frames, transform=transform), **all_data_params)

    # reload CRNN model
    cnn_encoder = EncoderCNN(img_x=img_x, img_y=img_y, fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                             drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                             h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

    # reload model parameters
    cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'cnn_encoder_epoch36_1GPU.pth')))
    rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'rnn_decoder_epoch36_1GPU.pth')))
    print('CRNN model reloaded!')

    # make all video predictions by reloaded model
    print('Predicting all {} videos:'.format(len(all_data_loader.dataset)))

    # set mode to train for calculating gradients
    cnn_encoder.train()
    rnn_decoder.train()

    total = 0 # total test number
    acc = 0 # predict correct number
    origin_result_lst = []
    adv_result_lst = []

    # attack initializing
    fgsm = FGSM([cnn_encoder, rnn_decoder], device)

    for X, lbl in all_data_loader:
        X = X.to(device)
        lbl = lbl[0]
        lbl = lbl.to(device)

        advX = fgsm.attack(X, lbl) # get adv sample
        singlevideo_save_to_path(advX, PATH_ADV_FRAMES) # save frames to path

        # get origin sample's predict result
        pred0 = rnn_decoder(cnn_encoder(X))
        _, ind0 = pred0.max(1)

        # get adv sample's predict result
        pred = rnn_decoder(cnn_encoder(advX))
        _, ind = pred.max(1)

        if ind.item() == ind0.item():
            print('fgsm failed')
        else:
            print('fgsm succeeded')

        total += len(X)
        acc += (ind.cpu() == lbl.cpu()).sum().item()

    print('----------------------Summary----------------------')
    print('SparseAdv attack result:')
    print('model predict correct rate:', acc / total)
    print('l2 distance of average perturbation:', dim.total_distance / dim.total_num)
    print('------------------------End------------------------')

if __name__ == "__main__":
    main()