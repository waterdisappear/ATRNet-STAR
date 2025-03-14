import sys
sys.path.append('..')
import torch
import numpy as np
import re
from tqdm import tqdm
import argparse
import torch.nn as nn

import collections
from functools import partial
import torchvision.transforms as transforms
from utils.DataLoad import load_data,data_transform
from utils.TrainTest import model_train, model_val, model_test
from model.ConvNeXt import convnext_1

def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    parser.add_argument('--data_path', type=str, default="../地距/EOC_azimuth/",
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=int, default=0,
                        help='GPU ids')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--classes', type=int, default=40,
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=1,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    return args

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

if __name__ == '__main__':
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)
    # torch.manual_seed(arg.seed)
    # torch.cuda.manual_seed(arg.seed)
    history = collections.defaultdict(list)  # 记录每一折的各种指标

    train_all = load_data(arg.data_path + 'train',data_transform)
    test_set = load_data(arg.data_path + 'test',data_transform)
    test_set_1 = load_data(arg.data_path + 'test_60',data_transform)
    test_set_2 = load_data(arg.data_path + 'test_120',data_transform)
    test_set_3 = load_data(arg.data_path + 'test_180',data_transform)
    test_set_4 = load_data(arg.data_path + 'test_240',data_transform)
    test_set_5 = load_data(arg.data_path + 'test_300',data_transform)

    torch.cuda.set_device(arg.GPU_ids)
    for k_F in tqdm(range(arg.fold)):
        # train_set, val_set = torch.utils.data.random_split(train_all, [len(train_all) - int(len(train_all) / arg.fold),
        #                                                                int(len(train_all) / arg.fold)])
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(val_set, batch_size=arg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)
        test_loader_1 = torch.utils.data.DataLoader(test_set_1, batch_size=arg.batch_size, shuffle=False)
        test_loader_2 = torch.utils.data.DataLoader(test_set_2, batch_size=arg.batch_size, shuffle=False)
        test_loader_3 = torch.utils.data.DataLoader(test_set_3, batch_size=arg.batch_size, shuffle=False)
        test_loader_4 = torch.utils.data.DataLoader(test_set_4, batch_size=arg.batch_size, shuffle=False)
        test_loader_5 = torch.utils.data.DataLoader(test_set_5, batch_size=arg.batch_size, shuffle=False)
        # print('train shape:{}, val shape{}, test shape{}'.format(len(train_loader.dataset), len(val_loader.dataset),
        #                                                          len(test_loader.dataset)))
        #
        model = convnext_1(arg.classes)

        # opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=arg.lr, weight_decay=4e-3)
        # opt = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=0.004)
        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
        best_test_accuracy = 0
        for epoch in range(1, arg.epochs + 1):
            print("##### " + str(k_F + 1) + " EPOCH " + str(epoch) + "#####")
            loss = model_train(model=model, data_loader=train_loader, opt=opt, sch=scheduler)
            accuracy = model_val(model, test_loader)
            print("Val Accuracy is:{:.2f} %: ".format(accuracy))

            # if best_test_accuracy <= accuracy:
            #     best_epoch = epoch
            #     best_test_accuracy = accuracy

        acc = model_test(model, test_loader)
        acc_1 = model_test(model, test_loader_1)
        acc_2 = model_test(model, test_loader_2)
        acc_3 = model_test(model, test_loader_3)
        acc_4 = model_test(model, test_loader_4)
        acc_5 = model_test(model, test_loader_5)
        print('test accuracy is {}, {}, {}, {}, {}, {}'.
              format(acc, acc_1, acc_2, acc_3, acc_4, acc_5))
        history['accuracy'].append(acc)
        history['accuracy_1'].append(acc_1)
        history['accuracy_2'].append(acc_2)
        history['accuracy_3'].append(acc_3)
        history['accuracy_4'].append(acc_4)
        history['accuracy_5'].append(acc_5)

    print('OA is {}, STD is {}'.format(np.mean(history['accuracy']), np.std(history['accuracy'])))
    print(history['accuracy'])

    print('OA is {}, STD is {}'.format(np.mean(history['accuracy_1']), np.std(history['accuracy_1'])))
    print(history['accuracy_1'])
    print('OA is {}, STD is {}'.format(np.mean(history['accuracy_2']), np.std(history['accuracy_2'])))
    print(history['accuracy_2'])
    print('OA is {}, STD is {}'.format(np.mean(history['accuracy_3']), np.std(history['accuracy_3'])))
    print(history['accuracy_3'])
    print('OA is {}, STD is {}'.format(np.mean(history['accuracy_4']), np.std(history['accuracy_4'])))
    print(history['accuracy_4'])
    print('OA is {}, STD is {}'.format(np.mean(history['accuracy_5']), np.std(history['accuracy_5'])))
    print(history['accuracy_5'])

    torch.save(model.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '.pth')
    np.save('./results/' + re.split('[/\\\]', arg.data_path)[-2] + '_result.npy', history)

        # torch.save(model.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '_Unet.pth')
        # if isinstance(model, torch.nn.DataParallel):
        #     torch.save(model.module.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '_Unet.pth')
        # else:
        #     torch.save(model.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '_Unet.pth')
