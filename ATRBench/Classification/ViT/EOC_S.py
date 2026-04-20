import sys
sys.path.append('..')
import os
import torch
import numpy as np
import re
from tqdm import tqdm
import argparse
import torch.nn as nn

import collections
from functools import partial
import torchvision.transforms as transforms
from utils.DataLoad import load_data, load_data_with_class_mapping, data_transform
from utils.TrainTest import model_train, model_val, model_test
from model.Model import convnext_1, ResNet_34

def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    parser.add_argument('--data_path', type=str, default="../../地距/EOC_scene/",
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=int, default=0,
                        help='GPU ids')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--classes', type=int, default=40,
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=1,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    # test-only by default
    parser.add_argument('--weights', type=str, default="",
                        help='path to model weights (.pth) for test-only')
    parser.add_argument('--test_only', action='store_true', default=True,
                        help='only run evaluation using --weights (default: enabled)')
    parser.add_argument('--train', dest='test_only', action='store_false',
                        help='disable test-only mode and run training')
    args = parser.parse_args()
    if not args.weights:
        here = os.path.dirname(os.path.abspath(__file__))
        w1 = os.path.join(here, 'model', 'EOC_scene.pth')
        w2 = os.path.join(here, 'Model', 'EOC_scene.pth')
        args.weights = w1 if os.path.exists(w1) else w2
    return args


if __name__ == '__main__':
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)
    # torch.manual_seed(arg.seed)
    # torch.cuda.manual_seed(arg.seed)
    history = collections.defaultdict(list)  # 记录每一折的各种指标

    train_all = load_data(arg.data_path + 'train', data_transform)
    # Use the training set mapping as the global label space.
    global_class_to_idx = train_all.class_to_idx
    arg.classes = len(train_all.classes)

    test_set = load_data_with_class_mapping(arg.data_path + 'test', data_transform, global_class_to_idx)
    test_set_1 = load_data_with_class_mapping(arg.data_path + 'test_urban', data_transform, global_class_to_idx)
    test_set_2 = load_data_with_class_mapping(arg.data_path + 'test_woodland', data_transform, global_class_to_idx)
    test_set_3 = load_data_with_class_mapping(arg.data_path + 'test_factory', data_transform, global_class_to_idx)

    torch.cuda.set_device(arg.GPU_ids)

    def _load_weights_into_model(model, weights_path: str):
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        return model

    if arg.test_only:
        model = ResNet_34(arg.classes)
        model = _load_weights_into_model(model, arg.weights)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)
        test_loader_1 = torch.utils.data.DataLoader(test_set_1, batch_size=arg.batch_size, shuffle=False)
        test_loader_2 = torch.utils.data.DataLoader(test_set_2, batch_size=arg.batch_size, shuffle=False)
        test_loader_3 = torch.utils.data.DataLoader(test_set_3, batch_size=arg.batch_size, shuffle=False)

        acc = model_test(model, test_loader)
        acc_1 = model_test(model, test_loader_1)
        acc_2 = model_test(model, test_loader_2)
        acc_3 = model_test(model, test_loader_3)
        print('test accuracy is {}, {}, {}, {}'.
              format(acc, acc_1, acc_2, acc_3))
        sys.exit(0)

    for k_F in tqdm(range(arg.fold)):
        # train_set, val_set = torch.utils.data.random_split(train_all, [len(train_all) - int(len(train_all) / arg.fold),
        #                                                                int(len(train_all) / arg.fold)])
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        # val_loader = torch.utils.data.DataLoader(val_set, batch_size=arg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)
        test_loader_1 = torch.utils.data.DataLoader(test_set_1, batch_size=arg.batch_size, shuffle=False)
        test_loader_2 = torch.utils.data.DataLoader(test_set_2, batch_size=arg.batch_size, shuffle=False)
        test_loader_3 = torch.utils.data.DataLoader(test_set_3, batch_size=arg.batch_size, shuffle=False)
        # print('train shape:{}, val shape{}, test shape{}'.format(len(train_loader.dataset), len(val_loader.dataset),
        #                                                          len(test_loader.dataset)))
        #
        model = ResNet_34(arg.classes)

        # opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=arg.lr, weight_decay=4e-3)
        # opt = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=0.004)
        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-4,betas=(0.9,0.95))
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
        print('test accuracy is {}, {}, {}, {}'.
              format(acc, acc_1, acc_2, acc_3))
        history['accuracy'].append(acc)
        history['accuracy_1'].append(acc_1)
        history['accuracy_2'].append(acc_2)
        history['accuracy_3'].append(acc_3)

        print('OA is {}, STD is {}'.format(np.mean(history['accuracy']), np.std(history['accuracy'])))
        print(history['accuracy'])

        print('OA is {}, STD is {}'.format(np.mean(history['accuracy_1']), np.std(history['accuracy_1'])))
        print(history['accuracy_1'])
        print('OA is {}, STD is {}'.format(np.mean(history['accuracy_2']), np.std(history['accuracy_2'])))
        print(history['accuracy_2'])
        print('OA is {}, STD is {}'.format(np.mean(history['accuracy_3']), np.std(history['accuracy_3'])))
        print(history['accuracy_3'])

        torch.save(model.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '.pth')
        np.save('./results/' + re.split('[/\\\]', arg.data_path)[-2] + '_result.npy', history)

        # torch.save(model.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '_Unet.pth')
        # if isinstance(model, torch.nn.DataParallel):
        #     torch.save(model.module.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '_Unet.pth')
        # else:
        #     torch.save(model.state_dict(), './Model/' + re.split('[/\\\]', arg.data_path)[-2] + '_Unet.pth')
