import os
import glob
import math
import time
import datetime
import re
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
from data import CAEDataset
from model import CDAE


def load_metrics(fname, dir):
    metrics = open(os.path.join(dir, fname)).readlines()
    epochs, mses = [], []
    for metric in metrics:
        epochs.append(int(metric.split(',')[0]))
        mses.append(float(metric.split(',')[1]))
    return epochs, mses


def psnr(mse):
    rmse = math.sqrt(mse)
    return 20 * math.log10(5.1 / rmse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mod", required=True, type=str, help="model type")
    parser.add_argument("-c", "--base_ch", type=int, default=8, help="base channel for unet")
    parser.add_argument("-n", "--noise", required=True, type=str, help="synthesised noise type")
    parser.add_argument("-rn", "--random_noise", dest="random_noise", action="store_true", help="random noise strength")
    parser.add_argument("-na", "--noise_app", required=True, type=str, help="noise app: naive, add_shift")
    parser.add_argument("-b", "--num_blocks", type=int, default=2, help="number of blocks in encoder/decoder")
    parser.add_argument("-e", "--epochs", type=int, default=201, help="number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("-a", "--aug", dest="aug", action="store_true", help="augmentation")
    parser.add_argument("--no_aug", dest="aug", action="store_false", help="no augmentation")
    parser.add_argument("-g", "--geoid", type=str, default="none", help="geoid dir")
    parser.add_argument("-p", "--prior", type=str, default="none", help="prior dir: default set to 'none' string, option to set to 'geodetic_gauss'")
    parser.add_argument("-i", "--insize", type=int, default=128, help="size of regions")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-ld", "--lr_decay", type=float, default=1)
    parser.add_argument("-v", "--var", type=str, default='cs', help="cs or mdt")
    parser.add_argument("-w", "--weight_dec", type=float, default=0, help="set weight decay")
    parser.add_argument("-do", "--dropout", dest="dropout", action="store_true", help="dropout")
    parser.add_argument("-el", "--elastic", dest="elastic", action="store_true", help="elastic transform")
    parser.add_argument("--no_el", dest="elastic", action="store_false", help="no elastic")
    parser.add_argument("-lo", "--loss", type=str, default='l2', help="l1 or l2")
    parser.set_defaults(aug=True, elastic=True)
    args = parser.parse_args()
    print("Starting train.py: ", datetime.datetime.now())

    model_type = args.mod
    base_ch = args.base_ch
    noise = args.noise
    noise_app = args.noise_app
    num_blocks = args.num_blocks
    epochs = args.epochs
    batch_size = args.batch_size
    augmentation = args.aug
    geoid = args.geoid
    prior = args.prior
    in_size = args.insize
    lr = args.learning_rate
    lr_decay = args.lr_decay
    var = args.var
    w_d = args.weight_dec
    dropout = args.dropout
    random_noise = args.random_noise
    elastic = args.elastic
    loss_type = args.loss

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if augmentation:
        if elastic:
            transforms = A.Compose([
                            A.augmentations.geometric.rotate.RandomRotate90(),
                            A.augmentations.transforms.Flip(),
                            A.augmentations.geometric.transforms.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                        ])
        else:
            transforms = A.Compose([
                    A.augmentations.geometric.rotate.RandomRotate90(),
                    A.augmentations.transforms.Flip()
                ])
    else:
        transforms = None

    mdt_bool = {'mdt': True, 'cs': False}

    f_prefix = '../a_mdt_data/'
    os_location = 'unipc_outputs'
    num_workers = 1
    root_folder = f'skip_models_{model_type}'
    training_paths = f'training_data/{var}_training_regions_{in_size}'
    val_path = f'training_data/{var}_validation_regions_{in_size}'
    test_path = f'training_data/{var}_testing_regions_{in_size}'
    metrics_save_dir = f'{root_folder}/{os_location}/{noise_app}/size{in_size}/b{num_blocks}/{noise}'

    if prior == 'none':
        prior_dir = None 
    elif prior == 'geodetic_gauss' or prior == 'gg':
        prior_dir = f_prefix + f'training_data/prior_{in_size}/prior_gtim6_gauss'
    else:
        print('prior dir not recognised')
        exit()

    if geoid == 'none':
        geoid_dir = None
        in_channels = 1
    elif geoid == 'gtim6_grads':
        geoid_dir = f_prefix + f'training_data/geoids_{in_size}/gtim6_norm_grads'
        in_channels = 2
    else:
        print('geoid dir not recognised')
        exit()

    metrics_save_dir = metrics_save_dir + f'_{var}'

    quilt_dir = f_prefix + f'quilting/{noise}_{var}_{in_size}'
    orig_quilt_dir = f_prefix + f'quilting/residual_{var}_{in_size}'

    train_data = CAEDataset(
        region_dir=f_prefix + training_paths,
        quilt_dir=quilt_dir,
        in_size=in_size,
        noise_app=noise_app,
        fwd=False,
        geoid_dir=geoid_dir,
        prior_dir=prior_dir,
        mdt=mdt_bool[var],
        transform=transforms,
        rand_noise=random_noise,
    )
    val_data = CAEDataset(
        region_dir=f_prefix + val_path,
        quilt_dir=quilt_dir,
        in_size=in_size,
        noise_app=noise_app,
        fwd=False,
        geoid_dir=geoid_dir,
        prior_dir=prior_dir,
        mdt=mdt_bool[var],
        rand_noise=random_noise,
        )
    test_data = CAEDataset(
        region_dir=f_prefix + test_path,
        quilt_dir=orig_quilt_dir,
        in_size=in_size,
        noise_app='test_regions',
        fwd=False,
        geoid_dir=geoid_dir,
        prior_dir=prior_dir,
        mdt=mdt_bool[var],
        rand_noise=random_noise
        )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
        )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
        )

    model = CDAE(num_blocks=num_blocks, input_ch=in_channels, base_ch=base_ch, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_d)

    def extract_number_from_string(path):
        fname = os.path.basename(path)
        no = re.findall("\d+", fname)
        return (int(no[0]) if no else -1, fname)

    best_val_score = 100000
    epoch_path = f'{metrics_save_dir}/epochs'
    if os.path.exists(epoch_path):
        print("Loading pretrained weights")
        trained_model_paths = glob.glob(os.path.join(epoch_path, '*.pth'))
        max_epoch_path = max(trained_model_paths, key=extract_number_from_string)
        print("max_epoch_path:", max_epoch_path)
        checkpoint = torch.load(max_epoch_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        _, val_mses = load_metrics(f'residual_validation_{var}_metrics.txt', metrics_save_dir)
        best_val_score = min(val_mses)
    else:
        os.makedirs(metrics_save_dir, exist_ok=True)
        start_epoch = 0
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    loss_functions = {
        'l1': nn.L1Loss,
        'l2': nn.MSELoss
    }
    loss_function_choice = loss_functions[loss_type]
    criterion = loss_function_choice(reduction='none')

    f = open(os.path.join(metrics_save_dir, 'model_params.txt'), 'a+')
    f.write(str(vars(args)))
    f.write(f'\n val_data: {val_path}')
    f.close()

    scheduler = lr_scheduler.ExponentialLR(optimizer, lr_decay)
    train_no_iters = len(train_loader)
    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss = 0.0
        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            start_time = time.time()
            images = data[0].to(device)
            targets = data[1].to(device)
            outputs = model(images)
            mask = targets != 0

            loss = (criterion(outputs, targets) * mask).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.detach()
            print('[{}/{}] Epoch: {}/{} \tTraining Loss: {:.6f}. iTime: {:.2f}  Time:{}  Lr:{:.6f}'.format(
                i,
                train_no_iters,
                epoch,
                epochs,
                train_loss,
                time.time() - start_time,
                datetime.datetime.now(),
                optimizer.state_dict()['param_groups'][0]['lr']
                ))

        f = open(os.path.join(metrics_save_dir, f'{noise}_{var}_training_metrics.txt'), 'a+')
        f.write(f'{epoch},{train_loss.item()/len(train_loader)}\n')
        f.close()

        val_loss = 0.0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = imgs.to(device)
                tgts = tgts.to(device)
                out = model(imgs)
                mse = nn.MSELoss(reduction='none')(out, tgts)
                msk = tgts != 0
                msk = msk.float()
                mask = msk.bool()
                mse = mse[mask].mean().item()
                val_loss = val_loss + mse

        mse_val_score = val_loss/len(val_loader)
        f = open(os.path.join(metrics_save_dir, f'{noise}_{var}_validation_metrics.txt'), 'a+')
        print(f'{epoch},{mse_val_score}\n')
        f.write(f'{epoch},{mse_val_score}\n')
        f.close()
        if mse_val_score < best_val_score:
            best_val_score = mse_val_score
            if torch.cuda.device_count() > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            os.makedirs(metrics_save_dir, exist_ok=True)
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(metrics_save_dir, f'{var}_best_val.pth'))

        scheduler.step()
        print('Epoch: {}/{}. eTime: {:.2f}  Time: {}   DTime: {}'.format(
            epoch,
            epochs,
            time.time() - epoch_start_time,
            time.time(),
            datetime.datetime.now()
        ))
    print("Finished: ", datetime.datetime.now())

    mses_test = []
    psnrs_test = []
    with torch.no_grad():
        for imgs, tgts in test_loader:
            imgs = imgs.to(device)
            tgts = tgts.to(device)
            out = model(imgs)
            se = nn.MSELoss(reduction='none')(out, tgts)
            l_msk = tgts != 0
            l_msk = l_msk.float()
            msk = l_msk.bool()
            mse = se*msk
            mse = mse.mean().item()
            mses_test.append(mse)
            psnrs_test.append(psnr(mse))

    mse_test_score = np.mean(mses_test)
    psnr_test_score = np.mean(psnrs_test)
    f = open(os.path.join(metrics_save_dir, f'{noise}_{var}_testing_metrics.txt'), 'a+')
    f.write(f'{epoch},{mse_test_score},{psnr_test_score}\n')
    f.close()


if __name__ == "__main__":
    main()