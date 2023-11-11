import glob
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor


class CAEDataset(Dataset):
    def __init__(self, region_dir, quilt_dir, in_size, noise_app,
                 fwd=False, geoid_dir=None, prior_dir=None,
                 mdt=False, transform=None, rand_noise=True):
        self.region_dir = region_dir
        self.quilt_dir = quilt_dir
        self.in_size = in_size
        self.noise_app = noise_app
        if "test_regions" in noise_app and noise_app.split("_")[-1] not in ['mdt', 'cs']:
            noise_app += "_mdt" if mdt else "_cs"
        self.fwd = fwd
        self.transform = transform
        self.geoid_dir = geoid_dir
        self.prior_dir = prior_dir
        self.mdt = mdt
        self.rand_noise = rand_noise

        if self.quilt_dir is not None:
            if not os.path.exists(self.quilt_dir):
                raise ValueError("Quilt dir doesn't exist")

        self.paths = glob.glob(os.path.join(region_dir, '*.npy'))

        if len(self.paths) == 0:
            raise ValueError("Training regions not found")
        if prior_dir is not None:
            self.prior_paths = glob.glob(os.path.join(prior_dir, '*.npy'))
            if not os.path.exists(prior_dir):
                raise ValueError("Prior directory doesn't exist")
            if len(self.prior_paths) == 0:
                raise ValueError("Prior regions not found")
        if geoid_dir is not None:
            self.geoid_paths = glob.glob(os.path.join(geoid_dir, '*.npy'))
            if not os.path.exists(geoid_dir):
                raise ValueError("Geoid directory doesn't exist")
            if len(self.geoid_paths) == 0:
                raise ValueError("Geoid regions not found")
        if self.quilt_dir is not None:
            self.quilt_paths = glob.glob(os.path.join(quilt_dir, '*.png'))
            if not os.path.exists(quilt_dir):
                raise ValueError("Quilt directory doesn't exist")
            if len(self.quilt_paths) == 0:
                raise ValueError("Quilt regions not found")

        self.noise_apps = {
            "naive": [self.load_norm_quilt, self.add_noise],
            "add_shift": [self.load_norm_quilt, self.add_shift_noise],
            "test_regions_mdt": [self.load_matching_quilt, self.add_noise],
            "test_regions_cs": [self.load_matching_quilt, self.add_shift_noise],
        }

    def __len__(self):
        return len(self.paths)

    def load_matching_quilt(self, x, y):
        matching_regions = self.get_quilt_regions(x, y)
        quilt = random.choice(matching_regions)
        quilt = self._open_and_normalise_quilt(quilt)
        return quilt

    def region_from_fpath(self, path):
        split_path = path[:len(path)-4].split('_')
        a, b = int(split_path[-2]), int(split_path[-1])
        return a, b

    def get_region_prior(self, x, y):
        for i in range(len(self.prior_paths)):
            a, b = self.region_from_fpath(self.prior_paths[i])
            if x == a and y == b:
                return np.load(self.prior_paths[i])

    def get_geoid_region(self, x, y):
        for i in range(len(self.geoid_paths)):
            a, b = self.region_from_fpath(self.geoid_paths[i])
            if x == a and y == b:
                return np.load(self.geoid_paths[i])

    def _open_and_normalise_quilt(self, quilt_path):
        quilt = Image.open(quilt_path)
        quilt = quilt.convert(mode='L')
        quilt = np.array(quilt).astype(np.float32)
        quilt = (quilt - np.min(quilt)) / (np.max(quilt) - np.min(quilt) + 1e-8)
        return quilt

    def load_norm_quilt(self):
        random_quilt = random.choice(self.quilt_paths)
        quilt = self._open_and_normalise_quilt(random_quilt)
        return quilt

    def add_noise(self, target_img, quilt, prior, noise_strength):
        return target_img + quilt * prior * noise_strength

    def add_shift_noise(self, target_img, quilt, prior, noise_strength):
        img = self.add_noise(target_img, quilt, prior, noise_strength)
        img = (img * (min(quilt.max(), target_img.max()) + abs((quilt.max() - target_img.max()))/4)) / max(quilt.max(), target_img.max())
        return img

    def __getitem__(self, idx):
        input_name = self.paths[idx]
        target_img = np.load(input_name)

        if self.rand_noise:
            if self.mdt:
                noise_strength = random.uniform(0.3, 0.7)
            else:
                noise_strength = random.uniform(0.5, 2.5)
        target_img[np.isnan(target_img)] = 0

        x, y = self.region_from_fpath(self.paths[idx])
        if self.fwd:
            if self.geoid_dir is not None:
                geoid = self.get_geoid_region(x, y)
                target_img = ToTensor()(np.array(target_img))
                geoid = ToTensor()(np.array(geoid))
                target_img = torch.cat((target_img, geoid), 0)
                return target_img, None

            return ToTensor()(target_img), None

        if self.noise_app == "test_regions_mdt" or self.noise_app == "test_regions_cs":
            quilt = self.noise_apps[self.noise_app][0](x, y)
        else:
            quilt = self.noise_apps[self.noise_app][0]()
        if self.mdt:
            mask = np.logical_not(np.logical_and(target_img < 1e-10, target_img > -1e-10))
            quilt = (quilt * 4) - 2
            quilt = quilt - (np.mean(quilt[mask]) if np.any(mask) else 0)
        else:
            mask = target_img > 1e-10

        x, y = self.region_from_fpath(self.paths[idx])

        if self.prior_dir is not None:
            prior = self.get_region_prior(x, y)
        else:
            prior = 1

        seed = np.random.randint(0, 2 ^ 31 - 1)
        if self.transform is not None:
            random.seed(seed)
            t = self.transform(image=target_img, mask=mask.astype(np.uint8))
            target_img = t['image']
            mask = t['mask']

            if self.prior_dir is not None:
                random.seed(seed)
                t = self.transform(image=prior, mask=mask.astype(np.uint8))
                prior = t['image']

        target_img *= mask

        img = self.noise_apps[self.noise_app][1](target_img, quilt, prior, noise_strength) * mask

        if self.mdt:
            target_img = (target_img + 2) / 4
            img = (img + 2) / 4
            img = np.clip(img, 0, 1)
        target_img[np.logical_not(mask)] = 0

        if self.geoid_dir is not None:
            x, y = self.region_from_fpath(self.paths[idx])
            geoid = self.get_geoid_region(x, y)
            if self.transform is not None:
                random.seed(seed)
                t = self.transform(image=geoid, mask=mask.astype(np.uint8))
                geoid = t['image']
            img = ToTensor()(np.array(img))
            geoid = ToTensor()(np.array(geoid))
            img = torch.cat((img, geoid), 0)

            return img, ToTensor()(np.array(target_img))
        return ToTensor()(np.array(img)), ToTensor()(np.array(target_img))
