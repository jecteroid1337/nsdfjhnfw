from torch.utils.data import Dataset
from load import load_ct_data, ct_paths_sort_key


class CTSegmentationDataset(Dataset):
    def __init__(self, data_folder):
        import glob
        import os

        images_folder = os.path.join(data_folder, 'images')
        masks_folder = os.path.join(data_folder, 'masks')

        image_paths = sorted(glob.glob(os.path.join(images_folder, '*.dcm')), key=ct_paths_sort_key)
        masks_paths = sorted(glob.glob(os.path.join(masks_folder, '*.nrrd')), key=ct_paths_sort_key)

        self._items = list(zip(image_paths, masks_paths))[:10]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        import torchvision
        torchvision.disable_beta_transforms_warning()
        import torchvision.transforms.v2 as transforms
        import torch

        image_conversion_transform = transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([-824.8447361975601], [778.8566926962736])
        ])

        image, mask = load_ct_data(self._items[index])
        mask = torch.LongTensor(mask)
        return image_conversion_transform(image), mask