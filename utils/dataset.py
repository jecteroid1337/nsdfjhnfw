from torch.utils.data import Dataset
from .load import load_ct_data, ct_paths_sort_key


class CTSegmentationDataset(Dataset):
    def __init__(self, data_folder='./data/segmentation_data', train=True):
        import glob
        import os

        images_folder = os.path.join(data_folder, 'images')
        masks_folder = os.path.join(data_folder, 'masks')

        image_paths = sorted(glob.glob(os.path.join(images_folder, '*.dcm')), key=ct_paths_sort_key)
        masks_paths = sorted(glob.glob(os.path.join(masks_folder, '*.nrrd')), key=ct_paths_sort_key)

        self._items = list(zip(image_paths, masks_paths))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        import torchvision.transforms.v2 as transforms
        import torch

        image_conversion_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=False),
            transforms.Normalize([-607.6368604278564], [471.41841392312153])
        ])

        image, mask = load_ct_data(self._items[index])
        image, mask = image_conversion_transform(image), torch.LongTensor(mask)

        return image, mask
