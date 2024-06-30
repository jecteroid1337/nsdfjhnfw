import numpy as np
from typing import Tuple


def ct_paths_sort_key(s: str) -> int:
    return int(s.split('/')[-1].split('.')[0])


def load_ct_data(path: Tuple[str, str]):
    from pydicom import dcmread
    from pydicom.pixel_data_handlers import apply_modality_lut
    from nrrd import read

    image_path, mask_path = path
    dicom = dcmread(image_path)
    image = np.clip(apply_modality_lut(dicom.pixel_array.astype(np.float32)[..., None], dicom), -1024., None)
    mask = read(mask_path)[0]

    return image, mask
