from collections import Counter
import json
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
from pathlib import Path
import torch
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from PIL import Image
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

class ColorAttribute(VisionDataset):
    
    def __init__(
        self,
        root: str,
        annotation: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._label_ids = {'Black': 0, 'Grey': 1, 'Blue': 2, 'White': 3, 'Brown': 4, 'Red': 5, 'Purple': 6, 'Green': 7, 'Pink': 8, 'Orange': 9, 'Yellow': 10}
        self._split = split
        self._is_train = True if self._split == 'train' else False
        self._contexts = os.listdir(root)
        self._contexts.sort()
        print(self._contexts)
        self._base_folder = root
        self._images_folder = self._base_folder + '/'
        self._annotation = annotation
        # self._is_train = 'True' if 'train' in split else 'False'
        # Xử lý các context cho từng split
        if self._split == 'train':
            self._contexts = self._contexts[:-2]  # Tất cả các context trừ 2 context cuối
        elif self._split == 'val':
            self._contexts = [self._contexts[-2]]  # Context thứ hai cuối
        elif self._split == 'test':
            self._contexts = [self._contexts[-1]]  # Context cuối cùng
        self._labels, self._image_files = self.get_dataset()
        print(Counter(self._labels))

    def get_dataset(self):
        all_images = []
        all_labels = []
        for context in self._contexts:

            with open(os.path.join(self._base_folder, context+'/Label_process.json'), 'r') as f:
                data = json.load(f)
                print(data[0]['is_train'])
                list_images = [d['image'] for d in data if d['color'] is not None and d['is_train'] == self._is_train]
                list_labels = [self._label_ids[d['color']] for d in data if d['color'] is not None and d['is_train'] == self._is_train]
                all_images.extend(list_images)
                all_labels.extend(list_labels)

        return all_labels, all_images

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

    
        # label = F.one_hot(label, num_classes=8).to(float)

        return image, label

if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = ColorAttribute('/mnt/data/PETA_dataset', '/home/data/annotation.txt',
                        'test', transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            
            normalize,
        ]))
    print(dataset[0])