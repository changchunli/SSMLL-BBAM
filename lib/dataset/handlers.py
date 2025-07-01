from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCO2014_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class COCO2014_mask_handler(Dataset):
    def __init__(self, X, Y, Mask, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        return x, y, mask

    def __len__(self):
        return len(self.X)


class COCO2014_mm_handler(Dataset):
    def __init__(self, X, Y, Mask, Mask_ns, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.Mask_ns = Mask_ns
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        mask_ns = self.Mask_ns[index]
        return x, y, mask, mask_ns

    def __len__(self):
        return len(self.X)


class COCO2014_handler_Cp(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class NUS_WIDE_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/image'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class NUS_WIDE_mask_handler(Dataset):
    def __init__(self, X, Y, Mask, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.transform = transform
        self.data_path = f'{data_path}/image'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        return x, y, mask

    def __len__(self):
        return len(self.X)


class NUS_WIDE_mm_handler(Dataset):
    def __init__(self, X, Y, Mask, Mask_ns, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.Mask_ns = Mask_ns
        self.transform = transform
        self.data_path = f'{data_path}/image'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        mask_ns = self.Mask_ns[index]
        return x, y, mask, mask_ns

    def __len__(self):
        return len(self.X)


class NUS_WIDE_handler_Cp(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/image'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class VOC2012_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/VOCdevkit/VOC2012'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' +
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class VOC2012_mask_handler(Dataset):
    def __init__(self, X, Y, Mask, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.transform = transform
        self.data_path = f'{data_path}/VOCdevkit/VOC2012'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' +
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        return x, y, mask

    def __len__(self):
        return len(self.X)


class VOC2012_mm_handler(Dataset):
    def __init__(self, X, Y, Mask, Mask_ns, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.Mask_ns = Mask_ns
        self.transform = transform
        self.data_path = f'{data_path}/VOCdevkit/VOC2012'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' +
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        mask_ns = self.Mask_ns[index]
        return x, y, mask, mask_ns

    def __len__(self):
        return len(self.X)


class VOC2012_handler_Cp(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/VOCdevkit/VOC2012'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' +
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

class AWA_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/JPEGImages/'

    def __getitem__(self, index):
        x = Image.open(self.data_path + 
                       self.X[index].split('_')[0] + '/' + 
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)
    
class AWA_mask_handler(Dataset):
    def __init__(self, X, Y, Mask, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.transform = transform
        self.data_path = f'{data_path}/JPEGImages/'

    def __getitem__(self, index):
        x = Image.open(self.data_path + 
                       self.X[index].split('_')[0] + '/' + 
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        return x, y, mask

    def __len__(self):
        return len(self.X)


class AWA_mm_handler(Dataset):
    def __init__(self, X, Y, Mask, Mask_ns, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.Mask_ns = Mask_ns
        self.transform = transform
        self.data_path = f'{data_path}/JPEGImages/'

    def __getitem__(self, index):
        x = Image.open(self.data_path + 
                       self.X[index].split('_')[0] + '/' + 
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        mask_ns = self.Mask_ns[index]
        return x, y, mask, mask_ns

    def __len__(self):

        return len(self.X)


class AWA_handler_Cp(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/JPEGImages/'

    def __getitem__(self, index):
        x = Image.open(self.data_path + 
                       self.X[index].split('_')[0] + '/' + 
                       self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


class CUB_200_2011_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/CUB_200_2011/images'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)


class CUB_200_2011_mask_handler(Dataset):
    def __init__(self, X, Y, Mask, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.transform = transform
        self.data_path = f'{data_path}/CUB_200_2011/images'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        return x, y, mask

    def __len__(self):
        return len(self.X)


class CUB_200_2011_mm_handler(Dataset):
    def __init__(self, X, Y, Mask, Mask_ns, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.Mask = Mask
        self.Mask_ns = Mask_ns
        self.transform = transform
        self.data_path = f'{data_path}/CUB_200_2011/images'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        mask = self.Mask[index]
        mask_ns = self.Mask_ns[index]
        return x, y, mask, mask_ns

    def __len__(self):
        return len(self.X)


class CUB_200_2011_handler_Cp(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = f'{data_path}/CUB_200_2011/images'

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')
        x = self.transform(x)
        y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
