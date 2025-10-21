from torch.utils.data import Dataset
import torch
import os
import h5py



class CollectiveDataset(Dataset):
    def __init__(self, args, **kwargs):
        self.init_train_index(args.index_path)
        self.dataset_length = len(self.train_idx)
        self.data_path = args.data_path
        self.init_data_loader()

    def init_data_loader(self):
        self.txt_loader = TxtDataLoader(self.data_path)
        self.aa_loader = AaDataLoader(self.data_path)

    def init_train_index(self, index_path):
        self.train_idx = []
        for line in open(index_path, 'rb'):
            line = line.decode().split()[0].split(',')
            self.train_idx.append([line[0],
                                   int(line[1]),
                                   int(line[2]),
                                   int(line[3])])
            # bacdive_id, txt_head, txt_tail, aa_index

    def __getitem__(self, index: int):
        """
        return:
        aa_seq_batch, property_seq_batch
        """
        key, txt_head, txt_tail, aa_index = self.train_idx[index]
        description = self.txt_loader(txt_head, txt_tail)
        aa_representation = self.aa_loader(aa_index)
        return description, aa_representation

    def __len__(self):
        return self.dataset_length


class BinaryDataLoader:
    def __init__(self, data_path):
        self.f_data = open(data_path, 'rb')

    def data_converter(self, info):
        raise NotImplementedError

    def __call__(self, head, tail):
        self.f_data.seek(head)
        info = self.f_data.read(tail - head)
        return self.data_converter(info)

    def __del__(self):
        self.f_data.close()


class TxtDataLoader(BinaryDataLoader):
    def __init__(self, data_path):
        data_path = os.path.join(data_path, "property.bin")
        super(TxtDataLoader, self).__init__(data_path)

    def data_converter(self, info):
        return "<|im_start|> " + info.decode()


class AaDataLoader:
    def __init__(self, data_path):
        self.h5_path = os.path.join(data_path, "microbe.h5")
        # 打开一次，但不加载数据
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f.attrs['index']

    def __len__(self):
        return self.length

    def __call__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            feature = f['microbe_features'][idx]  # 只读index
        return torch.tensor(feature, dtype=torch.float32)
