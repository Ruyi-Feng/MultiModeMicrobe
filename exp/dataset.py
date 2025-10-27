from torch.utils.data import Dataset
from utils.json_loader import load_json
import torch
import numpy as np
import os
import h5py


class IndividualDataset(Dataset):
    """
    individual 数据形式
    加载之后返回property description 和 protein repr
    其中property description是一串字符str
    """
    def __init__(self, args, **kwargs):
        self.init_train_index(args)
        self.dataset_length = len(self.train_idx)
        self.data_path = args.data_path
        self.max_seq_len = args.max_seq_len
        self.txt_loader = TxtDataLoader(self.data_path)

    def init_train_index(self, args):
        index_path = args.index_path
        protein_index_path = args.protein_index_path
        self.train_idx = []
        for line in open(index_path, 'rb'):
            line = line.decode().split()[0].split(',')
            self.train_idx.append([line[0],
                                   int(line[1]),
                                   int(line[2])])
            # bacdive_id, txt_head, txt_tail
        self.protein_index = load_json(protein_index_path)

    def _get_aa_repr_from_h5(self, aa_index_list, bacdive_id, ):
        aa_loader = AaIndividualLoader(self.data_path, f"{bacdive_id}.h5")
        aa_representation = None
        for aa_index in aa_index_list:
            for item in aa_index:
                if aa_representation is None:
                    aa_representation = aa_loader(item).unsqueeze(0)
                else:
                    aa_representation = torch.concat((aa_representation, aa_loader(item).unsqueeze(0)), axis=0)
        #         if len(aa_representation) >= self.max_seq_len:
        #             return aa_representation
        # if len(aa_representation) < self.max_seq_len:
        #     shape = aa_representation[0].shape
        #     aa_representation = aa_representation + np.zeros(shape).tolist() * (self.max_seq_len - len(aa_representation))
        return aa_representation

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index: int):
        """
        return:
        aa_seq_batch, property_seq_batch
        """
        key, txt_head, txt_tail = self.train_idx[index]
        description = self.txt_loader(txt_head, txt_tail)
        aa_index_list = self.protein_index[key]
        aa_representation = self._get_aa_repr_from_h5(aa_index_list, key)
        return description, aa_representation


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
    def __init__(self, data_path, file_name="microbe.h5"):
        self.h5_path = os.path.join(data_path, file_name)
        # 打开一次，但不加载数据
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f.attrs['index']

    def __len__(self):
        return self.length

    def __call__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            feature = f['microbe_features'][idx]  # 只读index
        return torch.tensor(feature, dtype=torch.float32)


class AaIndividualLoader:
    def __init__(self, data_path, file_name="microbe.h5"):
        self.h5_path = os.path.join(data_path, file_name)
        # 打开一次，但不加载数据
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f.attrs['index']

    def __len__(self):
        return self.length

    def __call__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            feature = f['protein_features'][idx]  # 只读index
        return torch.tensor(feature, dtype=torch.float32)
