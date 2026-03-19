
from Bio import SeqIO
from config import get_included_property, data_provider_args
from pathlib import Path
from tqdm import tqdm
from utils.json_loader import load_json, save_json

import esm
import h5py
import numpy as np
import os
import shutil
import time
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class DataProvider:
    """
    Individual 数据存储形式
    overall index
    -------------
    每行是一条数据: [bacdive_id, txt_head, txt_tail]

    protein_index
    -------------
    用于区分每个古菌的蛋白质是否被拆成了多个index
    {
        "bacdive_id": [[aa_idx, aa_idx],   一个蛋白放在一个list里
                       [aa_idx, aa_idx],
                       ...]
    }

    saved h5 file
    -------------
    用于存储每个蛋白的representation
    以bacdive id为文件名 “bacdive_id.h5”
    按index读取数据即可
    数据库有: protein_features, protein_id, microbe_id
    """


    def __init__(self, data_dir, save_path, spare_folders = [], **kwargs):
        self.data_dir = data_dir
        self.save_path = save_path
        self.spare_folders = self._init_spare_repr(spare_folders)
        """
        {
            "folder1": ["file1.h5", "file2.h5", ...],
            "folder2": ["file1.h5", "file2.h5", ...],
            ...
        }
        """
        self.no_exist_path = []
        args = data_provider_args()
        self.args = args
        self.tag = args.tag
        self.split_data = args.split_data
        print(f"Will select data with tag: {self.tag}, split: {self.split_data}")
        self.save_collective_representation = args.collective
        self._init_data()
        self.aa_rep_extractor = ESM2Representation(args)
        # self.aa_rep_extractor = ESMCRepresentation(args)

        if self.save_collective_representation:
            self.aa_storage = CollectiveFeatureStorage(self.save_path, max_shape=(None, self.aa_rep_extractor.dim))
        else:
            self.protein_index = dict()
        overview = os.path.join(data_dir, "microbex_data_with_eggnog.json")
        self.overview = load_json(overview)

        self.valid_property_keys = get_included_property(self.tag)
        self.data_length = len(self.overview)
        self._generate()

    def _init_data(self):
        self.property_f = open(os.path.join(self.save_path, self.args.property_file_name), 'ab+')
        self.property_head = 0
        self.index_f = open(os.path.join(self.save_path, self.args.index_file_name), 'ab+')

    def _init_spare_repr(self, spare_folders):
        spare = dict()
        for folder in spare_folders:
            data_list = os.listdir(folder)
            spare[folder] = data_list
        return spare

    def _generate_property(self, property_info):
        head = self.property_head
        property_info = str(property_info) + "\n"
        property_info = property_info.encode()
        write_len = self.property_f.write(property_info)
        tail = head + write_len
        self.property_head = tail
        return head, tail

    def _update_protein_index_by_h5(self, folder, bacdive_id):
        h5_path = os.path.join(folder, bacdive_id + ".h5")
        with h5py.File(h5_path, "r") as f:
            protein_id = f["protein_id"]
            last_item = ""
            index_list = []
            for i, item in enumerate(protein_id):
                if isinstance(item, (bytes, np.bytes_)):
                    item = item.decode("utf-8")
                if item == last_item:
                    index_list[-1].append(i)
                else:
                    index_list.append([i])
                    last_item = item
        self.protein_index[bacdive_id] = index_list
        print("resume protein index from: ", bacdive_id)

    def cp_file(self, flnm, folder, save_path):
        # 如果这两个folder是同一个，就不copy
        if folder == save_path:
            return
        src = os.path.join(folder, flnm)
        dst = os.path.join(save_path, flnm)
        shutil.copy2(src, dst)
        print(f"copy {flnm} from {folder} to {save_path}")

    def _search_current_repr(self, bacdive_id):
        exist = False
        for folder in self.spare_folders:
            flnm = bacdive_id + ".h5"
            if flnm in self.spare_folders[folder]:
                exist = True
                self._update_protein_index_by_h5(folder, bacdive_id)  # 这一步会更新 self.protein_index
                self.cp_file(flnm, folder, self.save_path)
                return exist
        return exist

    def _generate_collective_protein_repr(self, protein_path):
        aa_representation = self.aa_rep_extractor.get_collective_representation(protein_path)
        # 用index 和数据 的形式存放，每条aa representation是一条数据，全部数据加起来放在一个h5里存储。
        aa_index = self.aa_storage.append(aa_representation, protein_path)
        return aa_index

    def _generate_individual_protein_repr(self, protein_path, bacdive_id):
        """
        protein_index:
        {
            "bacdive_id": [[aa_index], [aa_index], [aa_index, aa_index], ...]  # 每个蛋白的index放在同一个列表里，用于区分
        }
        """
        self.protein_index.setdefault(bacdive_id, [])
        print(f"generate individual representation of {bacdive_id}")
        self.protein_index = self.aa_rep_extractor.get_individual_representation(protein_path=protein_path,
                                                                                 save_path=self.save_path,
                                                                                 max_shape=(None, self.aa_rep_extractor.dim),
                                                                                 protein_index=self.protein_index,
                                                                                 bacdive_id=bacdive_id)


    def _generate_index(self, bacdive_id, property_head, property_tail, aa_index=None):
        if aa_index is None:
            idx_s = "{:s},{:d},{:d}\n".format(bacdive_id, property_head, property_tail).encode()
            self.index_f.write(idx_s)
        else:
            idx_s = "{:s},{:d},{:d},{:d}\n".format(bacdive_id, property_head, property_tail, aa_index).encode()
            self.index_f.write(idx_s)

    def _valid_item(self, item):
        # if len(item["Protein_Paths"]) == 0:
        #     print(f"Warning, there is no protein file of {item['Genome Accession']}")
        #     return len(item["Protein_Paths"]) > 0
        if self.tag == "pH":
            valid_tag = item["culture_pH_optimum"]
            if valid_tag is None:
                return False
        if self.tag == "temperature":
            valid_tag = item["culture_temp_optimum"]
            if valid_tag is None:
                return False
        if self.tag == "salinity":
            valid_tag = item["NaCl_optimum"]
            if valid_tag is None:
                return False
        if self.tag == "oxygen":
            valid_tag = item["oxygen_binary"]
            if valid_tag is None:
                return False
        return True

    def rm_unit_in_property(self, property_info):
        if "culture_temp_optimum" in property_info:
            temp_str = property_info["culture_temp_optimum"]
            temp_str = temp_str.replace("°C", "")
            temp_str = temp_str.replace(" ", "")
            property_info["culture_temp_optimum"] = int(float(temp_str))
        if "NaCl_optimum" in property_info:
            nacl_str = property_info["NaCl_optimum"]
            nacl_str = nacl_str.replace("%", "")
            nacl_str = nacl_str.replace(" ", "")
            property_info["NaCl_optimum"] = int(float(nacl_str))

        return property_info

    def _generate(self):
        # i = 0
        for item in tqdm(self.overview):
            if not self._valid_item(item):
                continue
            # i += 1
            # if i < 800:
            #     continue
            time0 = time.time()
            property_info = {k: item[k] for k in self.valid_property_keys if k in item}
            property_info = self.rm_unit_in_property(property_info)
            property_head, property_tail = self._generate_property(property_info)

            # 检查h5是否已经存在，如果存在只需要生成index
            if not self.save_collective_representation:
                self.protein_index.setdefault(item["BacDive ID"], [])
                exist = self._search_current_repr(item["BacDive ID"])
                if exist:
                    self._generate_index(item["BacDive ID"], property_head, property_tail)
                    continue

            protein_path = item["Protein_Paths"][0]
            protein_path = os.path.join(self.data_dir, protein_path)

            # 检验文件是否存在
            if not os.path.exists(protein_path):
                self.no_exist_path.append(os.path.split(protein_path)[-1])
                continue

            property_time = time.time()
            print(f"property time: {(property_time - time0):.2f}")
            if self.save_collective_representation:
                aa_index = self._generate_collective_protein_repr(protein_path)
                # collective_time = time.time()
                # print(f"collective time: {collective_time - property_time}")
                self._generate_index(item["BacDive ID"], property_head, property_tail, aa_index)
                # print(f"index time: {time.time() - collective_time}")
            else:
                self._generate_individual_protein_repr(protein_path, item["BacDive ID"])
                individual_time = time.time()
                print(f"individual time: {(individual_time - property_time):.2f}")
                self._generate_index(item["BacDive ID"], property_head, property_tail)
                print(f"index time: {time.time() - individual_time}")
        if not self.save_collective_representation:
            save_json(self.protein_index, os.path.join(self.save_path, self.args.protein_index_file_name))
        print("finish generate")
        print(f"no exist path: {self.no_exist_path}")


class ESMCRepresentation:
    def __init__(self, args, **kwargs):

        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        self.cut_off = args.cut_off
        self.model = self.get_model(args.model_name)
        self.device = args.device
        self.model = self.model.to(self.device)
        self.model.eval()
        self.dim = self.model.embed.embedding_dim

    def get_model(self, model_name="esmc_300m"):
        if model_name == "esmc_300m":
            return ESMC.from_pretrained("esmc_300m")
        else:
            raise ValueError

    def load_data(self, protein_path):
        records = SeqIO.parse(protein_path, "fasta")
        for i, record in enumerate(records):
            aa_seq = "<cls> " + str(record.seq)
            if len(aa_seq) > self.cut_off:
                aa_seq = aa_seq[:self.cut_off]
            protein = ESMProtein(sequence=aa_seq)
            protein_tensor = self.model.encode(protein)
            yield protein_tensor

    def extract_representation(self, tokens):
        logits_output = self.model.logits(tokens,
                                          LogitsConfig(sequence=True, return_embeddings=True))
        return logits_output.embeddings[:, 0, :].cpu()


    def get_representation(self, protein_path):
        raise ValueError("not support")
        # 按每个file做一个循环，分batch load后取平均的repr
        self.sum_repr = torch.zeros(1, self.dim)
        self.num_tokens = 0
        self.model.eval()
        with torch.no_grad():
            for tokens in tqdm(self.load_data(protein_path)):
                tokens = tokens.to(self.device)
                repr = self.extract_representation(tokens)
                self.sum_repr += repr
                self.num_tokens += 1

        # 循环完所有的之后
        aa_representation = self.sum_repr / self.num_tokens
        # 返回representation 形式
        return aa_representation


class ESM2Representation:
    """
    测试证明这个处理5k的数据GPU会爆炸...
    """
    def __init__(self, args, **kwargs):
        self.cut_off = args.cut_off
        self.model, alphabet = self.get_model(args.model_name)
        self.device = args.device
        self.model = self.model.to(self.device)
        self.batch_size = args.batch_size
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        self.dim = self.model.embed_dim

    def get_model(self, model_name="esm2_t33_650M_UR50D"):
        if model_name == "esm2_t33_650M_UR50D":
            self.repr_layers_num = 33
            return esm.pretrained.esm2_t33_650M_UR50D()
        if model_name == "esm2_t30_150M_UR50D":
            self.repr_layers_num = 30
            return esm.pretrained.esm2_t30_150M_UR50D()
        if model_name == "esm2_t12_35M_UR50D":
            self.repr_layers_num = 12
            return esm.pretrained.esm2_t12_35M_UR50D()
        else:
            raise ValueError

    def _extract_collective_repr(self, batch_tokens, repr_layers):
        results = self.model(batch_tokens, repr_layers=[repr_layers], return_contacts=True)
        token_representations = results["representations"][repr_layers]
        # Batch, seq_len, repr_dim
        num_tokens = token_representations.shape[0]
        sum_repr = token_representations[:, 0, :].sum(0)
        return sum_repr.cpu(), num_tokens

    def _extract_individual_repr(self, batch_tokens, repr_layers):
        results = self.model(batch_tokens, repr_layers=[repr_layers], return_contacts=True)
        token_representations = results["representations"][repr_layers]
        # Batch, seq_len, repr_dim
        sum_repr = token_representations[:, 0, :]
        return sum_repr.cpu()

    def _load_data_in_batch(self, protein_path, collective=True):
        # --- 现在的写法没有定batch的上限。一个protein很长的情况也会被切在一个batch里，因此可能有内存爆炸的风险。
        records = SeqIO.parse(protein_path, "fasta")
        buffer = []
        for i, record in enumerate(records):
            if collective:
                aa_seq = "<cls> " + str(record.seq)[:-1]
                aa_id = record.id
                if len(aa_seq) > self.cut_off:
                    aa_seq = aa_seq[:self.cut_off]
                buffer.append((aa_id, aa_seq))
            else:
                aa_seq = str(record.seq)[:-1]
                aa_id = record.id
                if len(aa_seq) > self.cut_off:
                    for i in range(0, len(aa_seq), self.cut_off - 1):
                        seq_fragment = "<cls> " + aa_seq[i:i + self.cut_off]
                        buffer.append((aa_id, seq_fragment))
                else:
                    buffer.append((aa_id, "<cls> " + aa_seq))

            if (len(buffer) > self.batch_size):
                batch_labels, batch_strs, batch_tokens = self.batch_converter(buffer[:self.batch_size])
                buffer = []
                yield batch_labels, batch_strs, batch_tokens
        if len(buffer) > 0:
            batch_labels, batch_strs, batch_tokens = self.batch_converter(buffer)
            yield batch_labels, batch_strs, batch_tokens

    def get_individual_representation(self, protein_path, save_path, max_shape, protein_index, bacdive_id):
        # 保留protein的原始repr。超长的拆成了多个repr，对应的protein id保留用于识别是否是一个
        # 整个file的repr边算边存
        # 每个古菌都用一个单独的h5来存储。
        self.model.eval()
        last_id = None
        i = 0
        aa_storage = IndividualFeatureStorage(save_path, bacdive_id + ".h5", max_shape)
        with torch.no_grad():
            pbar = tqdm(self._load_data_in_batch(protein_path, collective=False))
            for batch_labels, batch_strs, batch_tokens in pbar:
                i += 1
                t0 = time.time()
                batch_tokens = batch_tokens.to(self.device)
                batch_reprs = self._extract_individual_repr(batch_tokens, repr_layers=self.repr_layers_num)   # batch, dim
                t1 = time.time()
                for protein_id, aa_representation in zip(batch_labels, batch_reprs):
                    aa_index = aa_storage.append(aa_representation, protein_id, protein_path)
                    if last_id != protein_id:
                        protein_index[bacdive_id].append([aa_index])
                        last_id = protein_id
                    else:
                        protein_index[bacdive_id][-1].append(aa_index)
                t2 = time.time()
                pbar.set_description(f"extract time: {(t1 - t0):.2f}, append time: {(t2 - t1):.2f}")
            return protein_index

    def get_collective_representation(self, protein_path):
        # 按每个file做一个循环，分batch load后取平均的repr
        self.sum_repr = torch.zeros(1, self.dim)
        self.num_tokens = 0
        self.model.eval()
        with torch.no_grad():
            for batch_labels, batch_strs, batch_tokens in tqdm(self._load_data_in_batch(protein_path, collective=True)):
                batch_tokens = batch_tokens.to(self.device)
                sum_repr, num_tokens = self._extract_collective_repr(batch_tokens, repr_layers=self.repr_layers_num)
                self.sum_repr += sum_repr
                self.num_tokens += num_tokens

        # 循环完所有的之后
        aa_representation = self.sum_repr / self.num_tokens
        # 返回representation 形式
        return aa_representation


class CollectiveFeatureStorage:
    def __init__(self, save_path, file_name="microbe.h5", max_shape=(None, 2048), chunk_size=1000, dtype='float32'):
        """
        初始化 HDF5 存储
        :param h5_path: HDF5 文件路径
        :param max_shape: 最大形状，第一维为 None 表示可扩展
        :param chunk_size: 每次扩展的块大小（行数）
        :param dtype: 数据类型
        """

        self.h5_path = os.path.join(save_path, file_name)
        self.chunk_size = chunk_size
        self.dtype = dtype

        # 确保目录存在
        Path(self.h5_path).parent.mkdir(exist_ok=True, parents=True)

        # 如果文件不存在，创建并初始化数据集
        if not os.path.exists(self.h5_path):
            with h5py.File(self.h5_path, 'w') as f:
                # 创建可扩展数据集
                f.create_dataset(
                    'microbe_features',
                    shape=(0, max_shape[1]),
                    maxshape=max_shape,
                    chunks=(chunk_size, max_shape[1]),
                    dtype=dtype,
                    compression='gzip'  # 可选压缩
                )
                # 创建存储 protein_path 的 dataset（可选）
                f.create_dataset(
                    'paths',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
                # 保存当前索引（模拟“计数器”）
                f.attrs['index'] = 0

    def append(self, aa_representation: np.ndarray, protein_path: str):
        """
        追加一条新的古菌特征
        :param aa_representation: 形状为 (d,) 或 (1, d) 的 numpy 数组
        :param protein_path: 古菌包含蛋白质文件路径（字符串）
        :return: 返回该条数据的 index
        """
        # 确保 aa_representation 是 (D,) 或 (1, D) 形状
        if aa_representation.ndim == 1:
            feature = aa_representation[None, :]  # 转为 (1, D)
        elif aa_representation.ndim == 2 and aa_representation.shape[0] == 1:
            feature = aa_representation
        else:
            raise ValueError(f"Expected 1D or (1, D) array, got {aa_representation.shape}")

        with h5py.File(self.h5_path, 'a') as f:
            current_index = int(f.attrs['index'])
            dataset = f['microbe_features']
            paths_dataset = f['paths']

            # 扩展数据集
            dataset.resize(current_index + 1, axis=0)
            dataset[current_index:current_index + 1] = feature

            # 存储路径（可选）
            paths_dataset.resize(current_index + 1, axis=0)
            paths_dataset[current_index] = str(protein_path)

            # 更新 index
            f.attrs['index'] = current_index + 1

            return current_index  # 返回当前 index


class IndividualFeatureStorage:
    def __init__(self, save_path, file_name="microbe.h5", max_shape=(None, 2048), chunk_size=1000, dtype='float32'):
        self.h5_path = os.path.join(save_path, file_name)
        self.chunk_size = chunk_size
        self.dtype = dtype

        Path(self.h5_path).parent.mkdir(exist_ok=True, parents=True)

        # 如果文件不存在，创建空文件
        if not os.path.exists(self.h5_path):
            with h5py.File(self.h5_path, 'w') as f:
                f.create_dataset(
                    'protein_features',
                    shape=(0, max_shape[1]),
                    maxshape=max_shape,
                    chunks=(chunk_size, max_shape[1]),
                    dtype=dtype,
                    compression='gzip'  # 可选压缩
                )
                # 创建存储 protein_path 的 dataset（可选）
                f.create_dataset(
                    'protein_id',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
                # 创建存储 protein_path 的 dataset（可选）
                f.create_dataset(
                    'microbe_id',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding='utf-8')
                )
                # 保存当前索引（模拟“计数器”）
                f.attrs['index'] = 0

    def append(self, aa_representation: np.ndarray, protein_id: str, microbe_id: str):
        """
        追加一条新的protein 特征数据
        :param aa_representation: 形状为 (2048,) 或 (1, 2048) 的 numpy 数组
        :param protein_path: 古菌包含蛋白质文件路径（字符串）
        :return: 返回该条数据的 index
        """
        # 确保 aa_representation 是 (D,) 或 (1, D) 形状
        if aa_representation.ndim == 1:
            feature = aa_representation[None, :]  # 转为 (1, D)
        elif aa_representation.ndim == 2 and aa_representation.shape[0] == 1:
            feature = aa_representation
        else:
            raise ValueError(f"Expected 1D or (1, D) array, got {aa_representation.shape}")

        with h5py.File(self.h5_path, 'a') as f:
            current_index = int(f.attrs['index'])
            protein_dataset = f['protein_features']
            protein_id_dataset = f['protein_id']
            microbe_id_dataset = f['microbe_id']

            # 扩展数据集
            protein_dataset.resize(current_index + 1, axis=0)
            protein_dataset[current_index:current_index + 1] = feature

            # 存储路径（可选）
            protein_id_dataset.resize(current_index + 1, axis=0)
            protein_id_dataset[current_index] = str(protein_id)

            microbe_id_dataset.resize(current_index + 1, axis=0)
            microbe_id_dataset[current_index] = str(microbe_id)

            # 更新 index
            f.attrs['index'] = current_index + 1

            return current_index  # 返回当前 index
