import os
from utils.json_loader import load_json
from config import get_included_property
from tqdm import tqdm
import torch
from Bio import SeqIO


import esm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class DataProvider:
    def __init__(self, data_dir, **kwargs):
        self.aa_rep_extractor = AARepresentation()
        self.data_dir = data_dir
        overview = os.path.join(data_dir, "microbex_data_with_protein.json")
        self.overview = load_json(overview)

        self.valid_property_keys = get_included_property()
        self.data_length = len(self.overview)
        self._generate()

    def _init_data(self):
        self.property_f = open(os.path.join(self.data_dir, "property.bin"), 'ab+')
        self.property_head = 0
        self.aa_f = open(os.path.join(self.data_dir, "aa_representations.bin"), 'ab+')
        self.aa_head = 0
        self.index_f = open(os.path.join(self.data_dir, "index.txt"), 'ab+')

    def _generate_property(self, property_info):
        head = self.property_head
        property_info = str(property_info)
        property_info = property_info.encode()
        write_len = self.property_f.write(property_info)
        tail = head + write_len
        self.property_head = tail
        return head, tail

    def _generate_protein(self, protein_path):
        aa_representation = self.aa_rep_extractor.get_representation(protein_path)
        aa_representation = str(aa_representation).encode()
        head = self.aa_head
        write_len = self.aa_f.write(aa_representation)
        tail = head + write_len
        self.aa_head = tail
        return head, tail

    def _generate_index(self, bacdive_id, property_head, property_tail, aa_head, aa_tail):
        idx_s = "{:s},{:d},{:d},{:d},{:d}\n".format(bacdive_id, property_head, property_tail, aa_head, aa_tail).encode()
        self.index_f.write(idx_s)

    def _generate(self):
        for item in tqdm(self.overview):
            property_info = item[self.valid_property_keys]
            property_head, property_tail = self._generate_property(property_info)
            protein_path = item["Protein_Paths"][0]
            aa_head, aa_tail = self._generate_protein(protein_path)
            self._generate_index(item["BacDive ID"], property_head, property_tail, aa_head, aa_tail)
        print("finish generate")


class AARepresentation:
    def __init__(self, args, **kwargs):
        self.model, alphabet = self.get_model(args.model_name)
        self.batch_size = args.batch_size
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        self.dim = self.model.embed_dim

    def get_model(self, model_name="esm2_t33_650M_UR50D"):
        if model_name == "esm2_t33_650M_UR50D":
            return esm.pretrained.esm2_t33_650M_UR50D()
        else:
            raise ValueError

    def extract_representation(self, batch_tokens, repr_layers):
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        # Batch, seq_len, repr_dim
        num_tokens = token_representations.shape[0]
        sum_repr = token_representations[:, 0, :].sum(0)
        return sum_repr, num_tokens

    def load_data_in_batch(self, protein_path):
        records = SeqIO.parse("file.faa", "fasta")
        buffer = []
        for i, record in enumerate(records):
            aa_seq = "<cls> " + str(record.seq)
            aa_id = record.id
            buffer.append((aa_id, aa_seq))
            if ((i % self.batch_size) == 0) and (len(buffer) > 0):
                batch_labels, batch_strs, batch_tokens = self.batch_converter(buffer)
                buffer = []
                yield batch_labels, batch_strs, batch_tokens
        if len(buffer) > 0:
            batch_labels, batch_strs, batch_tokens = self.batch_converter(buffer)
            yield batch_labels, batch_strs, batch_tokens


    def get_representation(self, protein_path):
        # 按每个file做一个循环，分batch load后取平均的repr
        self.sum_repr = torch.zeros(1, self.dim)
        self.num_tokens = 0
        batch_labels, batch_strs, batch_tokens = self.load_data_in_batch(protein_path)

        # 用模型提取representation
        sum_repr, num_tokens = self.extract_representation(batch_tokens, repr_layers=[33])

        # 返回representation list的形式
        return aa_representation
