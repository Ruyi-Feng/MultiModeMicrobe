
from Bio import SeqIO
from exp.data_provider import ESM2Representation, DataProvider
from config import data_provider_args
import os


def test_aa_seq_load():
    path = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples\\data\\protein_faa_files\\"
    files = os.listdir(path)
    c1024 = 0
    c8192 = 0
    max_len = 0
    total = 0

    for file in files:
        file_nm = os.path.join(path, file)
        records = SeqIO.parse(file_nm, "fasta")

        for record in records:
            max_len = max(max_len, len(record.seq))
            if len(record.seq) > 1024:
                c1024 += 1
            if len(record.seq) > 5000:
                c8192 += 1
            total += 1

    print(f"max_len: {max_len}, c1024: {c1024}, percent: {c1024 / total}, c8192: {c8192}, percent: {c8192 / total}")


def test_aa_provider():
    protein_path = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples\\data\\protein_faa_files\\GCA_000003925.1.faa"
    args = data_provider_args()
    aarepr = ESM2Representation(args)
    repr = aarepr.get_representation(protein_path)
    print(repr)


def test_data_provider():
    data_dir = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples\\"
    save_path = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\train_data"
    DataProvider(data_dir, save_path)
