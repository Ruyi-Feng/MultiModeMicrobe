
from Bio import SeqIO
from exp.data_provider import ESM2Representation, DataProvider
from config import data_provider_args


def test_aa_seq_load():
    path = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples\\data\\protein_faa_files\\GCA_000003925.1.faa"
    records = SeqIO.parse(path, "fasta")

    max_len = 0
    for record in records:
        max_len = max(max_len, len(record.seq))

    print(max_len)


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
