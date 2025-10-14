
from Bio import SeqIO
from exp.data_provider import AARepresentation
from config import data_provider_args


def test_aa_seq_load():
    path = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples\\data\\protein_faa_files\\GCA_000003925.1.faa"
    records = SeqIO.parse(path, "fasta")

    for record in records:
        print(record)


def test_aa_provider():
    protein_path = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples\\data\\protein_faa_files\\GCA_000003925.1.faa"
    args = data_provider_args()
    aarepr = AARepresentation(args)
    repr = aarepr.get_representation(protein_path)
    print(repr)
