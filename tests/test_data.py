
from Bio import SeqIO



def test_aa_seq_load():
    path = f"C:\\Users\\User\\WorkSpace\\data\\Zhiling\\aaseq_microbe\\examples\\data\\protein_faa_files\\GCA_000003925.1.faa"
    records = SeqIO.parse(path, "fasta")

    for record in records:
        print(record)


test_aa_seq_load()
