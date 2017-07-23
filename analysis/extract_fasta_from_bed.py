import pandas as pd
import gzip
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import sys
import os

datadir = sys.argv[1]
print(datadir)
filename = sys.argv[2]

h=open(datadir+ "hg19.fa")
genome = SeqIO.to_dict(SeqIO.parse(h, "fasta"))

filename

filename = filename.split(".")[0]
print(filename)

bed = pd.read_csv(datadir + filename +".bed", sep="\t", header = None,
        usecols = [0, 1, 2, 4], 
        names = ["chr", "start", "end", "cnt"])


#remove all random chromosomes
bed = bed[bed.chr.apply(lambda el: False if len(el.split("_"))>1 else True)]

bed.sort_values(by="cnt", inplace = True, ascending=False)

seqs = list()
for row in bed.iterrows():
    start = (row[1].end-row[1].start)//2 + row[1].start -100
    end = (row[1].end-row[1].start)//2 + row[1].start + 100
    id_ = row[1].chr+":"+str(start)+"-"+str(end)
    seqs.append(SeqRecord( \
            genome[row[1].chr].seq[start:end], \
            id = id_, name = '', description=''))

print("Writing " + filename + ".fasta")
SeqIO.write(seqs, datadir + filename + ".fasta", "fasta")
