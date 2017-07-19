import sys
sys.path.append("../code")

from getData import splitTrainingTest, readSeqsFromFasta

train_test_ratio = 0.1
########################################################

# get the data
splitTrainingTest('../data/stemcells.fa', train_test_ratio, 4000)
splitTrainingTest('../data/fibroblast.fa', train_test_ratio, 4000)

splitTrainingTest('../data/jund_data/GM12878_only.fasta', train_test_ratio, 4000)
splitTrainingTest('../data/jund_data/K562_only.fasta', train_test_ratio, 4000)
splitTrainingTest('../data/jund_data/HepG2_only.fasta', train_test_ratio, 4000)
splitTrainingTest('../data/jund_data/Hela_only.fasta', train_test_ratio, 4000)
splitTrainingTest('../data/jund_data/H1hesc_only.fasta', train_test_ratio, 4000)
