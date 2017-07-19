junddir=/project/Tf2motif/home/source/crbm/data/jund_data

cd $junddir

bedtools intersect -a GM12878_ENCFF002COV.bed -b \
    H1hesc_ENCFF001UBM.bed HeLa-S3_ENCFF001VIQ.bed \
    HepG2_ENCFF002CUF.bed K562_ENCFF001VRL.bed -wa -v > \
    GM12878_only.bed
bedtools intersect -a H1hesc_ENCFF001UBM.bed -b \
    GM12878_ENCFF002COV.bed HeLa-S3_ENCFF001VIQ.bed \
    HepG2_ENCFF002CUF.bed K562_ENCFF001VRL.bed -wa -v > \
    H1hesc_only.bed
bedtools intersect -a HeLa-S3_ENCFF001VIQ.bed -b \
    H1hesc_ENCFF001UBM.bed GM12878_ENCFF002COV.bed \
    HepG2_ENCFF002CUF.bed K562_ENCFF001VRL.bed -wa -v > \
    Hela_only.bed
bedtools intersect -a HepG2_ENCFF002CUF.bed -b \
    H1hesc_ENCFF001UBM.bed HeLa-S3_ENCFF001VIQ.bed \
    GM12878_ENCFF002COV.bed K562_ENCFF001VRL.bed -wa -v > \
    HepG2_only.bed
bedtools intersect -a K562_ENCFF001VRL.bed -b \
    H1hesc_ENCFF001UBM.bed HeLa-S3_ENCFF001VIQ.bed \
    HepG2_ENCFF002CUF.bed GM12878_ENCFF002COV.bed -wa -v > \
    K562_only.bed

cells=(GM12878 H1hesc Hela HepG2 K562)

for cell in ${cells[*]}; do
    python ../../script/extract_fasta_from_bed.py ./ ${cell}_only.bed
done
