junddir=/project/Tf2motif/home/source/crbm/data/jund_data
cells=(GM12878 H1hesc HeLa-S3 HepG2 K562)

cd $junddir

for  cell in ${cells[*]}; do
    sort -k 8 -r ${cell}_ENC*.bed |  head -n 4000 > ${cell}_stringent.bed 
done

for  cell in ${cells[*]}; do
    rem=(${cells[*]/$cell})
    remsuf=${rem[@]/%/_stringent.bed}
    echo "bedtools intersect -v -wa -a ${cell}_stringent.bed -b \
        ${rem[@]/%/_stringent.bed} > \
        ${cell}_only.bed"
    bedtools intersect -v -wa -a ${cell}_stringent.bed -b \
        ${rem[@]/%/_stringent.bed} > \
        ${cell}_only.bed

    bedtools intersect -u -wa -a ${cell}_stringent.bed -b \
        ${rem[@]/%/_stringent.bed} > \
        ${cell}_any.bed

    python ../../script/extract_fasta_from_bed.py ./ ${cell}_only.bed
done
for  cell in ${cells[*]}; do
    python ../../script/extract_fasta_from_bed.py ./ ${cell}_any.bed
done

