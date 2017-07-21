/project/Tf2motif/home/tools/bin/meme-chip \
    -oc $CRBM_OUTPUT_DIR/meme_oct4 \
    -meme-nmotifs 10 -meme-minw 15 -meme-maxw 15 ../data/stemcells_train.fa
/project/Tf2motif/home/tools/bin/meme-chip \
    -oc $CRBM_OUTPUT_DIR/meme_mafk \
    -meme-nmotifs 10 -meme-minw 15 -meme-maxw 15 ../data/fibroblast_train.fa

cat ../data/stemcells_train.fa ../data/fibroblast_train.fa > \
 ../data/joint_train.fa
cat ../data/stemcells_test.fa ../data/fibroblast_test.fa > \
 ../data/joint_test.fa

/project/Tf2motif/home/tools/bin/meme-chip \
    -oc $CRBM_OUTPUT_DIR/meme_joint_om \
    -meme-nmotifs 10 -meme-minw 15 -meme-maxw 15 ../data/joint_train.fa

cells=(K562 GM12878 HepG2 Hela H1hesc)

for cell in ${cells[*]}; do
    /project/Tf2motif/home/tools/bin/meme-chip \
        -oc $CRBM_OUTPUT_DIR/meme_jund_$cell \
        -meme-nmotifs 10 -meme-minw 15 -meme-maxw 15 \
        -meme-nmotifs 10 -meme-minw 15 -meme-maxw 15 \
        ../data/jund_data/${cell}_only_train.fa
done
