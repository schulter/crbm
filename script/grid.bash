
lrates=(0.05 0.1 0.2 0.5 0.7)
sparsities=(0.0 0.1 0.5)
batchsizes=(10 20 50)
rhos=(0.001 0.01 0.033 0.1)
spmethod=('relu' 'entropy')
epochs=(50 100 200)

lrates=(0.05)

for lr in ${lrates[*]}; do
for lam in ${sparsities[*]}; do
for bs in ${batchsizes[*]}; do
for rho in ${rhos[*]}; do
    for spm in ${spmethod[*]}; do
        for epoch in ${epochs[*]}; do
            if [ ! -f $CRBM_OUTPUT_DIR/grom/gridsearch_oct4_mafk_row.py_$lr_$lam_$bs_$rho_$spm_$epoch.csv ]; then
            echo "python gridsearch_oct4_mafk_row.py $lr $lam $bs $rho $spm $epoch"
            mxqsub -t 24h -j 8 -m 4G \
                --stdout $CRBM_OUTPUT_DIR/log/gridsearch_stdout.log \
                --stderr $CRBM_OUTPUT_DIR/log/gridsearch_stderr.log \
                -N gridsearch_om \
                bash gridsearch_oct4_mafk_row_wrapper.bash $lr \
                $lam $bs $rho $spm $epoch &
            sleep 2
            fi
        done
    done
done
done
done
done

cat $CRBM_OUTPUT_DIR/grom/*.csv > $CRBM_OUTPUT_DIR/om_gridsearch.csv
