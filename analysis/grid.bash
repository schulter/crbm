
lrates=(0.01 0.05 0.1 0.2 0.5)
sparsities=(0.0 0.1 0.5)
batchsizes=(10 20 50)
rhos=(0.001 0.01 0.033)
spmethod=('entropy')
epochs=(100)

#lrates=(0.05)

for lr in ${lrates[*]}; do
for lam in ${sparsities[*]}; do
for bs in ${batchsizes[*]}; do
for rho in ${rhos[*]}; do
    for spm in ${spmethod[*]}; do
        for epoch in ${epochs[*]}; do
            #break
            if [ ! -f $CRBM_OUTPUT_DIR/grom/gridsearch_oct4_mafk_row.py_$lr_$lam_$bs_$rho_$spm_$epoch.csv ]; then
            echo "python gridsearch_oct4_mafk_row.py $lr $lam $bs $rho $spm $epoch"
            mxqsub -t 24h -j 4 \
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

for lr in ${lrates[*]}; do
for lam in ${sparsities[*]}; do
for bs in ${batchsizes[*]}; do
for rho in ${rhos[*]}; do
    for spm in ${spmethod[*]}; do
        for epoch in ${epochs[*]}; do
            echo "python gridsearch_jund_row.py $lr $lam $bs $rho $spm $epoch"
            mxqsub -t 24h -j 4 \
                --stdout $CRBM_OUTPUT_DIR/log/gridsearch_stdout.log \
                --stderr $CRBM_OUTPUT_DIR/log/gridsearch_stderr.log \
                -N gridsearch_jund \
                bash gridsearch_jund_row_wrapper.bash $lr \
                $lam $bs $rho $spm $epoch &
            sleep 2
        done
    done
done
done
done
done

#afterwards collect all rows into one csv
#cat $CRBM_OUTPUT_DIR/grom/*.csv > $CRBM_OUTPUT_DIR/om_gridsearch.csv
#cat $CRBM_OUTPUT_DIR/grjund/*.csv > $CRBM_OUTPUT_DIR/jund_gridsearch.csv
