export PYTH=/home/kopp/anaconda2/bin/python
export CRBM_OUTPUT_DIR=/project/Tf2motif/home/crbm_suppl_overleaf/
#export THEANO_FLAGS='device=cpu,force_device=True'
export THEANO_FLAGS='floatX=float32, device=cpu,openmp=True, force_device=True, \
    optimizer=fast_run, blas.ldflags=-lopenblas -lgfortran -lpthread'
lr=$1
lam=$2
bs=$3
rho=$4
spm=$5
epoch=$6
#$PYTH -c "x='Hello'; print(x)"
#$PYTH -c "import theano; print(theano.__version__)"
$PYTH gridsearch_jund_row.py $lr $lam $bs $rho $spm $epoch
