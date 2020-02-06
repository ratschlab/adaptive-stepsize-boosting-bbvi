#!/bin/bash

RED='\033[0;31m'
HRED='\033[1;41m'
BROWN='\033[2;33m'
HCYAN='\033[1;46m'
HGREEN='\033[1;42m'
HPURPLE='\033[1;45m'
NC='\033[0m' # No Color

echo_and_run() { 
  echo -e "${BROWN}\$ $@${NC}" ; 
  "$@" ; 
}

# exit on fail
set -e

if [ -z "$UHOME" ]; then
  UHOME=/home/saurav
fi
TD=${UHOME}/thesis_data
T=${UHOME}/thesis
SRC=${T}/boosting-bbvi-private/boosting_bbvi
DOC=${T}/boosting-bbvi-private/docs
#OUTDIR=${TD}/line_search_test
#OUTDIR=${TD}/test
OUTDIR=${TD}/bmf_syn
#OUTDIR_ref=${TD}/mixture_model_variants

# Number of FW iterations
ITER=10 # SET THIS

## fw_variant - fixed, line_search, fc, adafw
FW_VAR_ref=fixed
FW_VAR=adafw

#***************** Running commands *****************#

seed=42

DDIR_ref=${OUTDIR}/${FW_VAR_ref}
DDIR=${OUTDIR}/${FW_VAR}

#if [ $1 == 'elbo' ]
#then
#  echo_and_run \
#    python ${SRC}/scripts/bmf_elbo.py \
#      --exp cbcl \
#      --outdir stdout \
#      --datapath ${SRC}/data/cbcl \
#      --VI_iter 100 \
#      --D 3 \
#      --seed ${seed}
#fi

if [ $1 == 'fixed' ]
then
  mkdir -p ${DDIR_ref}

  echo_and_run \
    python ${SRC}/scripts/matrix_factorization.py \
      --exp synthetic \
      --fw_variant ${FW_VAR_ref} \
      --outdir ${DDIR_ref} \
      --n_fw_iter ${ITER} \
      --LMO_iter 50 \
      --norestore \
      --D 3 \
      --seed ${seed}
fi

if [ $1 == 'restart' ]
then
  FW_VAR=ada_afw
  DDIR=${OUTDIR}/${FW_VAR}
  mkdir -p ${DDIR}
  mkdir -p ${DDIR}_restart

  echo_and_run \
    python ${SRC}/scripts/matrix_factorization.py \
      --exp synthetic \
      --fw_variant ${FW_VAR} \
      --outdir ${DDIR}_restart \
      --n_fw_iter 5 \
      --linit_fixed 1 \
      --damping_adafw 0.5 \
      --exp_adafw 2 \
      --norestore \
      --LMO_iter 100 \
      --D 3 \
      --seed ${seed}

  #echo_and_run \
  #  python ${SRC}/scripts/matrix_factorization.py \
  #    --exp synthetic \
  #    --fw_variant ${FW_VAR} \
  #    --outdir ${DDIR} \
  #    --n_fw_iter 10 \
  #    --linit_fixed 1 \
  #    --damping_adafw 0.5 \
  #    --exp_adafw 2 \
  #    --norestore \
  #    --LMO_iter 100 \
  #    --D 3 \
  #    --seed ${seed}

  echo_and_run \
    python ${SRC}/scripts/matrix_factorization.py \
      --exp synthetic \
      --fw_variant ${FW_VAR} \
      --outdir ${DDIR}_restart \
      --n_fw_iter 5 \
      --linit_fixed 1 \
      --damping_adafw 0.5 \
      --exp_adafw 2 \
      --restore \
      --LMO_iter 100 \
      --D 3 \
      --seed ${seed}
fi

if [ $1 == 'line_search' ]
then
  FW_VAR=line_search
  DDIR=${OUTDIR}/${FW_VAR}
  mkdir -p ${DDIR}

  echo_and_run \
    python ${SRC}/scripts/matrix_factorization.py \
      --exp synthetic \
      --fw_variant ${FW_VAR} \
      --outdir ${DDIR} \
      --n_fw_iter ${ITER} \
      --LMO_iter 100 \
      --linit_fixed 1e-8 \
      --D 3 \
      --seed ${seed}
fi

if [ $1 == 'adafw' ]
then
  mkdir -p ${DDIR}

  echo_and_run \
    python ${SRC}/scripts/matrix_factorization.py \
      --exp synthetic \
      --fw_variant ${FW_VAR} \
      --outdir ${DDIR} \
      --n_fw_iter ${ITER} \
      --linit_fixed 1 \
      --damping_adafw 0.5 \
      --exp_adafw 2 \
      --LMO_iter 100 \
      --D 3 \
      --seed ${seed}
fi

if [ $1 == 'ada_pfw' ]
then
  FW_VAR=ada_pfw
  DDIR=${OUTDIR}/${FW_VAR}
  mkdir -p ${DDIR}

  echo_and_run \
    python ${SRC}/scripts/matrix_factorization.py \
      --exp synthetic \
      --fw_variant ${FW_VAR} \
      --outdir ${DDIR} \
      --n_fw_iter ${ITER} \
      --linit_fixed 1 \
      --damping_adafw 0.5 \
      --exp_adafw 2 \
      --LMO_iter 100 \
      --D 3 \
      --seed ${seed}
fi

if [ $1 == 'ada_afw' ]
then
  FW_VAR=ada_afw
  DDIR=${OUTDIR}/${FW_VAR}
  mkdir -p ${DDIR}

  echo_and_run \
    python ${SRC}/scripts/matrix_factorization.py \
      --exp synthetic \
      --fw_variant ${FW_VAR} \
      --outdir ${DDIR} \
      --n_fw_iter ${ITER} \
      --linit_fixed 1 \
      --damping_adafw 0.5 \
      --exp_adafw 2 \
      --LMO_iter 100 \
      --D 3 \
      --seed ${seed}
fi


#tau=2.0
#eta=0.99
#linit_fixed=10.0
#FW_VAR=adafw
#
#DDIR=${OUTDIR}/${FW_VAR}_loss
#
#mkdir -p ${DDIR}
#
##bsub -sp 40 -n 1 -W 4:00 -J "bbbvi" -oo ${DDIR}/run.log -R "rusage[mem=64000]" \
#bsub -n 1 -W 4:00 -J "bbvi" -oo ${DDIR}/run.log -R "rusage[mem=40000]" \
#  python ${SRC}/scripts/bayesian_logistic_regression.py \
#    --exp chem \
#    --fw_variant ${FW_VAR} \
#    --base_dist ${base_dist} \
#    --outdir ${DDIR} \
#    --datapath ${SRC}/data/chem \
#    --n_fw_iter ${ITER} \
#    --seed ${seed} \
#    --LMO_iter 1000 \
#    --linit fixed \
#    --distance_metric kl \
#    --adafw_MAXITER 10 \
#    --linit_fixed ${linit_fixed} \
#    --damping_adafw ${eta} \
#    --exp_adafw ${tau} \
#    --iter0 ${iter0} \
#    --n_monte_carlo_samples 1000

  #"python ${SRC}/scripts/bayesian_logistic_regression.py --exp chem --fw_variant ${FW_VAR} --base_dist ${base_dist} --outdir ${DDIR} --datapath ${SRC}/data/chem --n_fw_iter ${ITER} --seed ${seed} --LMO_iter 1000 --linit fixed --distance_metric kl --adafw_MAXITER 10 --linit_fixed ${linit_fixed} --damping_adafw ${eta} --exp_adafw ${tau} --iter0 ${iter0} --n_monte_carlo_samples 1000"
