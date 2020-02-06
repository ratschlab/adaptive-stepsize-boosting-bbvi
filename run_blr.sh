#!/bin/bash

RED='\033[0;31m'
HRED='\033[1;41m'
BROWN='\033[2;33m'
HCYAN='\033[1;46m'
GREEN='\033[0;32m'
HGREEN='\033[1;42m'
HPURPLE='\033[1;45m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_and_run() { 
  echo -e "${GREEN}\$ $@${NC}" ; 
  "$@" ; 
}

echo_run_and_log() { 
  logfile="$1"; shift
  echo -e "${BROWN}\$ $@${NC}" ; 
  echo "$@" > ${logfile} ; 
  "$@" 2>&1 | tee -a ${logfile} ; 
  # remove unreadable characters from log 
  sed -i $'s/[^[:print:]\t]//g' ${logfile}
}

# exit on fail
set -e

if [ -z "$UHOME" ]; then
  UHOME=/home/saurav
fi

SRC=${UHOME}/thesis/boosting-bbvi-private/boosting_bbvi
OUTDIR=${UHOME}/thesis_data/blr/eicu


base_dist=mvl
iter0=vi
seed=42

DDIR_ref=${OUTDIR}/${FW_VAR_ref}

mkdir -p ${DDIR_ref}

bsub -n 1 -W 4:00 -J "bbvi" -oo ${DDIR_ref}/run.log -R "rusage[mem=32000]" \
  python ${SRC}/scripts/bayesian_logistic_regression.py \
    --exp chem \
    --fw_variant ${FW_VAR_ref} \
    --base_dist ${base_dist} \
    --outdir ${DDIR_ref} \
    --datapath ${SRC}/data/chem \
    --n_fw_iter ${ITER} \
    --seed ${seed} \
    --LMO_iter 1000 \
    --iter0 ${iter0} \
    --n_monte_carlo_samples 1000

#FW_VAR=line_search
#
#DDIR=${OUTDIR}/${FW_VAR}
#
#bsub -n 1 -W 8:00 -J "blr_l" -oo ${DDIR}/run.log -R "rusage[mem=128000]" \
#  python ${SRC}/scripts/bayesian_logistic_regression.py \
#    --exp chem \
#    --fw_variant ${FW_VAR} \
#    --base_dist ${base_dist} \
#    --outdir ${DDIR} \
#    --datapath ${SRC}/data/chem \
#    --n_fw_iter ${ITER} \
#    --seed ${seed} \
#    --linit_fixed 0.001 \
#    --LMO_iter 1000 \
#    --iter0 ${iter0} \
#    --n_line_search_iter 25 \
#    --n_monte_carlo_samples 1000

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

