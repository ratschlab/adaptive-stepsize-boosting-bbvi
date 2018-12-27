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
OUTDIR=${TD}/1d
#OUTDIR=${TD}/2d

## fw_variant - fixed, line_search, fc, adafw
FW_VAR=fixed

OUTDIR=${OUTDIR}/${FW_VAR}

# Clean output directory
echo -e "${HRED}Cleaning data $@${NC}" ; 
echo_and_run \
  mkdir -p ${OUTDIR}

## **********Run the mixture model************* ##
## exp - mixture, mixture_2d
echo -e "${HGREEN} Running Boosted BBVI $@${NC}" ; 
i=1000
#echo_and_run \
#  python ${SRC}/scripts/mixture_model_relbo.py \
#    --relbo_reg 1.0 \
#    --relbo_anneal linear \
#    --exp mixture \
#    --fw_variant ${FW_VAR} \
#    --outdir=${OUTDIR} \
#    --n_fw_iter=50 \
#    --LMO_iter=1000 \
#    --n_monte_carlo_samples ${i} \
#    --n_line_search_iter 25

## **********Running test script************* ##
echo -e "${HPURPLE} Running test $@${NC}" ; 
iters=( 5  10 )
#for i in "${iters[@]}"
#do
#  echo_and_run \
#    python ${SRC}/tests/test_step_size.py \
#            --exp mixture \
#            --outdir=${OUTDIR} \
#            --fw_variant=${FW_VAR} \
#            --n_monte_carlo_samples ${i} \
#            --n_line_search_iter 25
#done

#echo_and_run \
#  python ${SRC}/tests/test_gap.py \
#    --n_fw_iter=10 \
#    --LMO_iter=1000 \
#    --n_monte_carlo_samples 1000 \

## **********Plotting results************* ##
echo -e "${HCYAN} Plotting results $@${NC}" ; 

#echo_and_run \
#  python ${SRC}/plots/plot_single_mixture.py \
#    --outdir=stdout \
#    --target=${OUTDIR}/target_dist.npz \
#    --qt=${OUTDIR}/qt_latest.npz,${OUTDIR}/qt_iter20.npz \
#    --labels=latest,iter20 \
##    --grid2d
    #--qt=${OUTDIR}/qt_latest.npz,${OUTDIR}/qt_iter2.npz,${OUTDIR}/qt_iter0.npz \
    #--labels=latest,iter2,init \

echo_and_run \
  python ${SRC}/plots/plot_mixture_comps.py \
    --outdir=stdout \
    --qt=${OUTDIR}/qt_iter5.npz,${OUTDIR}/qt_iter3.npz \
    --labels=iter5,iter3

# outdir = stdout, ${DOC}/plots
#all_runs=$(ls ${OUTDIR}/gradients/*)
#all_runs_comma=$(echo ${all_runs} | sed "s/\s/,/g")
## metric = gamma,E_s,E_q
#METRIC=E_q
#echo_and_run \
#  python ${SRC}/plots/plot_line_search.py \
#    --metric=${METRIC} \
#    --outdir=stdout \
#    --extra=0.6 \
#    --outfile=test_e_q.png \
#    --runs=${all_runs_comma}
