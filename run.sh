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
#OUTDIR_ref=${TD}/mixture_model_variants

## fw_variant - fixed, line_search, fc, adafw
FW_VAR=adafw
FW_VAR_ref=fixed

OUTDIR_ref=${OUTDIR}/${FW_VAR_ref}
OUTDIR=${OUTDIR}/${FW_VAR}

# Clean output directory
echo -e "${HRED} Making data directory $@${NC}" ; 
echo_and_run \
  mkdir -p ${OUTDIR}

## **********Run the mixture model************* ##
## exp - mixture, mixture_2d
echo -e "${HGREEN} Running Boosted BBVI $@${NC}" ; 
#i=1000
#echo_and_run \
#  python ${SRC}/scripts/mixture_model_relbo.py \
#    --exp mixture \
#    --fw_variant ${FW_VAR} \
#    --outdir=${OUTDIR} \
#    --n_fw_iter=2 \
#    --LMO_iter=1000 \
#    --n_monte_carlo_samples ${i} \
#    --n_line_search_iter 25
    #--relbo_reg 1.0 \
    #--relbo_anneal linear \

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

echo_and_run \
  python ${SRC}/plots/plot_single_mixture.py \
    --outdir=stdout \
    --target=${OUTDIR}/target_dist.npz \
    --qt=${OUTDIR_ref}/qt_iter20.npz,${OUTDIR}/qt_iter20.npz \
    --labels=iter20_${FW_VAR_ref},iter20_${FW_VAR} \
    #--grid2d
    #--qt=${OUTDIR}/qt_latest.npz,${OUTDIR}/qt_iter2.npz,${OUTDIR}/qt_iter0.npz \
    #--labels=latest,iter2,init \

echo_and_run \
  python ${SRC}/plots/plot_mixture_comps.py \
    --outdir=stdout \
    --qt=${OUTDIR}/qt_iter5.npz \
    --label=iter1

echo_and_run \
  python ${SRC}/plots/plot_mixture_comps.py \
    --outdir=stdout \
    --qt=${OUTDIR_ref}/qt_iter5.npz \
    --label=iter1

echo_and_run \
  python ${SRC}/plots/plot_losses.py \
    --elbos_files=${OUTDIR}/elbos.csv,${OUTDIR_ref}/elbos.csv \
    --relbos_files=${OUTDIR}/relbos.csv,${OUTDIR_ref}/relbos.csv \
    --kl_files=${OUTDIR}/kl.csv,${OUTDIR_ref}/kl.csv \
    --times_files=${OUTDIR}/times.csv,${OUTDIR_ref}/times.csv \
    --labels=${FW_VAR},${FW_VAR_ref}

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
