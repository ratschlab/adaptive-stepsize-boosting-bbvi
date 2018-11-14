#!/bin/bash

#Black        0;30     Dark Gray     1;30
#Red          0;31     Light Red     1;31
#Green        0;32     Light Green   1;32
#Brown/Orange 0;33     Yellow        1;33
#Blue         0;34     Light Blue    1;34
#Purple       0;35     Light Purple  1;35
#Cyan         0;36     Light Cyan    1;36
#Light Gray   0;37     White         1;37
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

if [ -z "$ROOT" ]; then
  ROOT=/home/saurav
fi
TD=${ROOT}/thesis_data
T=${ROOT}/thesis
SRC=${T}/boosting-bbvi-private/boosting_bbvi
DOC=${T}/boosting-bbvi-private/docs
OUTDIR=${TD}/line_search_test
#OUTDIR=${TD}/test
#OUTDIR=${TD}/1d
#OUTDIR=${TD}/2d

# Clean output directory
echo -e "${HRED}Cleaning data $@${NC}" ; 
echo_and_run \
  mkdir -p ${OUTDIR}

## Run the mixture model
## exp - mixture, mixture_2d
## fw_variant - fixed, line_search, fc
#echo -e "${HGREEN} Running Boosted BBVI $@${NC}" ; 
#i=10
#echo_and_run \
#  python ${SRC}/scripts/mixture_model_relbo.py \
#    --relbo_reg 1.0 \
#    --relbo_anneal linear \
#    --exp mixture \
#    --fw_variant line_search \
#    --outdir=${OUTDIR} \
#    --n_fw_iter=10 \
#    --LMO_iter=20 \
#    --n_line_search_samples ${i} \
#    --n_line_search_iter 25

## Running test script
#echo -e "${HPURPLE} Running test $@${NC}" ; 
#iters=( 20  50 100 )
#for i in "${iters[@]}"
#for i in {10..50..25}
#do
#  echo_and_run \
#    python ${SRC}/tests/test_line_search.py \
#            --exp mixture \
#            --outdir=${OUTDIR} \
#            --n_line_search_samples ${i} \
#            --n_line_search_iter 25
#done

## Create single mixture plot
#echo -e "${HCYAN} Plotting results $@${NC}" ; 
#OUTDIR=${TD}/1d

#echo_and_run \
#  python ${SRC}/plots/plot_single_mixture.py \
#    --outdir=stdout \
#    --target=${OUTDIR}/target_dist.npz \
#    --qt=${OUTDIR}/qt_latest.npz,${OUTDIR}/qt_iter5.npz \
#    --labels=latest,iter5 \
#    #--grid2 d

# metric = gamma,E_s,E_q
# outdir = stdout, ${DOC}/plots
all_runs=$(ls ${OUTDIR}/gradients/*)
all_runs_comma=$(echo ${all_runs} | sed "s/\s/,/g")
echo_and_run \
  python ${SRC}/plots/plot_line_search.py \
    --metric=gamma \
    --extra=0.6 \
    --outdir=stdout \
    --outfile=test_e_q.png \
    --runs=${all_runs_comma}
#    --runs=${OUTDIR}/gradients/line_search_samples_10.npy,${OUTDIR}/gradients/line_search_samples_35.npy,${OUTDIR}/gradients/line_search_samples_5.npy,${OUTDIR}/gradients/line_search_samples_50.npy,${OUTDIR}/gradients/line_search_samples_20.npy
