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
DRED='\033[1;31m'
BROWN='\033[2;33m'
HCYAN='\033[1;46m'
HGREEN='\033[1;42m'
HPURPLE='\033[1;45m'
NC='\033[0m' # No Color

echo_and_run() { 
  echo -e "${BROWN}\$ $@${NC}" ; 
  "$@" ; 
}

if [ -z "$ROOT" ]; then
  ROOT=/home/saurav
fi
TD=${ROOT}/thesis_data
T=${ROOT}/thesis
SRC=${T}/boosting-bbvi-private/boosting_bbvi
OUTDIR=${TD}/test

## Clean output directory
#echo -e "${HPURPLE}Cleaning data $@${NC}" ; 
#echo_and_run \
#  rm -rf ${OUTDIR}/*
#
## Run the mixture model
## exp - mixture, mixture_2d
#echo -e "${HGREEN} Running Boosted BBVI $@${NC}" ; 
#echo_and_run \
#  python ${SRC}/scripts/mixture_model_relbo.py \
#    --relbo_reg 1.0 \
#    --relbo_annlear linear \
#    --exp mixture \
#    --fw_variant fixed \
#    --outdir=${OUTDIR} \
#    --n_fw_iter=10 \
#    --LMO_iter=20

# Create single mixture plot
echo -e "${HCYAN} Plotting results $@${NC}" ; 
OUTDIR=${TD}/1d
echo_and_run \
  python ${SRC}/plots/plot_single_mixture.py \
    --outdir=stdout \
    --target=${OUTDIR}/target_dist.npz \
    --qt=${OUTDIR}/qt_latest.npz,${OUTDIR}/qt_iter5.npz \
    --labels=latest,iter5 \
    #--grid2d
