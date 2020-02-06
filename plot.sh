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
## get experiment from argument
EXP=$1
OUTDIR=${TD}/${EXP}

## fw_variant - fixed, line_search, fc, adafw
FW_VAR=adafw

DDIR=${OUTDIR}/${FW_VAR}

## **********Plotting results************* ##

#echo_and_run \
#  python ${SRC}/plots/plot_losses.py \
#    --elbos_files=${DDIR_ref}/elbos.csv,${DDIR}/elbos.csv \
#    --mse_files=${DDIR_ref}/mse_test.csv,${DDIR}/mse_test.csv \
#    --labels=${FW_VAR_ref},${FW_VAR} \
#    --start=20 \
#    --smoothingWt 0.

## Plot 1 of the paper, with table and elbo
#echo -e "${HCYAN} Plotting blr $@${NC}" ; 
#echo_and_run \
#  python ${SRC}/plots/plot_runs.py \
#   --exp=${EXP} \
#   --cluster=True \
#   --datapath=${OUTDIR}/hp \
#   --adaptive_var adafw line_search ada_afw ada_pfw \
#   --outfile=${OUTDIR}/res.png \
#   --select_run=run_mean \
#   --n_fw_iter=50

echo -e "${HCYAN} Plotting ${EXP} ${NC}" ; 
## Synthetic data
if [ $EXP == 'rlmo' ]; then

  echo_and_run \
    python ${SRC}/plots/syn_runs.py \
      --datapath=${OUTDIR}/hp \
      --outfile stdout
      #--outfile ${OUTDIR}/rlmo.png

elif [ $EXP == 'bmf' ]; then
## Computing table and plots for BLR/BMF
  echo_and_run \
    python ${SRC}/plots/plot_runs.py \
    --exp=${EXP} \
    --cluster=True \
    --datapath=${OUTDIR}/hp_70 \
    --adaptive_var adafw line_search ada_afw ada_pfw \
    --select_run=run_median \
    --n_fw_iter=75 \
    --outfile=$OUTDIR/best_run_df.csv
    #--outfile=$OUTDIR/n_comps.png 
    #--outfile=$OUTDIR/bmf_elbo.png \
elif [ $EXP == 'eicu' ]; then
  echo_and_run \
    python ${SRC}/plots/plot_runs.py \
    --exp=blr \
    --cluster=True \
    --datapath=${OUTDIR}/hp \
    --outfile=stdout \
    --adaptive_var adafw line_search ada_afw ada_pfw \
    --select_run=run_median \
    --n_fw_iter=40 \
    --outfile=$OUTDIR/n_comps.png
    #--outfile=$OUTDIR/all_run_eicu.csv
elif [ $EXP == 'blr' ]; then
  echo_and_run \
    python ${SRC}/plots/plot_runs.py \
    --exp=blr \
    --cluster=True \
    --datapath=${OUTDIR}/hp \
    --adaptive_var adafw line_search ada_afw ada_pfw \
    --outfile stdout \
    --select_run=run_median \
    --n_fw_iter=50 \
    --outfile=$OUTDIR/all_run_chemreact.csv
    #--outfile=$OUTDIR/n_comps.png
    #--outfile=stdout
    #--outfile=$OUTDIR/dist_steps.png \
    #--outfile=$OUTDIR/blr_elbo_all.png \
elif [ $EXP == 'blr2' ]; then
  echo_and_run \
    python ${SRC}/plots/plot_runs.py \
    --exp=blr2 \
    --cluster=True \
    --datapath=${TD}/blr/hp,${TD}/eicu/hp \
    --adaptive_var adafw line_search ada_afw ada_pfw \
    --outfile=stdout \
    --select_run=run_median
    #--outfile=$TD/blr/elbo_all.png \
    #--n_fw_iter=50
fi
