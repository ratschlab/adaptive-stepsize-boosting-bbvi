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
OUTDIR=${UHOME}/thesis_data/blr


## fw_variant - fixed, line_search, fc, adafw
FW_VAR_ref=fixed

# Number of iterations
ITER=20

# 10 random seeds
for i in `seq 1 10`;
do
  seed=${RANDOM}
  for base_dist in mvn mvl # mvnormal, mvlaplace
  do
    for iter0 in random vi
    do
      # Run fixed reference
      DDIR_ref=${OUTDIR}/${FW_VAR_ref}_${base_dist}_init_${iter0}_${i}

      echo_and_run \
        mkdir -p ${DDIR_ref}

      echo_run_and_log ${DDIR_ref}/run.log \
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

        # Run adaptive on parameters
        tau_list=(1.01 1.1 1.5 2.0)
        eta_list=(0.1 0.01 0.5 0.99)
        linit_list=(0.01 1.0 100.0)
        #tau=2.0
        #eta=0.2
        #linit_fixed=10.0
        counter=1
        for tau in ${tau_list[*]}; do
          for eta in ${eta_list[*]}; do
            for linit_fixed in ${linit_list[*]}; do
              for FW_VAR in adafw ada_pfw ada_afw; do

                DDIR=${OUTDIR}/${FW_VAR}_${base_dist}_init_${iter0}_${i}_${counter}

                echo_and_run \
                  mkdir -p ${DDIR}

                echo_run_and_log ${DDIR}/run.log \
                  python ${SRC}/scripts/bayesian_logistic_regression.py \
                    --exp chem \
                    --fw_variant ${FW_VAR} \
                    --base_dist ${base_dist} \
                    --outdir ${DDIR} \
                    --datapath ${SRC}/data/chem \
                    --n_fw_iter ${ITER} \
                    --seed ${seed} \
                    --LMO_iter 1000 \
                    --linit fixed \
                    --distance_metric kl \
                    --linit_fixed ${linit_fixed} \
                    --damping_adafw ${eta} \
                    --exp_adafw ${tau} \
                    --iter0 ${iter0} \
                    --n_monte_carlo_samples 1000

              done
            done
          done
        done

    done
  done
done
