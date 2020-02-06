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
TD=${UHOME}/thesis_data
T=${UHOME}/thesis
SRC=${T}/boosting-bbvi-private/boosting_bbvi
DOC=${T}/boosting-bbvi-private/docs
#OUTDIR=${TD}/line_search_test
#OUTDIR=${TD}/rlmo
OUTDIR=${TD}/rlmo_curvature
#OUTDIR=${TD}/1d
#OUTDIR=${TD}/blr
#OUTDIR=${TD}/2d
#OUTDIR=${TD}/nd
#OUTDIR_ref=${TD}/mixture_model_variants

## fw_variant - fixed, line_search, fc, adafw
#FW_VAR=$1
#FW_VAR=ada_pfw
#FW_VAR_ref=line_search
#FW_VAR2=line_search
#FW_VAR3=adafw
#FW_VAR4=adafw
#FW_VAR5=fc

#DDIR_ref=${OUTDIR}/${FW_VAR_ref}
#DDIR2=${OUTDIR}/${FW_VAR2}
#DDIR3=${OUTDIR}/${FW_VAR3}
#DDIR4=${OUTDIR}/${FW_VAR4}
#DDIR5=${OUTDIR}/${FW_VAR5}

# Clean output directory
#echo -e "${HRED} Making data directory $@${NC}" ; 

## **********Run the mixture model************* ##
## exp - mixture, mixture_2d
#echo -e "${HGREEN} Running Boosted BBVI $@${NC}" ; 

#mkdir -p ${DDIR_ref}

eta=0.1
tau=2.0
linit=10.
dim=2
FW_VAR=adafw

for eta in 0.1 ; do
  seed=${RANDOM}
  for tau in 2.0 ; do

    DDIR=${OUTDIR}
    mkdir -p ${DDIR}

    echo_and_run \
      python ${SRC}/scripts/random_lmo_curvature.py \
        --outdir ${DDIR} \
        --seed $seed \
        --dimension $dim \
        --n_target_components 2 \
        --n_monte_carlo_samples 1000 \
        --adafw_MAXITER 10 \
        --damping_adafw ${eta} \
        --exp_adafw ${tau} \
        --linit_fixed ${linit} \
        --n_fw_iter 20 \
        --distance_metric=kl \
        --fw_variant ${FW_VAR} \
        --LMO random \
        --dist normal
  done
done

#eta=0.1
#tau=2.0
#linit=10.
#
#for dim in 2; do
#  seed=${RANDOM}
#  #for FW_VAR in fixed line_search adafw; do
#  for FW_VAR in adafw; do
#
#    DDIR=${OUTDIR}/${dim}d/${FW_VAR}
#    mkdir -p ${DDIR}
#
#    echo_and_run \
#      python ${SRC}/scripts/toy_random_lmo.py \
#        --outdir ${DDIR} \
#        --seed $seed \
#        --dimension $dim \
#        --n_target_components 2 \
#        --n_monte_carlo_samples 1000 \
#        --n_line_search_iter 10 \
#        --adafw_MAXITER 10 \
#        --damping_adafw ${eta} \
#        --exp_adafw ${tau} \
#        --linit_fixed ${linit} \
#        --n_fw_iter 10 \
#        --distance_metric=kl \
#        --fw_variant ${FW_VAR} \
#        --LMO random \
#        --dist normal
#  done
#done


#echo_and_run \
#  python ${SRC}/scripts/mixture_model_relbo.py \
#    --exp mixture_nd \
#    --fw_variant ${FW_VAR} \
#    --outdir=${DDIR} \
#    --n_fw_iter=20 \
#    --LMO_iter=1000 \
#    --n_monte_carlo_samples ${i} \
#    --n_line_search_iter 20 \
#    --iter0=random \
#    --distance_metric=kl \
#    --linit_fixed ${linit} \
#    --damping_adafw ${eta} \
#    --exp_adafw ${tau}

#for i in `seq 1 50`; do
#  seed=${RANDOM}
#  for FW_VAR in line_search adafw; do
#
#    DDIR=${OUTDIR}/${FW_VAR}_10d
#    #mkdir -p ${DDIR}
#
#    echo_and_run \
#      python ${SRC}/scripts/mixture_model_relbo.py \
#        --exp mixture_nd \
#        --fw_variant ${FW_VAR} \
#        --outdir=${DDIR} \
#        --n_fw_iter=1 \
#        --LMO_iter=1000 \
#        --n_monte_carlo_samples ${i} \
#        --n_line_search_iter 10 \
#        --iter0=random \
#        --distance_metric=kl \
#        --linit_fixed ${linit} \
#        --damping_adafw ${eta} \
#        --exp_adafw ${tau} \
#        --seed ${seed} \
#        --restore
#
#    done
#done


#echo_and_run \
#  python ${SRC}/scripts/mixture_model_relbo.py \
#    --exp mixture_2d \
#    --fw_variant ${FW_VAR} \
#    --outdir=${DDIR} \
#    --n_fw_iter=10 \
#    --LMO_iter=1000 \
#    --n_monte_carlo_samples ${i} \
#    --n_line_search_iter 25 \
#    --iter0=random

#echo_and_run \
#  python ${SRC}/scripts/mixture_model_relbo.py \
#    --exp mixture \
#    --fw_variant ${FW_VAR} \
#    --outdir=${DDIR} \
#    --n_fw_iter=10 \
#    --LMO_iter=1000 \
#    --n_monte_carlo_samples ${i} \
#    --iter0=random

## **********Running test script************* ##
echo -e "${HPURPLE} Running test $@${NC}" ; 
tau_list=(1.01 1.1 1.5 2.0)
eta_list=(0.1 0.01 0.5 0.99)
linit_list=(0.01 1.0 100.0)
#for tau in ${tau_list[*]}; do
#  for eta in ${eta_list[*]}; do
#    for linit_fixed in ${linit_list[*]}; do
#      seed=${RANDOM}
#      python ${SRC}/tests/test_step_size.py \
#          --exp mixture \
#          --outdir=${OUTDIR} \
#          --fw_variant adafw \
#          --n_monte_carlo_samples 50 \
#          --n_line_search_iter 25 \
#          --linit_fixed ${linit_fixed} \
#          --damping_adafw ${eta} \
#          --exp_adafw ${tau} \
#          --adafw_MAXITER 20 \
#          --seed ${seed}
#    done
#  done
#done

#iters=( 1000 )
#eta=0.1
#tau=2.0
#linit=10
##for i in "${iters[@]}"
#for i in `seq 1 30`;
#do
#  seed=${RANDOM}
#  echo_and_run \
#    python ${SRC}/tests/test_step_size.py \
#            --exp mixture \
#            --outdir=${DDIR} \
#            --fw_variant ${FW_VAR} \
#            --n_monte_carlo_samples 100 \
#            --linit_fixed ${linit} \
#            --damping_adafw ${eta} \
#            --exp_adafw ${tau} \
#            --adafw_MAXITER 20 \
#            --seed ${seed}
#
#  echo_and_run \
#    python ${SRC}/tests/test_step_size.py \
#            --exp mixture \
#            --outdir=${DDIR_ref} \
#            --fw_variant ${FW_VAR_ref} \
#            --n_monte_carlo_samples 50 \
#            --n_line_search_iter 20 \
#            --adafw_MAXITER 20 \
#            --seed ${seed}
#done

#echo_and_run \
#  python ${SRC}/tests/test_gap.py \
#    --n_fw_iter=10 \
#    --LMO_iter=1000 \
#    --n_monte_carlo_samples 1000 \

## **********Plotting results************* ##
echo -e "${HCYAN} Plotting results $@${NC}" ; 

# two fixed points run
#echo_and_run \
#  python ${SRC}/plots/plot_line_search.py \
#    --outdir=stdout \
#    --runs=${DDIR_ref}/gamma.csv,${DDIR}/gamma.csv

#echo_and_run \
#  python ${SRC}/plots/plot_single_mixture.py \
#    --outdir=stdout \
#    --target=${OUTDIR}/target_dist.npz \
#    --qt=${OUTDIR_ref}/qt_latest.npz,${OUTDIR}/qt_latest.npz \
#    --labels=latest_${FW_VAR_ref},latest_${FW_VAR} \
#    #--grid2d
#    #--qt=${OUTDIR}/qt_latest.npz,${OUTDIR}/qt_iter2.npz,${OUTDIR}/qt_iter0.npz \
#    #--labels=latest,iter2,init \

#echo_and_run \
#  python ${SRC}/plots/plot_mixture_comps.py \
#    --outdir=stdout \
#    --qt=${OUTDIR}/qt_iter5.npz \
#    --label=${FW_VAR}
#
#echo_and_run \
#  python ${SRC}/plots/plot_mixture_comps.py \
#    --outdir=stdout \
#    --qt=${OUTDIR_ref}/qt_iter5.npz \
#    --label=${FW_VAR_ref}
#
#echo_and_run \
#  python ${SRC}/plots/plot_mixture_comps.py \
#    --outdir=stdout \
#    --qt=${OUTDIR2}/qt_iter5.npz \
#    --label=${FW_VAR2}
#
#echo_and_run \
#  python ${SRC}/plots/plot_mixture_comps.py \
#    --outdir=stdout \
#    --qt=${OUTDIR3}/qt_iter5.npz \
#    --label=${FW_VAR3}
#
#echo_and_run \
#  python ${SRC}/plots/plot_mixture_comps.py \
#    --outdir=stdout \
#    --qt=${OUTDIR5}/qt_iter5.npz \
#    --label=${FW_VAR5}

#echo_and_run \
#  python ${SRC}/plots/plot_losses.py \
#    --elbos_files=${OUTDIR}/elbos.csv,${OUTDIR_ref}/elbos.csv,${OUTDIR2}/elbos.csv,${OUTDIR3}/elbos.csv \
#    --relbos_files=${OUTDIR}/relbos.csv,${OUTDIR_ref}/relbos.csv,${OUTDIR2}/relbos.csv,${OUTDIR3}/relbos.csv \
#    --kl_files=${OUTDIR}/kl.csv,${OUTDIR_ref}/kl.csv,${OUTDIR2}/kl.csv,${OUTDIR3}/kl.csv \
#    --times_files=${OUTDIR}/times.csv,${OUTDIR_ref}/times.csv,${OUTDIR2}/times.csv,${OUTDIR3}/times.csv \
#    --labels=${FW_VAR},${FW_VAR_ref},${FW_VAR2},${FW_VAR3}

#elbos_files=""
#kl_files=""
#labels=""
#folders=""
#for variant in $( ls ${OUTDIR} ); do
#  if [[ $variant =~ adafw_[0-9]+ ]]; then
#    #echo matches: $variant
#    elbos_files=${elbos_files}${OUTDIR}/${variant}/elbos.csv,
#    kl_files=${kl_files}${OUTDIR}/${variant}/kl.csv,
#    folders=${folders}${OUTDIR}/${variant},
#    labels=${labels}${variant},
#  #else
#    #echo !matches: $variant
#  fi
#done
## removing last comma
#elbos_files=${elbos_files%?};
#kl_files=${kl_files%?};
#labels=${labels%?};
#folders=${folders%?};
##python ${SRC}/plots/plot_losses.py \
##  --elbos_files=${elbos_files} --kl_files=${kl_files} --labels=${labels}
#
#python ${SRC}/plots/plot_heatmap.py \
#  --metric=kl \
#  --dirlist=${folders}


# outdir = stdout, ${DOC}/plots
#all_runs=$(ls ${OUTDIR}/gradients/*)
#all_runs_comma=$(echo ${all_runs} | sed "s/\s/,/g")
## metric = gamma,E_s,E_q
#METRIC=E_q
#echo_and_run \
#  python ${src}/plots/plot_line_search.py \
#    --metric=${metric} \
#    --outdir=stdout \
#    --extra=0.6 \
#    --outfile=test_e_q.png \
#    --runs=${all_runs_comma}

