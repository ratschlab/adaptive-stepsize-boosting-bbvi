
# Run

## Bayesian Logistic Regresssion

```
  python ${SRC}/scripts/bayesian_logistic_regression.py --exp chem \
   --fw_variant ${FW_VAR} \
   --base_dist mvl \
   --outdir ${DDIR} \
   --datapath ${SRC}/data/chem \
   --n_fw_iter ${ITER} \
   --seed ${seed} \
   --linit_fixed ${linit} \
   --LMO_iter 1000 \
   --iter0 vi \
   --n_line_search_iter 25 \
   --n_monte_carlo_samples 1000
```

Prameter values to create the results in Table 1 are `--linit_fixed 1.0 --damping_adafw 0.1 --exp_adafw 5.0` 
for `AdaAFW`, `--linit_fixed 0.01 --damping_adafw 0.99 --exp_adafw 1.01` for `AdaPFW`,
 `--linit_fixed 1.0 --damping_adafw 0.01 --exp_adafw 2.0` for `AdaFW` and
`--linit_fixed 0.05` for `line search`

Run on multiple seeds for more robust evaluation with
confidence intervals

Data is in `/cluster/home/shekhars/thesis_data/blr/hp`

For generating plot 1 and table 1 of table, run

```
python plot_runs.py \
  --cluster=True \
  --datapath=/path/to/data \
  --adaptive_var adafw line_search ada_afw ada_pfw \
  --outfile=/outdir/image.png \
  --select_run=run_mean \
  --n_fw_iter=50
```
