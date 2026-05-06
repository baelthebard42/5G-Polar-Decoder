# generating the systematic generator matrix, column permutations and other essentials for decoding from non systematic matrix.
python make_systematic.py \
    --input ./codes/LDPC_N121_K80.alist \
    --alist \
    --output ./systematic_forms/LDPC_N121_K80_transform.npz


# checking if the result actually works
python check_systematic.py \
    --code-hint LDPC_N121_K80 \
    --k 80 \
    --num_trials 1000 \
    --transform ./systematic_forms/LDPC_N121_K80_transform.npz


# simulating the transfer of data over noisy communication channel
 python simulate.py --code-hint LDPC_N121_K80 --path ./results/run_9/B28B07BE398116977740C70539E8AFCC  --snr_lower -10 --snr_upper 10 --snr_step 1 --img_data_path ./simulation_data/cifar_airplane_small --results_path ./simulation_results --decoder model  --transform ./systematic_forms/LDPC_N121_K80_transform.npz