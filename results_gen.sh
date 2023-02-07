python eval_results_gen.py \
 --batch_size 1 \
 --saved_data_file ConvKalmanNet_ablation_VAL \
 --iter VanillaUNet_start_channel_64_kf_32_seed_42_Abl_conv_layers_1_nflow_1_VAL_DEBUG \
 --eval \
 --conv_layers 1 \
 --n_flow 1 \
 --traj_len 10 \
 --use_saved_data \
 --data_type DARPA \
 --multi_attn 0 


 #CP_best_val_score
#  --use_saved_data \
#  --kalman_flag \
 # Stacked_ablation_VAL
 # flow_ablation_VAL
 #datafile: ConvKalmanNet_ablation_VAL
#  --load checkpoints/exp/ConvkalmanNet_Unet_start_channel_64_kf_32_seed_101_Abl_conv_layers_1_nflow_1/CP_3300.pth \
#  --load checkpoints/exp/VanillaUnet_start_channel_64_seed_42_Abl_conv_layers_1_nflow_1/CP_best_val_score.pth \
