python eval_two_stream_unet.py \
 --saved_data_file Stacked_Kfold1_VAL \
 --batch_size 1 \
 --iter ConvKalman_Unet_DARPA_start_channel_64_seed_42_VAL \
 --load checkpoints/exp/VanillaUnet_start_channel_64_seed_42_Abl_conv_layers_1_nflow_1/CP_best_val_score.pth \
 --eval \
 --conv_layers 1 \
 --traj_len 7 \
 --n_flow 4 \
 --needle_param \
 --use_saved_data \
 --data_type DARPA \
 --multi_attn 0

#  --kalman_flag \
#  --use_saved_data \
 #CP_best_val_score
#   flow_ablation_VAL
 #datafile: ConvKalmanNet_ablation_VAL
#  --load checkpoints/exp/ConvkalmanNet_Unet_start_channel_64_kf_32_seed_101_Abl_conv_layers_1_nflow_1/CP_3300.pth \
