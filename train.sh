python train_two_stream_unet.py \
 --saved_data_file ConvKalman_cross2_rot_flip_traj7 \
 --epochs 70 \
 --learning-rate 1e-4 \
 --batch_size 8 \
 --iter Convkalman2_cross2_BAMC_unetstart_64_kf_32_seed_42 \
 --store_weights \
 --eval_steps 120 \
 --traj_len 7 \
 --n_flow 1 \
 --conv_layers 1 \
 --kalman_flag \
 --use_saved_data \
 --tensorboard \
 --data_type DARPA \
 --multi_attn 0

# Kalman_data_Ablation_traj7 
# ConvKalman_BlueGel_trajlen=10
#  --load Attention_UNet_BlueGel_unetstart_64_seed_42_conv_layers_1_nflow_1 \
#  --load checkpoints/exp/Vanilla_UNet_iou_conv_layers_1_n_flow_1/CP_best.pth \video_data_len_10
# python train_two_stream_unet.py \
