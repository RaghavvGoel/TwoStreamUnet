python train_two_stream_unet.py \
 --saved_data_file Recurrent_StepGap5_Kfold2 \
 --epochs 30 \
 --learning-rate 1e-4 \
 --batch_size 4 \
 --iter Encoder_ConvKalman_Kfold2 \
 --store_weights \
 --eval_steps 150 \
 --traj_len 7 \
 --n_flow 1 \
 --conv_layers 1 \
 --kalman_flag \
 --tensorboard \
 --use_saved_data \
 --data_type DARPA \
 --multi_attn 0 \

# Kalman_data_Ablation_traj7 
# ConvKalman_BlueGel_trajlen=10
#  --load Attention_UNet_BlueGel_unetstart_64_seed_42_conv_layers_1_nflow_1 \
#  --load checkpoints/exp/Vanilla_UNet_iou_conv_layers_1_n_flow_1/CP_best.pth \video_data_len_10
# python train_two_stream_unet.py \
