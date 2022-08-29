python  trainUnetMotionModel.py \
 --saved_data_file 801_normalized_all_positive_nhistory=3_gauss_both_needle_label \
 --epochs 150 \
 --batch-size 10 \
 --iter baseline_without_motion\
 --attention_flag \
 --flow_history_flag \
 --use_saved_data \
 --multi_attn 3\
 --motion_flag False