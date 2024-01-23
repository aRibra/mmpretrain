





# pruned_swin_t_model_flat = "/mnt/disks/ext/swin_t_checkpoints/swin_t_backbone_Pruned_25.pth"
# pruned_weights = "/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_in200/pruned_swin_t_tiny_in200/pruned_swin_t_tiny_in200/epoch_2.pth"

# test original unpruned model
EXP_DIR=/mnt/disks/ext/exps_swin_t/swin_t_tiny_in200/swin_t_tiny_in200/swin_t_tiny_in200
CHECKPOINT=best_accuracy_top1_epoch_27.pth
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
CUDA_VISIBLE_DEVICES=0
METRICS_DIR=metrics_original
OUT_DIR=out_items
SHOW_DIR=show_original
LOGFILE=LOG.eval.swin_t_original
DET_RESULTS_FILENAME=result_original.pickle
SCREEN_SESSION_NAME=swin_t_original

command='cd $REPO_MMDETECTION_ROOT; CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' python3 tools/test.py '${CONFIG}' '$EXP_DIR'/'${CHECKPOINT}' --out  '$EXP_DIR'/'$OUT_DIR' --out-item pred --work-dir '$EXP_DIR'/'$METRICS_DIR' --show-dir '$EXP_DIR'/'$SHOW_DIR' --out '$EXP_DIR'/'$DET_RESULTS_FILENAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"





# test pruned model
# pruned_weights = "/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_in200/pruned_swin_t_tiny_in200/pruned_swin_t_tiny_in200/best_accuracy_top1_epoch_10.pth"
EXP_DIR=/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_in200/pruned_swin_t_tiny_in200/pruned_swin_t_tiny_in200
CHECKPOINT='dummy.pth'
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
CUDA_VISIBLE_DEVICES=0
METRICS_DIR=metrics_pruned
OUT_DIR=out_items
SHOW_DIR=show_pruned
LOGFILE=LOG.eval.swin_t_pruned
DET_RESULTS_FILENAME=result_pruned.pickle
SCREEN_SESSION_NAME=swin_t_pruned

command='cd $REPO_MMDETECTION_ROOT; CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' python3 tools/test.py '${CONFIG}' '${CHECKPOINT}' --out  '$EXP_DIR'/'$OUT_DIR' --out-item pred --work-dir '$EXP_DIR'/'$METRICS_DIR' --show-dir '$EXP_DIR'/'$SHOW_DIR' --out '$EXP_DIR'/'$DET_RESULTS_FILENAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"



# test pruned model CONT
# pruned_weights = "/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_in200_cont/pruned_swin_t_tiny_in200/pruned_swin_t_tiny_in200_cont/best_accuracy_top1_epoch_32.pth"
EXP_DIR=/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_in200_cont/pruned_swin_t_tiny_in200_cont/pruned_swin_t_tiny_in200_cont
CHECKPOINT='dummy.pth'
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
CUDA_VISIBLE_DEVICES=0
METRICS_DIR=metrics_pruned_cont
OUT_DIR=out_items_cont
SHOW_DIR=show_pruned_cont
LOGFILE=LOG.eval.swin_t_pruned_cont
DET_RESULTS_FILENAME=result_pruned_cont.pickle
SCREEN_SESSION_NAME=swin_t_pruned_cont

command='cd $REPO_MMDETECTION_ROOT; CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' python3 tools/test.py '${CONFIG}' '${CHECKPOINT}' --out  '$EXP_DIR'/'$OUT_DIR' --out-item pred --work-dir '$EXP_DIR'/'$METRICS_DIR' --show-dir '$EXP_DIR'/'$SHOW_DIR' --out '$EXP_DIR'/'$DET_RESULTS_FILENAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"

