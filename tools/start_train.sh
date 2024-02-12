




# Train SwinTransformer with imagenette2-320 - 10k samples - 10 calsses
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
GPU=0
LOGFILE=LOG.swin_t_tiny_IN320_JAN_27
SCREEN_SESSION_NAME=swin_t_tiny_IN320_JAN_27
EXP_ROOT_DIR=/mnt/disks/ext/exps_swin_t/swin_t_tiny_IN320_JAN_27
WORK_DIR_NAME=swin_t_tiny_IN320_JAN_27
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py '$CONFIG' --work-dir '$EXP_ROOT_DIR'/swin_t_tiny_IN320_JAN_27/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"
tail -f $LOGFILE



# 25 Train SwinTransformer with imagenette2-320 - 10k samples - 10 calsses
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
GPU=0
LOGFILE=LOG.pruned_swin_t_tiny_IN320_JAN_27
SCREEN_SESSION_NAME=pruned_swin_t_tiny_IN320_JAN_27
EXP_ROOT_DIR=/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_IN320_JAN_27
WORK_DIR_NAME=pruned_swin_t_tiny_IN320_JAN_27
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py '$CONFIG' --work-dir '$EXP_ROOT_DIR'/swin_t_tiny_IN320_JAN_27/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"
tail -f $LOGFILE



# 50 Pruned Train SwinTransformer with imagenette2-320 - 10k samples - 10 calsses
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
GPU=0
LOGFILE=LOG.pruned_50_swin_t_tiny_IN320_JAN_27
SCREEN_SESSION_NAME=pruned_50_swin_t_tiny_IN320_JAN_27
EXP_ROOT_DIR=/mnt/disks/ext/exps_swin_t/pruned_50_swin_t_tiny_IN320_JAN_27
WORK_DIR_NAME=pruned_50_swin_t_tiny_IN320_JAN_27
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py '$CONFIG' --work-dir '$EXP_ROOT_DIR'/pruned_50_swin_t_tiny_IN320_JAN_27/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"
tail -f $LOGFILE


# CONT 50 Pruned Train SwinTransformer with imagenette2-320 - 10k samples - 10 calsses
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
GPU=0
LOGFILE=LOG.pruned_50_CONT_swin_t_tiny_IN320_JAN_27
SCREEN_SESSION_NAME=pruned_50_CONT_swin_t_tiny_IN320_JAN_27
EXP_ROOT_DIR=/mnt/disks/ext/exps_swin_t/pruned_50_CONT_swin_t_tiny_IN320_JAN_27
WORK_DIR_NAME=pruned_50_CONT_swin_t_tiny_IN320_JAN_27
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py '$CONFIG' --work-dir '$EXP_ROOT_DIR'/pruned_50_CONT_swin_t_tiny_IN320_JAN_27/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"
tail -f $LOGFILE




# Train PRUNED SwinTransformer with TinyImagenet
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
GPU=0
LOGFILE=LOG.pruned_swin_t_tiny_in200
SCREEN_SESSION_NAME=pruned_swin_t_tiny_in200
EXP_ROOT_DIR=/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_in200
WORK_DIR_NAME=pruned_swin_t_tiny_in200
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py '$CONFIG' --work-dir '$EXP_ROOT_DIR'/pruned_swin_t_tiny_in200/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"



# Train PRUNED SwinTransformer with TinyImagenet
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_1k.py
GPU=0
LOGFILE=LOG.pruned_swin_t_tiny_in200_cont
SCREEN_SESSION_NAME=pruned_swin_t_tiny_in200_cont
EXP_ROOT_DIR=/mnt/disks/ext/exps_swin_t/pruned_swin_t_tiny_in200_cont
WORK_DIR_NAME=pruned_swin_t_tiny_in200_cont
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py '$CONFIG' --work-dir '$EXP_ROOT_DIR'/pruned_swin_t_tiny_in200/'$WORK_DIR_NAME' --resume'
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"





# Train SwinTransformer with ImageNet-1k imagenette2-320 - 34k samples - 1000 calsses
# kaggle datasets download -d ifigotin/imagenetmini-1000
CONFIG=configs/swin_transformer/swin-tiny_1xb256_in1k_224_34k.py
GPU=0
LOGFILE=LOG.swin_t_tiny_in1000_34k
SCREEN_SESSION_NAME=swin_t_tiny_in1000_34k
EXP_ROOT_DIR=/mnt/disks/ext/exps_swin_t/swin_t_tiny_in1000_34k
WORK_DIR_NAME=swin_t_tiny_in1000_34k
command='cd $REPO_MMDETECTION_ROOT; PORT=29503 CUDA_VISIBLE_DEVICES='$GPU' python ./tools/train.py '$CONFIG' --work-dir '$EXP_ROOT_DIR'/swin_t_tiny_in1000_34k/'$WORK_DIR_NAME''
screen -S $SCREEN_SESSION_NAME -L -Logfile $LOGFILE -dm bash -c "$command"

