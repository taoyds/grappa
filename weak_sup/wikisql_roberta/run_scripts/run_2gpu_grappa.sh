EXP_ID="exp0"
EXP_NAME="2gpu_run_grappa_1080"

export CUDA_VISIBLE_DEVICES=0,1
python train_seq.py ${EXP_ID} ${EXP_NAME}
