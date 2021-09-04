EXP_ID="exp0"
EXP_NAME="2_gpu_grappa_1080"
mkdir -p log/${EXP_ID}_${EXP_NAME}

export CUDA_VISIBLE_DEVICES=0,1
python train_seq.py ${EXP_ID} ${EXP_NAME}
