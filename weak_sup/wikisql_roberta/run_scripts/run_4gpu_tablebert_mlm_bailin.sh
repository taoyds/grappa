EXP_ID="tap13"
EXP_NAME="4gpu_run_roberta_large_augment_mlm_arxiv_ft_Ep3"

export CUDA_VISIBLE_DEVICES=4,5,6,7
python train_seq.py ${EXP_ID} ${EXP_NAME}