EXP_ID="tap15"
EXP_NAME="2gpu_run_roberta_large_augment_mlm_arxiv_ft_Ep3"

export CUDA_VISIBLE_DEVICES=2,3
python train_seq.py ${EXP_ID} ${EXP_NAME}