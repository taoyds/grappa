

GRAPPA_PATH="../grappa/grappa_logs_checkpoints/mlm_ssp/"

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --seed 1 \
                --bS 12 \
                --accumulate_gradients 2 \
                --lr 0.001 \
                --lr_bert 0.00001 \
                --train_size 300000 \
                --max_seq_leng 218 \
                --model_path "$GRAPPA_PATH" \
                >& logs_mlm_ssp.out &
