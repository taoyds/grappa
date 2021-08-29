CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --seed 1 \
                --bS 16 \
                --accumulate_gradients 2 \
                --fine_tune \
                --lr 0.001 \
                --lr_bert 0.00001 \
                --train_size 30000 \
                --max_seq_leng 208 \
                --path_wikisql ../data/wikisql \
                --model_path ../column_roberta_dev/logs_augment_mlm_ft_spider_emnlp/ \
                > logs_augment_mlm_ft_spider_emnlp_30000.out &
