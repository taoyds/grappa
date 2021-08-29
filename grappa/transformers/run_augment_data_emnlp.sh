
wandb login 67747854969713744403de273b3218fb1ecd031a

LOGDIR="logs_augment_data_emnlp"
rm -r $LOGDIR
mkdir $LOGDIR

export NGPU=3;
nohup python -u -m torch.distributed.launch --nproc_per_node=$NGPU finetuning_roberta.py --train_corpus data/augment_data.txt \
                                   --eval_corpus data/spider_dev_data_v2.txt \
                                   --train_eval_corpus data/spider_train_data_small_v2.txt \
                                   --bert_model roberta-base \
                                   --output_dir $LOGDIR/ \
                                   --do_train \
                                   --do_eval \
                                   --train_batch_size 24 \
                                   --max_seq_length 198 \
                                   --num_train_epochs 300 \
                                   > $LOGDIR/log.out &
