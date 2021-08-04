
LOGDIR="grappa_logs_checkpoints/mlm_ssp/"
rm -r $LOGDIR
mkdir $LOGDIR

#export NGPU=2;
#CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=$NGPU finetuning_roberta.py --train_corpus data/augment_mlm.txt \
export NGPU=8;
nohup python -u -m torch.distributed.launch --nproc_per_node=$NGPU finetuning_roberta.py --train_corpus data/augment_mlm.txt \
                                   --eval_corpus data/spider_dev_data_v2.txt \
                                   --train_eval_corpus data/spider_train_data_small_v2.txt \
                                   --bert_model roberta-large \
                                   --output_dir $LOGDIR/ \
                                   --do_train \
                                   --do_eval \
                                   --mlm_loss \
                                   --train_batch_size 12 \
                                   --max_seq_length 218 \
                                   --num_train_epochs 10 \
                                   > $LOGDIR/log.out
