from pathlib import Path

class Config():
    def __init__(self,
        reader_pkl,
        sketch_train_pkl,
        sketch_dev_pkl,
        sketch_test_pkl,
        sketch_action_file,
        prod_file,
        roberta_path,
        roberta_lr,
        roberta_finetune,
        token_embed_size,
        token_dropout,
        token_rnn_size,
        token_indicator_size,
        slot_dropout,
        prod_embed_size,
        prod_rnn_size,
        prod_dropout,
        op_embed_size,
        column_type_embed_size,
        column_indicator_size,
        slot_hidden_score_size,
        model_type,
        lr,
        l2,
        clip_norm,
        nodes,
        gpu_ids,
        node_rank,
        addr,
        port,
        epochs,
        batch_size,
        seed,
        wandb,
    ):
        # data dir
        self.reader_pkl = reader_pkl

        # sketch file
        self.sketch_train_pkl = sketch_train_pkl
        self.sketch_dev_pkl = sketch_dev_pkl
        self.sketch_test_pkl = sketch_test_pkl
        self.sketch_action_file = sketch_action_file
        self.prod_file = prod_file

        # config for roberta
        self.roberta_path = roberta_path
        self.roberta_lr = roberta_lr
        self.roberta_finetune = roberta_finetune

        # config for programmer
        self.token_embed_size = token_embed_size
        self.token_dropout = token_dropout
        self.token_rnn_size = token_rnn_size
        self.token_indicator_size = token_indicator_size
        self.slot_dropout = slot_dropout
        self.prod_embed_size = prod_embed_size
        self.prod_rnn_size = prod_rnn_size
        self.prod_dropout = prod_dropout
        self.op_embed_size = op_embed_size

        self.column_type_embed_size = column_type_embed_size
        self.column_indicator_size = column_indicator_size
        self.slot_hidden_score_size = slot_hidden_score_size

        self.model_type = model_type

        # config for optimization
        self.lr = lr
        self.l2 = l2
        self.clip_norm = clip_norm

        # config for DistributedDataParallel
        self.nodes = nodes
        self.gpu_ids = gpu_ids
        self.gpus = len(self.gpu_ids)
        self.world_size = self.gpus * self.nodes
        self.node_rank = node_rank
        self.addr = addr
        self.port = port

        # config for training
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        # visualization
        self.wandb = wandb

    def __repr__(self):
        return str(vars(self))

configs = {}

configs["2gpu_run_grappa_1080"]=Config(
    # data dir
    reader_pkl="processed/wikisql_glove_42B_minfreq_3.pkl",

    # sketch file
    sketch_train_pkl="processed/train.pkl",
    sketch_dev_pkl="processed/dev.pkl",
    sketch_test_pkl="processed/test.pkl",
    sketch_action_file="processed/sketch.actions",
    prod_file="processed/productions.txt",

    # config for roberta
    roberta_path="/home/s1844182/grappa_logs_checkpoints/mlm_ssp",
    roberta_lr=1e-5,
    roberta_finetune=True,

    # config for programmer
    token_embed_size=1024,
    token_dropout=0.35,
    token_rnn_size=256,
    token_indicator_size=16,
    slot_dropout=0.25,
    prod_embed_size=512,
    prod_rnn_size=512,
    prod_dropout=0.25,
    op_embed_size=256,

    column_type_embed_size=16,
    column_indicator_size=16,
    slot_hidden_score_size=512,

    model_type="struct",

    # config for optimization
    lr=5e-4,
    l2=1e-5,
    clip_norm=3,

    # config for DistributedDataParallel
    nodes=1,
    gpu_ids=[0,1],
    node_rank=0,
    addr='127.0.0.1',
    port='8006',

    # config for training
    epochs=15,
    batch_size=1,
    seed=3264,

    # config for visualization
    wandb=False,
)

configs["2gpu_run_V100"]=Config(
    # data dir
    reader_pkl="processed/wikisql_glove_42B_minfreq_3.pkl",

    # sketch file
    sketch_train_pkl="processed/train.pkl",
    sketch_dev_pkl="processed/dev.pkl",
    sketch_test_pkl="processed/test.pkl",
    sketch_action_file="processed/sketch.actions",
    prod_file="processed/productions.txt",

    # config for roberta
    roberta_path="~/grappa_logs_checkpoints/mlm_ssp",
    roberta_lr=1e-5,
    roberta_finetune=True,

    # config for programmer
    token_embed_size=1024,
    token_dropout=0.35,
    token_rnn_size=256,
    token_indicator_size=16,
    slot_dropout=0.25,
    prod_embed_size=512,
    prod_rnn_size=512,
    prod_dropout=0.25,
    op_embed_size=256,

    column_type_embed_size=16,
    column_indicator_size=16,
    slot_hidden_score_size=512,

    model_type="struct",

    # config for optimization
    lr=5e-4,
    l2=1e-5,
    clip_norm=3,

    # config for DistributedDataParallel
    nodes=1,
    gpu_ids=[0,1],
    node_rank=0,
    addr='127.0.0.1',
    port='8006',

    # config for training
    epochs=15,
    batch_size=3,
    seed=3264,

    # config for visualization
    wandb=False,
)

configs["4gpu_run_grappa_V100"]=Config(
    # data dir
    reader_pkl="processed/wikisql_glove_42B_minfreq_3.pkl",

    # sketch file
    sketch_train_pkl="processed/train.pkl",
    sketch_dev_pkl="processed/dev.pkl",
    sketch_test_pkl="processed/test.pkl",
    sketch_action_file="processed/sketch.actions",
    prod_file="processed/productions.txt",

    # config for roberta
    roberta_path="~/grappa_logs_checkpoints/mlm_ssp",
    roberta_lr=1e-5,
    roberta_finetune=True,

    # config for programmer
    token_embed_size=1024,
    token_dropout=0.35,
    token_rnn_size=256,
    token_indicator_size=16,
    slot_dropout=0.25,
    prod_embed_size=512,
    prod_rnn_size=512,
    prod_dropout=0.25,
    op_embed_size=256,

    column_type_embed_size=16,
    column_indicator_size=16,
    slot_hidden_score_size=512,

    model_type="struct",

    # config for optimization
    lr=5e-4,
    l2=1e-5,
    clip_norm=3,

    # config for DistributedDataParallel
    nodes=1,
    gpu_ids=[0,1,2,3],
    node_rank=0,
    addr='127.0.0.1',
    port='8008',

    # config for training
    epochs=15,
    batch_size=3,
    seed=3264,

    # config for visualization
    wandb=False,
)
