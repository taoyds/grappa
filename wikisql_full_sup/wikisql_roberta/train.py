# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018
import os, sys, argparse, re, json

from matplotlib.pylab import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import random as python_random
# import torchvision.datasets as dsets

# BERT
from pytorch_transformers.tokenization_roberta import RobertaTokenizer
from pytorch_transformers.modeling_roberta import RobertaConfig, RobertaModel

from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
# from sqlnet.dbengine import DBEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_hyper_param(parser):
    parser.add_argument('--tepoch', default=300, type=int)
    parser.add_argument("--bS", default=32, type=int,
                        help="Batch size")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--fine_tune',
                        default=False,
                        action='store_true',
                        help="If present, BERT is trained.")

    parser.add_argument("--model_type", default='Seq2SQL_v1', type=str,
                        help="Type of model.")
    parser.add_argument("--path_wikisql",
                        default='data/', type=str,
                        help="path to wikisql dataset")
    parser.add_argument("--path_save_for_evaluation",
                        default='./', type=str,
                        help="path to save for evaluation")
    parser.add_argument("--train_size",
                        default=100000, type=int, # Set based on maximum length of input tokens.
                        help="The number of training data to load ")

    # 1.2 BERT Parameters
    parser.add_argument("--bert_type",
                        default='bert', type=str,
                        help="bert model type")
    parser.add_argument("--model_path",
                        default='models/bert', type=str,
                        help="path to bert models")
    parser.add_argument("--max_seq_length",
                        default=222, type=int, # Set based on maximum length of input tokens.
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--num_target_layers",
                        default=2, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model')
    parser.add_argument("--bert_type_abb", default='uS', type=str,
                        help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")

    # 1.3 Seq-to-SQL module parameters
    parser.add_argument('--lS', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dr', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--hS", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")

    # 1.4 Execution-guided decoding beam-size. It is used only in test.py
    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    args = parser.parse_args()

    # Decide whether to use lower_case.
    args.do_lower_case = True

    # Seeds for random number generation
    seed(args.seed)
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #args.toy_model = not torch.cuda.is_available()
    args.toy_model = False
    args.toy_size = 12

    return args



def load_from_pretrained(model, state_dict):
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # Load from a PyTorch state_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ''
    model_to_load = model
    start_prefix = model.base_model_prefix + '.'

    load(model_to_load, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           model.__class__.__name__, "\n\t".join(error_msgs)))

    if hasattr(model, 'tie_weights'):
        model.tie_weights()  # make sure word embedding weights are still tied

#     loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
    
    return model


def get_bert(model_path, bert_type, do_lower_case, no_pretraining):
    
#     bert_config_file = os.path.join(model_path, f'roberta-base-config.json')
#     vocab_file = os.path.join(model_path, f'roberta-base-vocab.json')
#     merges_file = os.path.join(model_path, f'roberta-base-merges.txt')
#     init_checkpoint = os.path.join(model_path, f'roberta-base-pytorch_model.bin')
    
    bert_config_file = os.path.join(model_path, f'config.json')
    vocab_file = os.path.join(model_path, f'vocab.json')
    merges_file = os.path.join(model_path, f'merges.txt')
    init_checkpoint = os.path.join(model_path, f'pytorch_model.bin')


    bert_config = RobertaConfig.from_json_file(bert_config_file)
    print("bert_config.output_hidden_states: ", bert_config.output_hidden_states)
    tokenizer = RobertaTokenizer(vocab_file=vocab_file, merges_file=merges_file)

    model_bert = RobertaModel(bert_config)
    if no_pretraining:
        pass
    else:
        state_dict = torch.load(init_checkpoint, map_location='cpu')
        model_bert = load_from_pretrained(model_bert, state_dict)
        print("Load pre-trained parameters.")
    model_bert.to(device)

    return model_bert, tokenizer, bert_config

def get_opt(model, model_bert, fine_tune):
    if fine_tune:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)

        opt_bert = torch.optim.Adam(filter(lambda p: p.requires_grad, model_bert.parameters()),
                                    lr=args.lr_bert, weight_decay=0)
    else:
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr, weight_decay=0)
        opt_bert = None

    return opt, opt_bert

def get_models(args, model_path, trained=False, path_model_bert=None, path_model=None):
    # some constants
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']  # do not know why 'OP' required. Hence,

    print(f"Batch_size = {args.bS * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")

    # Get BERT
    model_bert, tokenizer, bert_config = get_bert(model_path, args.bert_type, args.do_lower_case,
                                                  args.no_pretraining)
    args.iS = bert_config.hidden_size * args.num_target_layers  # Seq-to-SQL input vector dimenstion

    # Get Seq-to-SQL

    n_cond_ops = len(cond_ops)
    n_agg_ops = len(agg_ops)
    print(f"Seq-to-SQL: the number of final BERT layers to be used: {args.num_target_layers}")
    print(f"Seq-to-SQL: the size of hidden dimension = {args.hS}")
    print(f"Seq-to-SQL: LSTM encoding layer size = {args.lS}")
    print(f"Seq-to-SQL: dropout rate = {args.dr}")
    print(f"Seq-to-SQL: learning rate = {args.lr}")
    model = Seq2SQL_v1(args.iS, args.hS, args.lS, args.dr, n_cond_ops, n_agg_ops)
    model = model.to(device)

    if trained:
        assert path_model_bert != None
        assert path_model != None

        if torch.cuda.is_available():
            res = torch.load(path_model_bert)
        else:
            res = torch.load(path_model_bert, map_location='cpu')
        model_bert.load_state_dict(res['model_bert'])
        model_bert.to(device)

        if torch.cuda.is_available():
            res = torch.load(path_model)
        else:
            res = torch.load(path_model, map_location='cpu')

        model.load_state_dict(res['model'])

    return model, model_bert, tokenizer, bert_config

def get_data(path_wikisql, args):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(path_wikisql, args.toy_model, args.toy_size, no_w2i=True, no_hs_tok=True, train_size=args.train_size)
    train_loader, dev_loader = get_loader_wikisql(train_data, dev_data, args.bS, shuffle_train=True)

    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader


def train(train_loader, train_table, model, model_bert, opt, bert_config, tokenizer,
          max_seq_length, num_target_layers, accumulate_gradients=1, check_grad=True,
          st_pos=0, opt_bert=None, path_db=None, dset_name='train'):
    model.train()
    model_bert.train()

    ave_loss = 0
    cnt = 0 # count the # of examples
    cnt_sc = 0 # count the # of correct predictions of select column
    cnt_sa = 0 # of selectd aggregation
    cnt_wn = 0 # of where number
    cnt_wc = 0 # of where column
    cnt_wo = 0 # of where operator
    cnt_wv = 0 # of where-value
    cnt_wvi = 0 # of where-value index (on question tokens)
    cnt_lx = 0  # of logical form acc
    cnt_x = 0   # of execution acc

    # Engine for SQL querying.
#     engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, t in enumerate(train_loader):
        cnt += len(t)

        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, train_table, no_hs_t=True, no_sql_t=True)
        # nlu  : natural language utterance
        # nlu_t: tokenized nlu
        # sql_i: canonical form of SQL query
        # sql_q: full SQL query text. Not used.
        # sql_t: tokenized SQL query
        # tb   : table
        # hs_t : tokenized headers. Not used.

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

        # wemb_n: natural language embedding
        # wemb_h: header embedding
        # l_n: token lengths of each question
        # l_hpu: header token lengths
        # l_hs: the number of columns (headers) of the tables.
        try:
            #
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            # e.g. train: 32.
            continue

        # score
        s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs,
                                                   g_sc=g_sc, g_sa=g_sa, g_wn=g_wn, g_wc=g_wc, g_wvi=g_wvi)

        # Calculate loss & step
        loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

        # Calculate gradient
        if iB % accumulate_gradients == 0: # mode
            # at start, perform zero_grad
            opt.zero_grad()
            if opt_bert:
                opt_bert.zero_grad()
            loss.backward()
            if accumulate_gradients == 1:
                opt.step()
                if opt_bert:
                    opt_bert.step()
        elif iB % accumulate_gradients == (accumulate_gradients-1):
            # at the final, take step with accumulated graident
            loss.backward()
            opt.step()
            if opt_bert:
                opt_bert.step()
        else:
            # at intermediate stage, just accumulates the gradients
            loss.backward()

        # Prediction
        pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
        pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        # Sort pr_wc:
        #   Sort pr_wc when training the model as pr_wo and pr_wvi are predicted using ground-truth where-column (g_wc)
        #   In case of 'dev' or 'test', it is not necessary as the ground-truth is not used during inference.
        pr_wc_sorted = sort_pr_wc(pr_wc, g_wc)
        pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc_sorted, pr_wo, pr_wv_str, nlu)


        # Cacluate accuracy
        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='train')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = [0], [0], [0]#get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # statistics
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]

    aux_out = 1

    return acc, aux_out

def report_detail(hds, nlu,
                  g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                  pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                  cnt_list, current_cnt):
    cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x = current_cnt

    print(f'cnt = {cnt} / {cnt_tot} ===============================')

    print(f'headers: {hds}')
    print(f'nlu: {nlu}')

    # print(f's_sc: {s_sc[0]}')
    # print(f's_sa: {s_sa[0]}')
    # print(f's_wn: {s_wn[0]}')
    # print(f's_wc: {s_wc[0]}')
    # print(f's_wo: {s_wo[0]}')
    # print(f's_wv: {s_wv[0][0]}')
    print(f'===============================')
    print(f'g_sc : {g_sc}')
    print(f'pr_sc: {pr_sc}')
    print(f'g_sa : {g_sa}')
    print(f'pr_sa: {pr_sa}')
    print(f'g_wn : {g_wn}')
    print(f'pr_wn: {pr_wn}')
    print(f'g_wc : {g_wc}')
    print(f'pr_wc: {pr_wc}')
    print(f'g_wo : {g_wo}')
    print(f'pr_wo: {pr_wo}')
    print(f'g_wv : {g_wv}')
    # print(f'pr_wvi: {pr_wvi}')
    print('g_wv_str:', g_wv_str)
    print('p_wv_str:', pr_wv_str)
    print(f'g_sql_q:  {g_sql_q}')
    print(f'pr_sql_q: {pr_sql_q}')
    print(f'g_ans: {g_ans}')
    print(f'pr_ans: {pr_ans}')
    print(f'--------------------------------')

    print(cnt_list)

    print(f'acc_lx = {cnt_lx/cnt:.3f}, acc_x = {cnt_x/cnt:.3f}\n',
          f'acc_sc = {cnt_sc/cnt:.3f}, acc_sa = {cnt_sa/cnt:.3f}, acc_wn = {cnt_wn/cnt:.3f}\n',
          f'acc_wc = {cnt_wc/cnt:.3f}, acc_wo = {cnt_wo/cnt:.3f}, acc_wv = {cnt_wv/cnt:.3f}')
    print(f'===============================')

def test(data_loader, data_table, model, model_bert, bert_config, tokenizer,
         max_seq_length,
         num_target_layers, detail=False, st_pos=0, cnt_tot=1, EG=False, beam_size=4,
         path_db=None, dset_name='test'):
    model.eval()
    model_bert.eval()

    ave_loss = 0
    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_wvi = 0
    cnt_lx = 0
    cnt_x = 0

    cnt_list = []
#     engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))
    results = []
    for iB, t in enumerate(data_loader):

        cnt += len(t)
        if cnt < st_pos:
            continue
        # Get fields
        nlu, nlu_t, sql_i, sql_q, sql_t, tb, hs_t, hds = get_fields(t, data_table, no_hs_t=True, no_sql_t=True)

        g_sc, g_sa, g_wn, g_wc, g_wo, g_wv = get_g(sql_i)
        g_wvi_corenlp = get_g_wvi_corenlp(t)

        wemb_n, wemb_h, l_n, l_hpu, l_hs, \
        nlu_tt, t_to_tt_idx, tt_to_t_idx \
            = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                            num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)
        try:
            g_wvi = get_g_wvi_bert_from_g_wvi_corenlp(t_to_tt_idx, g_wvi_corenlp)
            g_wv_str, g_wv_str_wp = convert_pr_wvi_to_string(g_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)

        except:
            # Exception happens when where-condition is not found in nlu_tt.
            # In this case, that train example is not used.
            # During test, that example considered as wrongly answered.
            for b in range(len(nlu)):
                results1 = {}
                results1["error"] = "Skip happened"
                results1["nlu"] = nlu[b]
                results1["table_id"] = tb[b]["id"]
                results.append(results1)
            continue

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            s_sc, s_sa, s_wn, s_wc, s_wo, s_wv = model(wemb_n, l_n, wemb_h, l_hpu, l_hs)

            # get loss & step
            loss = Loss_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, g_sc, g_sa, g_wn, g_wc, g_wo, g_wvi)

            # prediction
            pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi = pred_sw_se(s_sc, s_sa, s_wn, s_wc, s_wo, s_wv, )
            pr_wv_str, pr_wv_str_wp = convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_tt, tt_to_t_idx, nlu)
            # g_sql_i = generate_sql_i(g_sc, g_sa, g_wn, g_wc, g_wo, g_wv_str, nlu)
            pr_sql_i = generate_sql_i(pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, nlu)
        else:
            # Execution guided decoding
            prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                            l_hs, engine, tb,
                                                                                            nlu_t, nlu_tt,
                                                                                            tt_to_t_idx, nlu,
                                                                                            beam_size=beam_size)
            # sort and generate
            pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)

            # Follosing variables are just for the consistency with no-EG case.
            pr_wvi = None # not used
            pr_wv_str=None
            pr_wv_str_wp=None
            loss = torch.tensor([0])



        g_sql_q = generate_sql_q(sql_i, tb)
        pr_sql_q = generate_sql_q(pr_sql_i, tb)


        # Saving for the official evaluation later.
        for b, pr_sql_i1 in enumerate(pr_sql_i):
            results1 = {}
            results1["query"] = pr_sql_i1
            results1["table_id"] = tb[b]["id"]
            results1["nlu"] = nlu[b]
            results.append(results1)

        cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, \
        cnt_wc1_list, cnt_wo1_list, \
        cnt_wvi1_list, cnt_wv1_list = get_cnt_sw_list(g_sc, g_sa,g_wn, g_wc,g_wo, g_wvi,
                                                                   pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wvi,
                                                                   sql_i, pr_sql_i,
                                                                   mode='test')

        cnt_lx1_list = get_cnt_lx_list(cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list,
                                       cnt_wo1_list, cnt_wv1_list)

        # Execution accura y test
        cnt_x1_list = []
        # lx stands for logical form accuracy

        # Execution accuracy test.
        cnt_x1_list, g_ans, pr_ans = [0], [0], [0]#get_cnt_x_list(engine, tb, g_sc, g_sa, sql_i, pr_sc, pr_sa, pr_sql_i)

        # stat
        ave_loss += loss.item()

        # count
        cnt_sc += sum(cnt_sc1_list)
        cnt_sa += sum(cnt_sa1_list)
        cnt_wn += sum(cnt_wn1_list)
        cnt_wc += sum(cnt_wc1_list)
        cnt_wo += sum(cnt_wo1_list)
        cnt_wv += sum(cnt_wv1_list)
        cnt_wvi += sum(cnt_wvi1_list)
        cnt_lx += sum(cnt_lx1_list)
        cnt_x += sum(cnt_x1_list)

        current_cnt = [cnt_tot, cnt, cnt_sc, cnt_sa, cnt_wn, cnt_wc, cnt_wo, cnt_wv, cnt_wvi, cnt_lx, cnt_x]
        cnt_list1 = [cnt_sc1_list, cnt_sa1_list, cnt_wn1_list, cnt_wc1_list, cnt_wo1_list, cnt_wv1_list, cnt_lx1_list,
                     cnt_x1_list]
        cnt_list.append(cnt_list1)
        # report
        if detail:
            report_detail(hds, nlu,
                          g_sc, g_sa, g_wn, g_wc, g_wo, g_wv, g_wv_str, g_sql_q, g_ans,
                          pr_sc, pr_sa, pr_wn, pr_wc, pr_wo, pr_wv_str, pr_sql_q, pr_ans,
                          cnt_list1, current_cnt)

    ave_loss /= cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wvi = cnt_wvi / cnt
    acc_wv = cnt_wv / cnt
    acc_lx = cnt_lx / cnt
    acc_x = cnt_x / cnt

    acc = [ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x]
    return acc, results, cnt_list


def print_result(epoch, acc, dname):
    ave_loss, acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wvi, acc_wv, acc_lx, acc_x = acc

    print(f'{dname} results ------------')
    print(
        f" Epoch: {epoch}, ave loss: {ave_loss}, acc_sc: {acc_sc:.3f}, acc_sa: {acc_sa:.3f}, acc_wn: {acc_wn:.3f}, \
        acc_wc: {acc_wc:.3f}, acc_wo: {acc_wo:.3f}, acc_wvi: {acc_wvi:.3f}, acc_wv: {acc_wv:.3f}, acc_lx: {acc_lx:.3f}, acc_x: {acc_x:.3f}"
    )

if __name__ == '__main__':

    ## 1. Hyper parameters
    parser = argparse.ArgumentParser()
    args = construct_hyper_param(parser)

    ## 2. Paths
    path_wikisql = args.path_wikisql
    model_path = args.model_path

    ## 3. Load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wikisql, args)
    # test_data, test_table = load_wikisql_data(path_wikisql, mode='test', toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)
    # test_loader = torch.utils.data.DataLoader(
    #     batch_size=args.bS,
    #     dataset=test_data,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=lambda x: x  # now dictionary values are not merged!
    # )
    ## 4. Build & Load models
    model, model_bert, tokenizer, bert_config = get_models(args, model_path)

    ## 4.1.
    # To start from the pre-trained models, un-comment following lines.
    # path_model_bert =
    # path_model =
    # model, model_bert, tokenizer, bert_config = get_models(args, model_path, trained=True, path_model_bert=path_model_bert, path_model=path_model)

    ## 5. Get optimizers
    opt, opt_bert = get_opt(model, model_bert, args.fine_tune)

    ## 6. Train
    acc_lx_t_best = -1
    epoch_best = -1
    for epoch in range(args.tepoch):
        # train
        acc_train, aux_out_train = train(train_loader,
                                         train_table,
                                         model,
                                         model_bert,
                                         opt,
                                         bert_config,
                                         tokenizer,
                                         args.max_seq_length,
                                         args.num_target_layers,
                                         args.accumulate_gradients,
                                         opt_bert=opt_bert,
                                         st_pos=0,
                                         path_db=path_wikisql,
                                         dset_name='train')

        # check DEV
        with torch.no_grad():
            acc_dev, results_dev, cnt_list = test(dev_loader,
                                                dev_table,
                                                model,
                                                model_bert,
                                                bert_config,
                                                tokenizer,
                                                args.max_seq_length,
                                                args.num_target_layers,
                                                detail=False,
                                                path_db=path_wikisql,
                                                st_pos=0,
                                                dset_name='dev', EG=args.EG)


        print_result(epoch, acc_train, 'train')
        print_result(epoch, acc_dev, 'dev')

        # save results for the official evaluation
#         save_for_evaluation(path_save_for_evaluation, results_dev, 'dev')



        # save best model
        # Based on Dev Set logical accuracy lx
        acc_lx_t = acc_dev[-2]
        if acc_lx_t > acc_lx_t_best:
            acc_lx_t_best = acc_lx_t
            epoch_best = epoch
            # save best model
#             state = {'model': model.state_dict()}
#             torch.save(state, os.path.join('.', 'model_best.pt') )

#             state = {'model_bert': model_bert.state_dict()}
#             torch.save(state, os.path.join('.', 'model_bert_best.pt'))

        print(f" Best Dev lx acc: {acc_lx_t_best} at epoch: {epoch_best}")
