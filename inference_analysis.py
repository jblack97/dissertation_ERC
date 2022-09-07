#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import os
import logging
import argparse
import random
import pickle
import shutil
import itertools
#from tqdm import tqdm, trange
from tqdm.notebook import tqdm , trange
import numpy as np
import torch
import pdb
from BERT import tokenization
from BERT.BERT import BertConfig, Bert_only


from optimization import BERTAdam

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from hyper_param_search import bert_args


import json
import re

from data import TUCOREGCNDataset, TUCOREGCNDataloader

n_classes = {
        "DialogRE": 36,
        "MELD": 7,
        "EmoryNLP": 7,
        "DailyDialog": 7,
    }

reverse_order = False
sa_step = False


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def calc_test_result(logits, labels_all, data_name):
    #pdb.set_trace()
    true_label=[]
    predicted_label=[]
    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    for i in range(len(logits)):
        true_label.append(np.argmax(labels_all[i]))
        predicted_label.append(np.argmax(logits[i]))

    conf_matrix = confusion_matrix(true_label, predicted_label)

    
    p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(true_label, predicted_label, average=None)

    return p_weighted, r_weighted, f_weighted, support_weighted, conf_matrix

def get_logits4eval_ERC(model, dataloader, savefile, resultsavefile, device, data_name):
    model.eval()
    logits_all = []
    labels_all = []
    for batch in tqdm(dataloader, desc="Iteration"):
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        input_masks = batch['input_masks'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        label_ids = batch['label_ids'].to(device)
        target_mask = batch['target_mask'].to(device)


        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, speaker_ids=speaker_ids, labels=label_ids, target_mask=target_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        for i in range(len(logits)):
            logits_all += [logits[i]]
        for i in range(len(label_ids)):
            labels_all.append(label_ids[i])
    
    p_weighted, r_weighted, f_weighted, support_weighted, conf_matrix = calc_test_result(logits_all, labels_all, data_name)

    return f_weighted, conf_matrix



def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_examples', default = 100000000000000000, type = int)
    parser.add_argument('--model_name', required = True, type = str)
    parser.add_argument("--encoder_type",
                        default='BERT',
                        type=str,
                        help="The type of pre-trained model.")
    parser.add_argument('--hidden_dropout_prob', default = 0.05, type = float)
    parser.add_argument('--attention_probs_dropout_prob', default = 0.05, type = float)
    parser.add_argument('--score_json_path', 
                        default = None, 
                        type = str,
                        help = 'Json file containing best f1 scores and hyperparameter sets for each model type')
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the dataset to train.")

    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument("--merges_file",
                        default=None,
                        type=str,
                        help="The merges file that the RoBERTa model was trained on.")
    
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=6e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=666,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume the training.")
    parser.add_argument("--f1eval",
                        default=True,
                        action='store_true',
                        help="Whether to use f1 for dev evaluation during training.")

    parser.add_argument("--log",
                        default=True,
                        action='store_true',
                        help="Whether to use wandb for logging")
    base_args = parser.parse_args()
    args = base_args
    
    n_class = n_classes[args.data_name]

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    
    arg_dict = vars(base_args)
    args = base_args
    
    #set arguments in config
    bert_args(args.config_file, args.attention_probs_dropout_prob, args.hidden_dropout_prob)
    #load config
    config = BertConfig.from_json_file(args.config_file)
    #load model
    model = Bert_only(config, n_class)

    
    test_set = TUCOREGCNDataset(src_file=args.data_dir, save_file=args.data_dir + "/test_" + 'BERT' + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type)
    test_loader = TUCOREGCNDataloader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=7, max_length=args.max_seq_length)
    bert_args(args.config_file, args.attention_probs_dropout_prob, args.hidden_dropout_prob)

    
    for model_num in range(3):

        model_path = os.path.join('/content/drive/MyDrive/thesis/models/', f"fine_tuned_models/{args.model_name}_{model_num}.pt")

        model.load_state_dict(torch.load(model_path))
        model.to(device)
        if model_num == 0:
            test_f1, conf_matrix = get_logits4eval_ERC(model, test_loader, None, None, device, args.data_name)
        else:
            temp_test_f1, temp_conf_matrix = get_logits4eval_ERC(model, test_loader, None, None, device, args.data_name)
        
            test_f1 += temp_test_f1
            conf_matrix += temp_conf_matrix
    test_f1 /= 3
    conf_matrix /=3

    print(test_f1)
    print(conf_matrix)

if __name__ == '__main__':
    main()