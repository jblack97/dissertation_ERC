import os
import sys
import copy
import pdb
import numpy as np
import tensorflow as tf
import torch
import transformers
import argparse
import json
import math
import six
import torch
import torch.nn as nn
import re
from models.BERT.BERT import BertConfig, BertModel



def main():
    '''Converts MPC-BERT model from pytorch to tensorflow'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint', type = str)
    parser.add_argument('--config_file', type = str)
    parser.add_argument('--pytorch_dump_path', type = str)

    args = parser.parse_args()
    # Initialise PyTorch model
    config = BertConfig.from_json_file(args.config_file)
    model = BertModel(config)
    #model.embeddings.speaker_embedding = torch.nn.Embedding(16,768)

    # Load weights from TF model
    path = args.tf_checkpoint
    
    print("Converting TensorFlow checkpoint from {}".format(path))

    init_vars = tf.train.list_variables(path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading {} with shape {}".format(name, shape))
        array = tf.train.load_variable(path, name)
        print("Numpy array shape {}".format(array.shape))
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "bert/"
        print("Loading {}".format(name))
        name = name.split('/')
        #pdb.set_trace()
        #if name[1] == 'speaker_embedding':
          #pdb.set_trace()
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer",
                  "AdamWeightDecayOptimizer_1", "l_step",
                  'ddressee_recognize', 'ointer_consistency_distinct',
                  'hared_node_detect', 'hared_utterance_restore', 'peaker_restore']
            for n in name
        ):
          print('Skipping')
          continue
        if name[0] in ['redictions', 'eq_relationship']:
            print("Skipping")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings' or m_name[-10:] == '_embedding':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    torch.save(model.state_dict(), args.pytorch_dump_path)


    
if __name__ == "__main__":
    main()