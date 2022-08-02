import json
import math
import os
import pickle
import random
from collections import defaultdict
from itertools import permutations
import logging
from models.BERT import tokenization
import pdb
import dgl
import numpy as np
import torch
import re
from torch.utils.data import IterableDataset, DataLoader

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

class oldInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, speaker_ids, mention_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.speaker_ids = speaker_ids
        self.mention_ids = mention_ids

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, speaker_ids, target_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.speaker_ids = speaker_ids
        self.target_mask = target_mask
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class oldbertsProcessor(DataProcessor): #bert_s
    def __init__(self, src_file, n_class):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()
        
        def rename(d, x, y):
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y
            
        random.seed(42)
        self.D = [[], [], []]
        pdb.set_trace()
        for sid in range(3):
            with open(src_file+["/train.json", "/dev.json", "/test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(n_class):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    d, h, t = rename('\n'.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    d = [d,
                         h,
                         t,
                         rid]
                    self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))



class bertsProcessor(DataProcessor): #bert_s
    def __init__(self, src_file, n_class):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()
        
        def rename(d, x, y):
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y
            
        random.seed(42)
        self.D = [[], [], []]
        #pdb.set_trace()
        for sid in range(3):
            with open(src_file+["/train.json", "/dev.json", "/test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(n_class):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    d, h, t = ('\n'.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    d = [d,
                         h,
                         t,
                         rid]
                    self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))
                
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i][0]
            text_b = data[i][1]
            text_c = data[i][2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))
            
        return examples


def tokenize(text, tokenizer, start_mention_id):
    speaker2id = {'[unused1]' : 11, '[unused2]' : 12, 'speaker 1' : 1, 'speaker 2' : 2, 'speaker 3' : 3, 'speaker 4' : 4, 'speaker 5' : 5, 'speaker 6' : 6, 'speaker 7' : 7, 'speaker 8' : 8, 'speaker 9' : 9}
    D = ['[unused1]', '[unused2]', 'speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']
    text_tokens = []
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    speaker_ids = []
    mention_ids = []
    mention_id = start_mention_id
    speaker_id = 0
    for t in textraw:
        if t in ['speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']:
            speaker_id = speaker2id[t]
            mention_id += 1
            tokens = tokenizer.tokenize(t+" ")
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)
        elif t in ['[unused1]', '[unused2]']:
            speaker_id = speaker2id[t]
            mention_id += 1
            text += [t]
            speaker_ids.append(speaker_id)
            mention_ids.append(mention_id)
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)

    return text, speaker_ids, mention_ids

def tokenize2(text, tokenizer):
    speaker2id = {'[unused1]' : 11, '[unused2]' : 12, 'speaker 1' : 1, 'speaker 2' : 2, 'speaker 3' : 3, 'speaker 4' : 4, 'speaker 5' : 5, 'speaker 6' : 6, 'speaker 7' : 7, 'speaker 8' : 8, 'speaker 9' : 9}
    D = ['[unused1]', '[unused2]', 'speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']
    text_tokens = []
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    speaker_ids = []
    speaker_id = 0
    for t in textraw:
        if t in ['speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']:
            speaker_id = speaker2id[t]
            tokens = tokenizer.tokenize(t+" ")
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
        elif t in ['[unused1]', '[unused2]']:
            speaker_id = speaker2id[t]
            text += [t]
            speaker_ids.append(speaker_id)
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)

    return text, speaker_ids

def get_target_mask(tokens_a, tokens_c):
    '''Finds index of the [CLS] token at the start
    of the target utterance and creates mask'''
    target_mask = []
    for index, token in enumerate(tokens_a):
        
        if token == '[CLS]':
            if tokens_a[index+1: index + 1 + len(tokens_c)] == tokens_c:    
                target_mask.append(1)
            else:
                target_mask.append(0)
        else:
            target_mask.append(0)
    return target_mask
            
def remove_speaker_tokens(tokens_a, tokens_a_speaker_ids, tokens_b, tokens_c):
    '''prepends each utterance with a [CLS] token and removes speaker tokens'''
    
    speaker_tokens = ['[unused1]', '[unused2]', 'speaker']
    target_mask = []
    found_target = False
    for index, token in enumerate(tokens_a):            
        if token == 'speaker':
            #pdb.set_trace()
            #check if it is target utterance and target speaker
            if (not found_target) & (tokens_a[index+3: index + 3 + len(tokens_c)] == tokens_c) & (tokens_a[index:index + 2] == tokens_b):
                found_target = True    
                target_mask.append(1)
            else:
                target_mask.append(0)
            try:
              if re.match('[0-9]$', tokens_a[index + 1]):
                tokens_a[index] = '[CLS]'
              #get rid of speaker : tokens
                del tokens_a[index + 1:index + 3], tokens_a_speaker_ids[index + 1: index + 3]          
            except:
              pdb.set_trace()      
        else:
            target_mask.append(0)
    tokens_a.append("[SEP]")
    tokens_a_speaker_ids.append(0)
    target_mask.append(0)
    #pdb.set_trace()
    assert len(target_mask) == len(tokens_a) == len(tokens_a_speaker_ids)
    return tokens_a, tokens_a_speaker_ids, target_mask
    
def truncate_sequence(tokens, speaker_ids, target_mask, ex):
  '''makes sure sequences of tokens are the correct length
  if the target utterance is contained entirely before 512, cut off the end
  if target utterance is at the end or after, cut off from the beginning'''
  #if sequence shorter than limit we are fine
  if len(tokens) <= 512:
    return tokens, speaker_ids, target_mask
  target_index = target_mask.index(1)
  
  #target_utterance_end = 0
  #find index where target utterance ends
  target_speaker = speaker_ids[target_index]
  
  for index, speaker in enumerate(speaker_ids):
      if (index >= target_index) & (speaker_ids[index + 1] != target_speaker):
        target_utterance_end = index + 1
        break
  #if ex == 5885:
  #  pdb.set_trace()
  #if target utterance is after the max length, need to truncate from front
  if target_utterance_end > 512:
      while len(tokens) > 510:
        tokens.pop(0)
        speaker_ids.pop(0)
        target_mask.pop(0)
      #make sure sequence starts with a CLS
      if tokens[0] != ['[CLS]']:
        tokens.insert(0, '[CLS]')
        speaker_ids.insert(0, speaker_ids[0])
        target_mask.insert(0, 0)
  #if target utterance ends before end of conversation truncate the end
  else:
      while len(tokens) > 511:
        tokens.pop()
        speaker_ids.pop()
        target_mask.pop()        
      if tokens[-1] != ['[SEP]']:
        tokens.append('[SEP]')
        speaker_ids.append(0)
        target_mask.append(0)
  assert len(target_mask) == len(tokens) == len(speaker_ids)
  return tokens, speaker_ids, target_mask




def convert_examples_to_my_features(examples, max_seq_length, tokenizer):
    print("#examples", len(examples))
    speaker_tokens = ['[unused1]', '[unused2]', 'speaker 1', 'speaker 2', 'speaker 3', 'speaker 4', 'speaker 5', 'speaker 6', 'speaker 7', 'speaker 8', 'speaker 9']
    features = [[]]
    dic = {x : 0 for x in range(7)}
    for (ex_index, example) in enumerate(examples):
        #first get rid of shared utterances
        example.text_a = re.sub(', speaker [1-9]', '', example.text_a)
        example.text_b = re.sub(', speaker [1-9]', '', example.text_b)
        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids = tokenize(example.text_a, tokenizer, 0)
        tokens_b, tokens_b_speaker_ids = tokenize2(example.text_b, tokenizer)
        tokens_c, tokens_c_speaker_ids = tokenize2(example.text_c, tokenizer)
        
        #get into correct format with cls tokens
        tokens , speaker_ids, target_mask = remove_speaker_tokens(tokens_a, tokens_a_speaker_ids, tokens_b, tokens_c)
        #truncate sequences to right length
        tokens, speaker_ids, target_mask = truncate_sequence(tokens, speaker_ids, target_mask, ex_index)
        
        ###Checks###
        assert tokens[0] == '[CLS]'
        assert tokens[-1] == '[SEP]'
        assert tokens[target_mask.index(1)] == '[CLS]'
        try:
          assert tokens[target_mask.index(1) + 1:target_mask.index(1) + 1 + len(tokens_c) ] == tokens_c
        except:
          pdb.set_trace()
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        input_mask = [1] * len(input_ids)

        if sum(target_mask) != 1:
            pdb.set_trace()
        
        segment_ids = [1 for _ in range(len(input_ids))]

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            speaker_ids.append(0)
            target_mask.append(0)
            segment_ids.append(0)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        
        label_id = example.label 
        label = label_id.index(1)
        dic[label] +=1
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            
            logger.info(
                    "speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))


        features[-1].append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        label_id=label_id,
                        speaker_ids=speaker_ids,
                        target_mask = target_mask,
                        segment_ids = segment_ids
                        ))
        if len(features[-1]) == 1:
            features.append([])
    #pdb.set_trace()
    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features





def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def mention2mask(mention_id):
    slen = len(mention_id)
    mask = []
    turn_mention_ids = [i for i in range(1, np.max(mention_id) - 1)]
    for j in range(slen):
        tmp = None
        if mention_id[j] not in turn_mention_ids:
            tmp = np.zeros(slen, dtype=bool)
            tmp[j] = 1
        else:
            start = mention_id[j]
            end = mention_id[j]
            if mention_id[j] - 1 in turn_mention_ids:
                start = mention_id[j] - 1

            if mention_id[j] + 1 in turn_mention_ids:
                end = mention_id[j] + 1
            tmp = (mention_id >= start) & (mention_id <= end)
        mask.append(tmp)
    mask = np.stack(mask)
    return mask

class TUCOREGCNDataset(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type):

        super(TUCOREGCNDataset, self).__init__()

        self.data = None
        self.input_max_length = max_seq_length

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            self.data = []

            bertsProcessor_class = bertsProcessor(src_file, n_class)
            if "train" in save_file:
                examples = bertsProcessor_class.get_train_examples(save_file)
            elif "dev" in save_file:
                examples = bertsProcessor_class.get_dev_examples(save_file)
            elif "test" in save_file:
                examples = bertsProcessor_class.get_test_examples(save_file)
            else:
                print(error)
            #pdb.set_trace()
            if encoder_type == "BERT":
                features = convert_examples_to_my_features(examples, max_seq_length, tokenizer)
            else:
                features = convert_examples_to_features_roberta(examples, max_seq_length, tokenizer)

            for f in features:
                self.data.append({
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'speaker_ids': np.array(f[0].speaker_ids),
                    'label_ids': np.array(f[0].label_id),
                    'target_mask': np.array((f[0].target_mask))
                    })
            # save data
        with open(file=save_file, mode='wb') as fw:
            pickle.dump({'data': self.data}, fw)
        print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)
    
    def turn2speaker(self, turn):
        return turn.split()[1]
    
    def make_speaker_infor(self, speaker_id, mention_id):
        tmp = defaultdict(set)
        for i in range(1, len(speaker_id)):
            if speaker_id[i] == 0:
                break
            tmp[speaker_id[i]].add(mention_id[i])
        
        speaker_infor = dict()
        for k, va in tmp.items():
            speaker_infor[k] = list(va)
        return speaker_infor
    
 
class TUCOREGCNDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, relation_num=36, max_length=512):
        super(TUCOREGCNDataloader, self).__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length

        self.relation_num = relation_num

        self.order = list(range(self.length))

    def __iter__(self):
        # shuffle
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)]

        # begin
        input_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        segment_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        input_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        speaker_id = torch.LongTensor(self.batch_size, self.max_length).cpu()
        target_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        label_ids = torch.Tensor(self.batch_size, self.relation_num).cpu()

        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch)

            for mapping in [input_ids, segment_ids, input_mask,  label_ids, target_mask, speaker_id]:
                if mapping is not None:
                    mapping.zero_()

            graph_list = []

            for i, example in enumerate(minibatch):
                mini_input_ids, mini_segment_ids, mini_input_mask, mini_label_ids, mini_speaker_id, mini_target_mask = \
                    example['input_ids'], example['segment_ids'], example['input_mask'], example['label_ids'], \
                    example['speaker_ids'], example['target_mask']

                word_num = mini_input_ids.shape[0]
                relation_num = mini_label_ids.shape[0]

                input_ids[i, :word_num].copy_(torch.from_numpy(mini_input_ids))
                segment_ids[i, :word_num].copy_(torch.from_numpy(mini_segment_ids))
                input_mask[i, :word_num].copy_(torch.from_numpy(mini_input_mask))
                speaker_id[i, :word_num].copy_(torch.from_numpy(mini_speaker_id))
                target_mask[i, :word_num].copy_(torch.from_numpy(mini_target_mask))
                label_ids[i, :relation_num].copy_(torch.from_numpy(mini_label_ids))

            context_word_mask = input_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()
            yield {'input_ids': get_cuda(input_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'segment_ids': get_cuda(segment_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'input_masks': get_cuda(input_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'speaker_ids': get_cuda(speaker_id[:cur_bsz, :batch_max_length].contiguous()),
                   'label_ids': get_cuda(label_ids[:cur_bsz, :self.relation_num].contiguous()),
                   'target_mask': get_cuda(target_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'graphs': graph_list
                   }
