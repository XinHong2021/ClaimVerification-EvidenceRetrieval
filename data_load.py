import json 
from make_vocab import *
import numpy as np

def load_vocab():
    vocab = open('./data/vocab.txt', encoding='utf-8').readlines()
    slice2idx = {}
    idx2sclie = {}
    cnt = 0
    for char in vocab:
        char = char.strip('\n')
        slice2idx[char] = cnt
        idx2sclie[cnt] = char
        cnt += 1
    return slice2idx, idx2sclie

def padding(text, maxlen=100):
    pad_text = []
    for sentence in text:
        pad_sentence = np.zeros(maxlen).astype('int64')
        cnt = 0
        for index in sentence:
            pad_sentence[cnt] = index
            cnt += 1
            if cnt == maxlen:
                break
        pad_text.append(pad_sentence)
    return pad_text

def char_index(text_a, text_b):
    slice2idx, idx2sclie = load_vocab()
    text_a_ids, text_b_ids = [], []
    for a_sentence, b_sentence in zip(text_a, text_b):
        # text_a和text_b的数据格式一样
        a = []
        b = []
        for slice in lst_gram(a_sentence, n=3):
            if slice in slice2idx.keys():
                a.append(slice2idx[slice])
            else:
                a.append(1) #没被收录的vocab被记做‘UNK
        for slice in lst_gram(b_sentence, n=3):
            if slice in slice2idx.keys():
                b.append(slice2idx[slice])
            else:
                b.append(1)
        text_a_ids.append(a)
        text_b_ids.append(b)
    a_list = padding(text_a_ids)
    b_list = padding(text_b_ids)
    return a_list, b_list

def load_char_data(vocab_path, claim_df, evid_df, HASH_SPACE_DIM = 2000, n_gram = 3, max_length = 100, NEG = None):
    text_a, text_b, label = [], [], []
    for claim_id, values in claim_df.items():
        tt_evd_dict = {}
        if NEG:
            tt_evds = values['evidences'] + values['neg_evidences']
            for i in values['evidences']:
                tt_evd_dict[i] = 1
            for i in values['neg_evidences']:
                tt_evd_dict[i] = 0
        else:
            tt_evds = values['evidences']
            for i in values['evidences']:
                tt_evd_dict[i] = 1

        for i, evid_id in enumerate(tt_evd_dict):
            # remove the grouth truth not in climate evidence 
            if evid_id in evid_df.keys():
                text_a.append(claim_df[claim_id]['norm_claim'])
                text_b.append(evid_df[evid_id])
                if tt_evd_dict[evid_id] == 1:
                    label.append(1)
                else:
                    label.append(0)
    # a_index, b_index = char_index(text_a, text_b)
    hashed_vocab = vocab_to_hash(vocab_path = vocab_path,
                            HASH_SPACE_DIM = HASH_SPACE_DIM)
    ngrams_list = list(hashed_vocab.keys())  # 获取所有 n-gram 作为列表
    indices_list = list(hashed_vocab.values())  # 获取对应的哈希索引列表
    # 将 n-grams 和索引转化为 GPU 上的张量
    ngrams_tensor = torch.tensor([hash(ngram) % HASH_SPACE_DIM for ngram in ngrams_list]).to("mps")
    indices_tensor = torch.tensor(indices_list).to("mps")

    claim_index = process_batch_char(text_a, indices_tensor, 
                        HASH_SPACE_DIM, n_gram = 3, max_length = 100)
    evidence_index = process_batch_char(text_b, indices_tensor,
                        HASH_SPACE_DIM, n_gram = 3, max_length = 100)

    return claim_index, evidence_index, label

def load_all_evidences(evidence_path):
    with open(evidence_path, 'r', encoding='utf-8') as f:
        evidences = json.load(f)
    
    text_a, text_b = [], []
    for key, value in evidences.items():
        text_b.append(value)
        text_a.append(' ')
    assert len(text_a) == len(text_b)
    label = [0]*len(text_b)
    a_index, b_index = char_index(text_a, text_b)
    return a_index, b_index, label

def load_evd_text(vocab_path, evid_text, HASH_SPACE_DIM = 2000, n_gram = 3, max_length = 100, batch_size = 49152):
    hashed_vocab = vocab_to_hash(vocab_path = vocab_path,
                            HASH_SPACE_DIM = HASH_SPACE_DIM)
    ngrams_list = list(hashed_vocab.keys())  # 获取所有 n-gram 作为列表
    indices_list = list(hashed_vocab.values())  # 获取对应的哈希索引列表
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # 将 n-grams 和索引转化为 GPU 上的张量
    ngrams_tensor = torch.tensor([hash(ngram) % HASH_SPACE_DIM for ngram in ngrams_list]).to("mps")
    indices_tensor = torch.tensor(indices_list).to("mps")

    start_index = 0
    all_evd_chars = []
    while start_index < len(evid_text):
        # print("current processed batch index",start_index)
        batch_data, start_index = get_batch(evid_text, batch_size, start_index)
        batch_char_list = process_batch_char(batch_data, indices_tensor, HASH_SPACE_DIM, n_gram = 3, max_length = 100)
        batch_char_tensor_cpu = batch_char_list.to("cpu")
        all_evd_chars.append(batch_char_tensor_cpu)
    return all_evd_chars

def load_claim_text_to_char(vocab_path, claim_df, HASH_SPACE_DIM, n_gram = 3, max_length = 100, batch_size = 49152):
    hashed_vocab = vocab_to_hash(vocab_path = vocab_path,
                            HASH_SPACE_DIM = HASH_SPACE_DIM)    
    ngrams_list = list(hashed_vocab.keys())  # 获取所有 n-gram 作为列表
    indices_list = list(hashed_vocab.values())  # 获取对应的哈希索引列表
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # 将 n-grams 和索引转化为 GPU 上的张量
    ngrams_tensor = torch.tensor([hash(ngram) % HASH_SPACE_DIM for ngram in ngrams_list]).to("mps")
    indices_tensor = torch.tensor(indices_list).to("mps")
    all_char_list = []
    for claim_id, values in claim_df.items():
        claim_text = values['norm_claim']
        text_tensor = torch.tensor([ord(c) for c in claim_text]).to("mps")
        ngrams = n_gram_gpu(text_tensor, n_gram)

        ngram_hashes = torch.tensor([hash(ngram) % HASH_SPACE_DIM for ngram in ngrams], dtype=torch.long).to("mps")
        hash_indices = indices_tensor[ngram_hashes]

        all_char_list.append(hash_indices)
        # 在 GPU 上填充序列
    all_char_list = torch.nn.utils.rnn.pad_sequence(all_char_list, batch_first=True, padding_value=0).to("mps")
    # 截断或补充到 max_length
    if all_char_list.size(1) > max_length:
        # 截断
        all_char_list = all_char_list[:, :max_length]
    else:
        # 补充
        padding_size = max_length - all_char_list.size(1)
        all_char_list = F.pad(all_char_list, (0, padding_size), value=0)

    return all_char_list