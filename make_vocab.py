import torch

def n_gram(word, n=3):
    s = []
    word = '#' + word + '#'
    for i in range(len(word) - n + 1):
        s.append(word[i:i + n])
    return s

def lst_gram(lst, n=3):
    s = []
    for word in str(lst).lower().split():
        s.extend(n_gram(word))
    return s 

def hashing_trick(word, HASH_SPACE_DIM= 2000):
    return hash(word) % HASH_SPACE_DIM

def vocab_to_hash(vocab_path, HASH_SPACE_DIM = 2000):
    vocab_list = open(vocab_path, encoding='utf-8').readlines()
    # hashed_vocab = [hashing_trick(word, hash_space_dim) for word in vocab_list]
    hashed_vocab = {word: hashing_trick(word, HASH_SPACE_DIM) for word in vocab_list}
    return hashed_vocab

def n_gram_gpu(text_tensor, n=3):
    ngrams = [text_tensor[i:i+n] for i in range(len(text_tensor) - n + 1)]
    return ngrams
    
def get_batch(evid_text, batch_size=1024, start_index = 0):
    end_index = start_index + batch_size
    batch_data = evid_text[start_index:end_index]
    return batch_data, end_index

def process_batch_char(batch_data, indices_tensor, HASH_SPACE_DIM, n_gram = 3, max_length = 100):
    batch_char_list = []
    for evidence in batch_data:
        text_tensor = torch.tensor([ord(c) for c in evidence]).to("mps")
        ngrams = n_gram_gpu(text_tensor, n_gram)
        
        ngram_hashes = torch.tensor([hash(ngram) % HASH_SPACE_DIM for ngram in ngrams], dtype=torch.long).to("mps")
        hash_indices = indices_tensor[ngram_hashes]
        batch_char_list.append(hash_indices)
    
        # 在 GPU 上填充序列
    batch_char_tensor = torch.nn.utils.rnn.pad_sequence(batch_char_list, batch_first=True, padding_value=0).to("mps")
    
    # 截断或补充到 max_length
    if batch_char_tensor.size(1) > max_length:
        # 截断
        batch_char_tensor = batch_char_tensor[:, :max_length]
    else:
        # 补充
        padding_size = max_length - batch_char_tensor.size(1)
        batch_char_tensor = F.pad(batch_char_tensor, (0, padding_size), value=0)
    
    return batch_char_tensor

def load_evd_text(vocab_path, evid_text,HASH_SPACE_DIM = 2000, n_gram = 3, max_length = 100):
    hashed_vocab = vocab_to_hash(vocab_path = vocab_path,
                            HASH_SPACE_DIM = HASH_SPACE_DIM)
    ngrams_list = list(hashed_vocab.keys())  # 获取所有 n-gram 作为列表
    indices_list = list(hashed_vocab.values())  # 获取对应的哈希索引列表

    # 将 n-grams 和索引转化为 GPU 上的张量
    ngrams_tensor = torch.tensor([hash(ngram) % HASH_SPACE_DIM for ngram in ngrams_list]).to("mps")
    indices_tensor = torch.tensor(indices_list).to("mps")

    start_index = 0
    batch_size = 49152
    all_evd_chars = []
    while start_index < len(evid_text):
        print("current processed batch index",start_index)
        batch_data, start_index = get_batch(evid_text, batch_size, start_index)
        batch_char_list = process_batch_char(batch_data, indices_tensor, HASH_SPACE_DIM, n_gram = 3, max_length = 100)
        batch_char_tensor_cpu = batch_char_list.to("cpu")
        all_evd_chars.append(batch_char_tensor_cpu)
    return all_evd_chars
    
def process_batch_char(batch_data, indices_tensor, HASH_SPACE_DIM, n_gram = 3, max_length = 100):
    batch_char_list = []
    for evidence in batch_data:
        text_tensor = torch.tensor([ord(c) for c in evidence]).to("mps")
        ngrams = n_gram_gpu(text_tensor, n_gram)
        
        ngram_hashes = torch.tensor([hash(ngram) % HASH_SPACE_DIM for ngram in ngrams], dtype=torch.long).to("mps")
        hash_indices = indices_tensor[ngram_hashes]
        batch_char_list.append(hash_indices)
    
        # 在 GPU 上填充序列
    batch_char_tensor = torch.nn.utils.rnn.pad_sequence(batch_char_list, batch_first=True, padding_value=0).to("mps")
    
    # 截断或补充到 max_length
    if batch_char_tensor.size(1) > max_length:
        # 截断
        batch_char_tensor = batch_char_tensor[:, :max_length]
    else:
        # 补充
        padding_size = max_length - batch_char_tensor.size(1)
        batch_char_tensor = F.pad(batch_char_tensor, (0, padding_size), value=0)
    
    return batch_char_tensor


