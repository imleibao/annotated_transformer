# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import seaborn
seaborn.set_context(context='talk')


class EncoderDecoder(nn.Module):
    """
    包含了encoder 和 decoder 的最外层模型
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        encoder: an Encoder object
        decoder: a Decoder object
        src_embed: an nn.Sequential object, source language word embedding
            generator which contain regular embeding and positional info, see
            function make_model
        tgt_embed: similar to src_embed
        generator: a Generator object
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        """
        src:a batch of samples, each row present indexs of word in that sample.
            dim= #samples * #words_sentence(all samples have the same length)
        src_mask: boolean matrix, mask element where the value is True
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        memory: the output of encoder
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        args: same as self.encode and self.decode
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    """
    a linear projection followed by a softmax function to convert the decoder
    output to predicted next-token probabilities. 
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers i.e. create N  modules"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        一个 layer 就是一个 Attention Block, 论文中用了6个.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    构建一个实现了 LayerNorm 的 Module, 即下面定义的 Encoderlayer 对象.
    """
    def __init__(self, features, eps=1e-6):
        """
        features: shape of features need to be normalized
        eps: epsilon, using to avoid division by 0.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    实现一个 residual connection 的模组,在这里为了编程方便,把论文中对每一层
    的输出的正则化处理挪到前面,相当于在每一层的输入的时候对输入(也就是上一层的
    输出)做正则化处理.
    该模组接受 sublayer 参数, 以泛化处理 FFN 和 multi-head Attention 中的 residual
    connection
    """
    def __init__(self, size, dropout):
        """
        args:
           size: d_model
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    该对象是一个 Transformer block, 包含了一个 self-attn, 和一个 feed forward
    网络.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        args:
            size: d_model
            self_attn: MultiHeadedAttention object
            feed_forward: PositionwiseFeedForward object
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        x dim = sample * d_model
        mask is a boolean array, dim = sample * d_model
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking, 每个 layer 就是一个 Block"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: 目标语言的词向量
        memory: Encoder 的最终输出
        return: 
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward(defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        size: d_model
        self_attn: MultiHeadedAttention object
        feed_forward: PositionalEncoding object
        dropout: float hyper parameter
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        args:
            x: 目标语言的词向量
            memory: encoder 的最终输出
            src_mask: 
            tgt_mask:
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    在 decoder 的 self-attention 中为了避免处理某位置时能看到其后未来的位置的
    信息, 保证在预测 i 时刻时只能依靠已知的 i 时刻之前的输出.
  
    Args:
        size : 需要遮盖的输入大小
    return:
        返回一个 size 大小的上三角 tensor, 其对角线为 0.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """
    when comes to decoder,  x,x,x,tgt_mask or x,m,m,src_mask
    Args:
        query: #samples * #heads * #words_of_query_sample * d_k
        key: #samples * #heads * #words_of_key_sample * d_k
        value:
        mask: #
        dropout:

    scores = f(query, key):
        #samples * #heads * #words_of_query_sample * #words_of_key_sample
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        """
        h : parallel attention layers, or heads. 论文设置为 8,
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 克隆四层, 包括了由输入 embedding 计算出 query, key, value 的三层和
        # 将所有 heads 的输出连接起来后的最后一层 Linear projection
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"
    """ 
    两层网络,中间有个 Relu 激活函数
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.droupout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.droupout(F.relu(self.w_1(x))))

    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        args:
            d_model
            vocab: int
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    """
     args:
        max_len: int 表示输入序列的最大长度
        d_model: 单个词的词向量的维度
        dropout: float drop 率
    return:
        positional encodings: dim = (1, max_len, d_model)
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
       
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8,
               dropout=0.1):
    """
    args:
        src_vocab: int,
        tgt_vocab: int,
    return:
        the transformer model which is an EncoderDecoder object, 
    """
    cp = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, cp(attn), cp(ff), dropout), N),
        Decoder(DecoderLayer(d_model, cp(attn), cp(attn), cp(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), cp(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), cp(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training"
    def __init__(self, src, trg=None, pad=0):
        """
        src: n*m 的索引矩阵,每个元素表示该 token 的 embedding 的索引. n表示这个
        batch 有多少样本, 一个样本一条语句, m 表示每个语句有多少词
        trg: src 对应的目标语言翻译结果的索引矩阵, 是 gold truth
        pad: 用于对齐的填充索引值, pad = 0 表示, 在一个样本中,如果其长度不足需
             需要对齐, 那么在 src 和 trg 中对应的位置的索引值填充为0
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]    # 用于计算损失函数
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_mask != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a make to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss.detach().numpy()
        total_tokens += batch.ntokens.numpy()
        tokens += batch.ntokens.numpy()
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss.detach().numpy() / batch.ntokens.numpy(), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

    
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    """
    V: vocabulary size
    batch: 每个batch 的样本数
    nbatches: batch 数
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function. 作为参数传给run_epoch"
    """
    计算 loss, 并执行 backpopagate 计算梯度(xxx.backward()), 并更新参数
    (xxx.step())
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.float()

# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model, 
                    SimpleLossCompute(model.generator, criterion, None)))


from torchtext import data, datasets

if True:
    import spacy
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)
    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
