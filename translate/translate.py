#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_token = 0
EOS_token = 1

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {0: "<SOS>", 1: "<EOS>", -1: "<unk>"}
        self.idx = 2 # 当前长度（包括 SOS and EOS）

    # 记录word和id之间的映射
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
        # 得到某个单词的id

    def __call__(self, word):
        if not word in self.word2idx:
            return -1
        return self.word2idx[word]

    # vaocabulary的容量
    def __len__(self):
        return self.idx

class EncoderRNN(nn.Module):
    # 在构造函数内定义了一个Embedding层和GRU层，
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # input_size代表输入语言的所有单词的数量，hidden_size是GRU网络隐藏层节点数
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size) #一个是h0的size，一个是x的size，这里简化
    # 前向传播
    def forward(self, input, hidden):
        # seq_Len=1，batch=1
        embedded = self.embedding(input).view(1, 1, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return hidden
    # 最终执行函数
    def sample(self,seq_list):
        word_inds = torch.LongTensor(seq_list).to(device)
        h = self.initHidden()
        for word_tensor in word_inds:
            h = self(word_tensor,h)
        return h
    # 初始化第一层的h0，随机生成一个
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    # 构造函数，比编码器多了全连接层和激活函数
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.maxlen = 10
        # output_size是输出语言的所有单词的数量，hidden_size是GRU网络的隐藏层节点数
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        # Linear作用是将前面GRU的输出结果变成目标语言的单词的长度
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    # 前向传播
    def forward(self, seq_input, hidden):
        output = self.embedding(seq_input).view(1, 1, -1)
        output = F.relu(output)
        output,hidden = self.gru(output,hidden)
        output = self.softmax(self.out(output[0]))
        return output,hidden
    # 最终执行函数
    def sample(self,pre_hidden):
        inputs = torch.tensor([SOS_token], device=device)
        hidden = pre_hidden
        res = [SOS_token]
        # 循环编码
        for i in range(self.maxlen):
            output,hidden=self(inputs,hidden)
            # 获取最大的索引作为生成单词的id
            topv,topi = output.topk(1)  # value,index
            #遇到句子结束符，解码结束
            if topi.item() == EOS_token:
                res.append(EOS_token)
                break
            else:
                res.append(topi.item())
            # 将生成的单词作为下一时刻的输入，squeeze去掉维度为1的维度，detach保证梯度不传导
            inputs = topi.squeeze().detach()
        return res
# 处理句子，将句子转换成Tensor
def sentence2tensor(lang,sentence):
    indexes=[lang(word) for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)

# 将（input，target）的pair都转换成Tensor
def pair2tensor(pair):
    input_tensor=sentence2tensor(lan1,pair[0])
    target_tensor=sentence2tensor(lan2,pair[1])
    return (input_tensor,target_tensor)

# 定义句子和Vocabulary类
lan1 = Vocabulary()
lan2 = Vocabulary()
# 载入数据
data = []
for line in open("cmn.txt","r",encoding='utf-8'): #设置文件对象并读取每一行文件
    data_1= line.split('CC-BY')[0].strip().split('\t')
    data.append(data_1)
# data = [['Hi .','嗨 。'],
#         ['Hi .','你 好 。'],
#         ['Run .','跑'],
#         ['Wait !','等等 ！'],
#         ['Hello !','你好 。'],
#         ['I try .','让 我 来 。'],
#         ['I won !','我 赢 了 。'],
#         ['I am OK .','我 没事 。']]
for i,j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)
print(len(lan1))
print(len(lan2))

# 定义参数
learning_rate = 0.001
hidden_size = 256

encoder = EncoderRNN(len(lan1),hidden_size).to(device)
decoder = DecoderRNN(hidden_size,len(lan2)).to(device)
# 网络参数
params = list(encoder.parameters()) + list(decoder.parameters())
# 定义优化器
optimizer = optim.Adam(params,lr = learning_rate)
loss = 0
criterion = nn.NLLLoss()   # Negative Log Likelihood Loss
# 一共训练多少轮
turns=2000
print_every = 20
print_loss_total = 0

import random
# 创建随机数据
training_pairs = [pair2tensor(random.choice(data)) for pair in range(turns)]

# 训练过程
for turn in range(turns):
    optimizer.zero_grad()
    loss = 0
    x,y =training_pairs[turn]
    input_length=x.size(0)
    target_length=y.size(0)
    # 初始化Encoder中的h0
    h=encoder.initHidden()
    # 对input进行Encoder
    for i in range(input_length):
        h = encoder(x[i],h)
    #Decoder的第一个input
    decoder_input = torch.LongTensor([SOS_token]).to(device)

    for i in range(target_length):
        decoder_output,h=decoder(decoder_input,h)
        topv,topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss=loss+criterion(decoder_output,y[i])
        if decoder_input.item() == EOS_token:
            break
    print_loss_total = print_loss_total+loss.item()/target_length
    if(turn+1) % print_every == 0:
        print('loss:{:.4f}'.format(print_loss_total/print_every))
        print_loss_total=0
    #反向传播
    loss.backward()
    #参数更新
    optimizer.step()

# 测试过程
def translate(s):
    t = [lan1(i) for i in s.split()]
    t.append(EOS_token)
    f = encoder.sample(t) #编码
    s = decoder.sample(f) #解码
    # 根据id得到单词
    r = [lan2.idx2word[i] for i in s]
    return ' '.join(r) # 生成句子
print(translate('I try .'))