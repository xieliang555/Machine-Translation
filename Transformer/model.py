import torch
import torch.nn as nn
import utils

import math
import copy


class Transformer(nn.Module):
    """
            src_vocab_size: the source vocabulary size
            tgt_vocab_size: the target vocabulary size
            d_model: the embedding feature dimension
    """

    def __init__(self, device, src_vocab_size, tgt_vocab_size, d_model=512,
                 nhead=8, num_enc_layers=6, num_dec_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(Transformer, self).__init__()
        self.device = device
        self.d_model = d_model

        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PostionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_enc_layers,
                                          num_dec_layers, dim_feedforward,
                                          dropout, activation)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def get_pad_mask(self, data):
        # the index of '<pad>' is 1
        mask = data.eq(1).transpose(0, 1)
        mask = mask.masked_fill(mask == True, float(
            '-inf')).masked_fill(mask == False, float(0.0))
        return mask

    def get_square_subsequent_mask(self, tgt):
        seq_len = tgt.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.float().masked_fill(mask == 0, float(
            0.0)).masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src, tgt):
        self.tgt_subsequent_mask = self.get_square_subsequent_mask(
            tgt).to(self.device)
        self.src_pad_mask = self.get_pad_mask(src).to(self.device)
        self.tgt_pad_mask = self.get_pad_mask(tgt).to(self.device)
        self.memory_pad_mask = self.get_pad_mask(src).to(self.device)

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.dropout(self.pos_encoder(src))
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.dropout(self.pos_encoder(tgt))
        out = self.transformer(src, tgt,
                               tgt_mask=self.tgt_subsequent_mask,
                               src_key_padding_mask=self.src_pad_mask,
                               tgt_key_padding_mask=self.tgt_pad_mask,
                               memory_key_padding_mask=self.memory_pad_mask)
        out = self.out(out)
        return out


class PostionalEncoding(nn.Module):
    """docstring for PostionEncoder"""

    def __init__(self, d_model, max_len=5000):
        super(PostionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, d_model,
                                            2).float() * math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# beam search >1 ?
def greedy_decoder(model, src, tgt, device):
    '''
        only using src to generate decoder input
    Args:
        src: [S, N]
        tgt: [T, N]
    Return:
        ['<sos>', 'w1', 'w2'...'wn']
    '''
    src_pad_mask = model.get_pad_mask(src).to(device)
    tgt_pad_mask = model.get_pad_mask(tgt).to(device)
    memory_pad_mask = model.get_pad_mask(src).to(device)
    tgt_subsequent_mask = model.get_square_subsequent_mask(tgt).to(device)
    
    src = model.embedding(src) * math.sqrt(model.d_model)
    src = model.dropout(model.pos_encoder(src))
    encoder_outputs = model.transformer.encoder(
        src, mask=None, src_key_padding_mask=src_pad_mask)
    
    ret = tgt.clone().to(device)
    tgt = model.embedding(tgt) * math.sqrt(model.d_model)
    tgt = model.dropout(model.pos_encoder(tgt))

    for t in range(1, len(ret)):
        decoder_outputs = model.transformer.decoder(
            tgt, encoder_outputs, tgt_mask=tgt_subsequent_mask,
            tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_pad_mask)
        outs = model.out(decoder_outputs)
        ret[t] = outs.max(-1)[1][t-1]
        tgt = model.embedding(ret) * math.sqrt(model.d_model)
        tgt = model.dropout(model.pos_encoder(tgt))
        
    return ret


def train(model, train_iter, criterion, optimizer, TRG, epoch, writer, device):
    model.train()
    running_loss = 0.0
    running_bleu = 0.0
    for batch_idx, batch in enumerate(train_iter):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:-1, :])
        loss = criterion(
            output.view(-1, output.shape[-1]), trg[1:, :].view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_bleu += utils.count_bleu(output, trg[1:, :], TRG)

        if batch_idx % 25 == 24:
            writer.add_scalar('train loss',
                              running_loss / 25,
                              epoch * len(train_iter) + batch_idx)
            writer.add_scalar('train bleu',
                              running_bleu / 25,
                              epoch * len(train_iter) + batch_idx)

            running_loss = 0.0
            running_bleu = 0.0

            
# combine evaluate and test
def evaluate(model, val_iter, criterion, TRG, device):
    model.eval()
    epoch_loss = 0.0
    epoch_bleu = 0.0
    for batch_idx, batch in enumerate(val_iter):
        src = batch.src.to(device)
        tgt = batch.trg.to(device)
        output = model(src, tgt[:-1, :])
        loss = criterion(
            output.view(-1, output.shape[-1]), tgt[1:, :].view(-1))

        epoch_loss += loss.item()
        epoch_bleu += utils.count_bleu(output, tgt[1:, :], TRG)

    return epoch_loss / len(val_iter), epoch_bleu / len(val_iter)


def test(model, test_iter, criterion, TRG, device):
    model.eval()
    epoch_loss = 0.0
    epoch_bleu = 0.0
    for batch_idx, batch in enumerate(test_iter):    
        src = batch.src.to(device)
        target = batch.trg.to(device)
        tgt = greedy_decoder(model, src, target[:-1, :], device)
        # to compute loss
        output = model(src, tgt)
        loss = criterion(output.view(-1, output.shape[-1]), target[1:, :].view(-1))

        epoch_loss += loss.item()
        epoch_bleu += utils.count_bleu(output, target[1:, :], TRG)
        
    return epoch_loss / len(test_iter) , epoch_bleu / len(test_iter)