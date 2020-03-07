from torchtext.data.metrics import bleu_score


def itos(idx_list, TRG):
    sentence = [TRG.vocab.itos[idx] for idx in idx_list]
    return sentence


def count_bleu(output, trg, TRG):
    # output shape: [T, N, E]
    # trg shape: [T, N]
    # corpus level
    output = output.permute(1, 0, 2).max(2)[1]
    trg = trg.permute(1, 0)
    
    mask = trg.ne(TRG.vocab.stoi['<pad>'])
    output = output.masked_select(mask)
    trg = trg.masked_select(mask)
    candidate_corpus = [itos(output, TRG)]
    references_corpus = [[itos(trg, TRG)]]
    
    return bleu_score(candidate_corpus, references_corpus)
