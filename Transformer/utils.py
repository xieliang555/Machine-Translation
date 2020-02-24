from torchtext.data.metrics import bleu_score


def itos(idx_list, TRG):
	sentence = [TRG.vocab.itos[idx] for idx in idx_list]
	return sentence


def count_bleu(output, trg, TRG):
	# output shape: seq_len * batch_size * feature 
	# trg shape: seq_len * batch_size
	# corpus level or sentence level bleu ?
	output = output.permute(1,0,2).max(2)[1]
	trg = trg.permute(1,0)
	candidate_corpus = [itos(idx_list, TRG) for idx_list in output]
	references_corpus = [[itos(idx_list, TRG)] for idx_list in trg]
	return bleu_score(candidate_corpus, references_corpus)