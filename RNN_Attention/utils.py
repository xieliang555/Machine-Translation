from torchtext.data import Field
import torch.nn as nn
from torch import Tensor
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def init_weights(m: nn.Module):
	'''
		initialize weights by normal distribution,
		bias by constant distribution
	'''
	for name, param in m.named_parameters():
		if 'weight' in name:
			nn.init.normal_(param.data, mean=0, std=0.01)
		else:
			nn.init.constant_(param.data, 0)


def count_parameters(m: nn.Module):
	return sum(p.numel() for p in m.parameters() if p.requires_grad)



def epoch_time(stare_time: int, end_time: int):
	'''
		compute epoch duration
	'''
	duration = int(end_time - stare_time)
	duration_mins = duration // 60
	duration_secs = duration % 60

	return duration_mins, duration_secs



def itos(field: Field, idx_sequence: Tensor):
	'''
		transform index sequence to sentence
	'''
	return [field.vocab.itos[i] for i in idx_sequence]


def count_bleu(outputs: Tensor, trg: Tensor, TRG: Field):

	trg = trg.permute(1,0)
	references = [[itos(TRG, seq)] for seq in trg]

	# outputs shape: seq_len * batch * feature
	outputs = outputs.max(2)[1]
	outputs = outputs.permute(1,0)
	candidates = [itos(TRG, output) for output in outputs]

	return bleu_score(candidates, references)
 

def vis_samples(
	dataloader: DataLoader,
	writer: SummaryWriter, 
	SRC: Field, 
	TRG: Field):
	'''
		visualize a random batch 
	'''
	data_iter = iter(dataloader)
	batch = next(data_iter)
	src = batch.src
	trg = batch.trg
	src = src.permute(1,0)
	src = [' '.join(itos(SRC, seq)) for seq in src]
	trg = trg.permute(1,0)
	trg = [' '.join(itos(TRG, seq)) for seq in trg]
	writer.add_text('visualize src', str(src))
	writer.add_text('visualize trg', str(trg))
	writer.close()




