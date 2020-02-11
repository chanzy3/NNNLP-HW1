import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNN(nn.Module):
	def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, 
		vocab_size, embedding_length, weights, fine_tune = False):
		super(CNN, self).__init__()
		

		self.batch_size = batch_size
		self.output_size = output_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_heights = kernel_heights
		self.stride = stride
		self.padding = padding
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=fine_tune)
		self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
		self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
		self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
		self.dropout = nn.Dropout(keep_probab)
		self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)
	
	def conv_block(self, input, conv_layer):
		conv_out = conv_layer(input)
		activation = F.relu(conv_out.squeeze(3))
		max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
		
		return max_out
	
	def forward(self, input_sentences, batch_size=None):
	
		
		input = self.word_embeddings(input_sentences)
		input = input.unsqueeze(1)
		max_out1 = self.conv_block(input, self.conv1)
		max_out2 = self.conv_block(input, self.conv2)
		max_out3 = self.conv_block(input, self.conv3)
		
		all_out = torch.cat((max_out1, max_out2, max_out3), 1)
		fc_in = self.dropout(all_out)
		logits = self.label(fc_in)
		
		return logits

class CNN_Multichannel(nn.Module):
	def __init__(self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, 
		vocab_size, embedding_length, weights):
		super(CNN_Multichannel, self).__init__()
		

		self.batch_size = batch_size
		self.output_size = output_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_heights = kernel_heights
		self.stride = stride
		self.padding = padding
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.fixed_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.fixed_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.tuned_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.tuned_embeddings.weight = nn.Parameter(weights, requires_grad=True)

		self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
		self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
		self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
		self.conv_lst = nn.ModuleList([self.conv1, self.conv2, self.conv3])
		self.dropout = nn.Dropout(keep_probab)
		self.label = nn.Linear(len(kernel_heights)*out_channels, output_size)
	
	def conv_block(self, input, conv_layer):
		conv_out = conv_layer(input)
		activation = F.relu(conv_out.squeeze(3))
		max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
		
		return max_out
	
	def forward(self, input_sentences, batch_size=None):
	
		
		fixed_input = self.fixed_embeddings(input_sentences)
		tuned_input = self.tuned_embeddings(input_sentences)
		fixed_input = fixed_input.unsqueeze(1)
		tuned_input = tuned_input.unsqueeze(1)
		all_out = []
		for conv in self.conv_lst:
			comb_emb = torch.add(fixed_input, tuned_input)
			pooled = self.conv_block(comb_emb, conv)
			all_out.append(pooled)

		
		all_out = torch.cat(all_out, 1)
		fc_in = self.dropout(all_out)
		logits = self.label(fc_in)
		
		return logits