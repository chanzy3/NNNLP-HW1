import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from model_scratch import CNN, CNN_Multichannel
from data_loader_scratch import tokenizer, build_vocab, create_word_embedding, data_loader, MyDataSet, collate_sequence
import config
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.to(DEVICE)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, (text, target) in enumerate(train_iter):

        #target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.to(DEVICE)
            target = target.to(DEVICE)
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        #print((torch.max(prediction, 1)[1]))
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(text)
        loss.backward()
        optim.step()
        steps += 1
        
        if steps % 1000 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, (text,target) in enumerate(val_iter):

            #target = torch.autograd.Variable(target).long()
            #print(text.shape)
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(text)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
if __name__ == '__main__':
    unique_vocab = build_vocab("data_hw1")
    w2i, embedding_matrix = create_word_embedding(unique_vocab)
    train_X, train_y = data_loader('data_hw1/topicclass_train.txt', w2i)
    val_X, val_y = data_loader('data_hw1/topicclass_valid.txt', w2i)
    train_dataset = MyDataSet(train_X, train_y)
    val_dataset = MyDataSet(val_X, val_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn = collate_sequence)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, collate_fn = collate_sequence)



    #self, batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, weights):
    model = CNN_Multichannel(config.batch_size, config.output_size, config.in_channels, 
        config.out_channels, config.kernel_heights, config.stride, config.padding, config.keep_probab, len(w2i), config.embedding_length, embedding_matrix)
    loss_fn = F.cross_entropy

    for epoch in range(30):
        train_loss, train_acc = train_model(model, train_loader, epoch)
        val_loss, val_acc = eval_model(model, val_loader)
        torch.save(model.state_dict(), 'experiment10{:4f}_{:4f}'.format(epoch, val_acc))
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
        
    test_loss, test_acc = eval_model(model, test_iter)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')