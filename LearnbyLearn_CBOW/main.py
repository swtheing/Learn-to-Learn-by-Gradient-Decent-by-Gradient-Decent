import argparse
import operator
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from meta_optimizer import MetaModel, MetaOptimizer, FastMetaOptimizer
from model import Model
from torch.autograd import Variable
from Preprocess import read_tsv, read_dic, Batch_Gen
import os

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--model', type=int, default=0, metavar='CBOW',
        help='0: CBOW; 1: CBOW_Learn')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=1000, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=100, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=50, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=4, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--embedding_dim', type=int, default=50, metavar='N',
                    help='embedding size of Model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cuda_id', type=int, default=0, metavar='N',
                            help='cuda_id (default: 0)')
parser.add_argument('--test_optimizer_steps', type=int, default=20, metavar='N',
                            help='test optimizer step')
parser.add_argument('--train_steps', type=int, default=100000,
                            help='train model steps')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.optimizer_steps % args.truncated_bptt_step == 0

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def test_optimizer(optimizer, context, dic, score, cuda_id = 0):
    model = Model(len(dic) + 1, args.embedding_dim)
    if args.cuda:
        model.cuda(cuda_id)
    loss_sum = 0.0 
    for i in range(args.test_optimizer_steps):
        x, y, score_t = Batch_Gen(dic, context, score, args.batch_size, 1722, len(context), 2)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        if args.cuda:
            x, y = x.cuda(cuda_id), y.cuda(cuda_id)
        x, y = Variable(x), Variable(y)

        # First we need to compute the gradients of the model
        f_x = model(x)
        loss = F.nll_loss(f_x, y)
        model.zero_grad()
        loss.backward()

        # Perfom a meta update using gradients from model
        # and return the current meta model saved in the optimizer
        meta_model = optimizer.meta_update(model, loss.data, skip = [0, 2], lr = 0.01)

        loss_sum +=loss.data[0]

    return loss_sum / args.test_optimizer_steps
    
def train_optimizer(cuda_id = 0):
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    id, context, score = read_tsv("clean.tsv")
    dic = read_dic()
    NLL_loss = nn.NLLLoss()
    meta_model = Model(len(dic) + 1, args.embedding_dim)
    if args.cuda:
        meta_model.cuda(cuda_id)
    #meta_net = torch.nn.DataParallel(meta_model, device_ids=[1, 2, 5])

    meta_optimizer = MetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    if args.cuda:
        meta_optimizer.cuda(cuda_id)

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)
    
    min_loss = 10000.0
    for epoch in range(args.max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        for i in range(args.updates_per_epoch):

            # Sample a new model
            model = Model(len(dic) + 1, args.embedding_dim)
            if args.cuda:
                model.cuda(cuda_id)
            #net = torch.nn.DataParallel(model, device_ids=[1, 2, 5])

            x, y, score_t = Batch_Gen(dic, context, score, args.batch_size, 0, len(context), 2)
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)
            if args.cuda:
                x, y = x.cuda(cuda_id), y.cuda(cuda_id)
            x, y = Variable(x), Variable(y)

            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = F.nll_loss(f_x, y)


            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda, cuda_id = cuda_id)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda(cuda_id)
                for j in range(args.truncated_bptt_step):
                    x, y, score_t = Batch_Gen(dic, context, score, args.batch_size, 0, 1722, 2)
                    x = torch.LongTensor(x)
                    y = torch.LongTensor(y)
                    if args.cuda:
                        x, y = x.cuda(cuda_id), y.cuda(cuda_id)
                    x, y = Variable(x), Variable(y)

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    loss = F.nll_loss(f_x, y)
                    model.zero_grad()
                    loss.backward()

                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, loss.data, skip = [0, 2], lr = 0.01)

                    # Compute a loss for a step the meta optimizer
                    x, y, score_t = Batch_Gen(dic, context, score, args.batch_size, 0, 1722, 2)
                    x = torch.LongTensor(x)
                    y = torch.LongTensor(y)
                    if args.cuda:
                        x, y = x.cuda(cuda_id), y.cuda(cuda_id)
                    x, y = Variable(x), Variable(y)
                    f_x = meta_model(x)
                    loss = F.nll_loss(f_x, y)
                    loss_sum += (loss - Variable(prev_loss))
                    prev_loss = loss.data
 
                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            # Compute relative decrease in the loss function w.r.t initial
            # value
            decrease_in_loss += loss.data[0] / initial_loss.data[0]
            final_loss += loss.data[0]
        test_loss = test_optimizer(meta_optimizer, context, dic, score, cuda_id)
        if min_loss > test_loss:
            torch.save(meta_optimizer, 'optimizer.pkl')
            min_loss = test_loss
        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}, test data loss: {}".format(epoch, final_loss / args.updates_per_epoch,
                                                                              decrease_in_loss / args.updates_per_epoch, test_loss))

def train(cuda_id = 0):
    id, context, score = read_tsv("clean.tsv")
    dic = read_dic()
    meta_optimizer = torch.load('optimizer.pkl')
    model = Model(len(dic) + 1, args.embedding_dim)
    if args.cuda:
        model.cuda(cuda_id)
    if os.path.exists('CBOW_params.pkl'):
        model.load_state_dict(torch.load('CBOW_params.pkl'))
    loss_sum = 0.0
    for i in range(args.train_steps):
        x, y, score_t = Batch_Gen(dic, context, score, args.batch_size, 0, 1722, 2)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        if args.cuda:
            x, y = x.cuda(cuda_id), y.cuda(cuda_id)
        x, y = Variable(x), Variable(y)

        # First we need to compute the gradients of the model
        f_x = model(x)
        loss = F.nll_loss(f_x, y)
        model.zero_grad()
        loss.backward()

        # Perfom a meta update using gradients from model
        # and return the current meta model saved in the optimizer
        meta_model = meta_optimizer.meta_update(model, loss.data, skip = [0, 2], lr = 0.01)
        loss_sum += loss.data[0]
        if i % 100 == 0:
            torch.save(meta_model.state_dict(), 'CBOW_params.pkl')
            print(loss_sum / 100)
            loss_sum = 0.0


def Train_CBOW(cuda_id = 0):
    id, context, score = read_tsv("clean.tsv")
    dic = read_dic()
    NLL_loss = nn.NLLLoss()
    model = Model(len(dic) + 1, args.embedding_dim)
    if args.cuda:
        model.cuda(cuda_id)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    if os.path.exists('CBOW_params.pkl'):
        model.load_state_dict(torch.load('CBOW_params.pkl'))
        print("load")

    total_loss = 0.0
    #if args.cuda:
    #    total_loss = total_loss.cuda(cuda_id)
    for epoch in range(100000):
        data, target, score_t = Batch_Gen(dic, context, score, args.batch_size, 0, len(context), 2)
        context_var = torch.LongTensor(data)
        target = torch.LongTensor(target)
        if args.cuda:
            context_var = context_var.cuda(cuda_id)
            target = target.cuda(cuda_id)
        context_var = Variable(context_var)
        target = Variable(target)

        model.zero_grad()
        log_probs = model(context_var)
        loss = NLL_loss(log_probs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]
        if epoch % 100 == 0:
            torch.save(model.state_dict(), 'CBOW_params.pkl')
            print(total_loss / (epoch+1))

if __name__ == "__main__":
    if args.model == 0:
        Train_CBOW(args.cuda_id)
    else:
        #train_optimizer(args.cuda_id)
        train(args.cuda_id)
