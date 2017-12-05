import argparse
import operator
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_batch
from meta_optimizer import MetaModel, MetaOptimizer, FastMetaOptimizer
from model import Model
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--mode', type=int, default=0, metavar='N',
                    help='the mode of training, if 0, train a metaoptimizer, else if 1, test a metaoptimizer, else, test a sgd optimizer')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=60, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=10, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=50, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--test_optimizer_steps', type=int, default=20, metavar='N',
                    help='test optimizer step')
parser.add_argument('--train_steps', type=int, default=40,
                    help='train model steps')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.optimizer_steps % args.truncated_bptt_step == 0

kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

def test_optimizer(optimizer, cuda_id = 0):
    test_iter = iter(test_loader)
    model = Model()
    loss_sum = 0.0
    if args.cuda:
        model = model.cuda(cuda_id)
    for i in range(args.test_optimizer_steps):
        x, y = next(test_iter)
        if args.cuda:
            x, y = x.cuda(cuda_id), y.cuda(cuda_id)
        x, y = Variable(x), Variable(y)

        # First we need to compute the gradients of the model
        f_x = model(x)
        loss = F.nll_loss(f_x, y)
        model.zero_grad()
        loss.backward()
        meta_model = optimizer.meta_update(model, loss.data)
        # Compute a loss for a step the meta optimizer
        x, y = next(test_iter)
        if args.cuda:
            x, y = x.cuda(cuda_id), y.cuda(cuda_id)
        x, y = Variable(x), Variable(y)
        f_x = model(x)
        loss_new = F.nll_loss(f_x, y)
        loss_sum += loss.data[0]
    return loss_sum / args.test_optimizer_steps

def test(optimizer, cuda_id=0):
    model = Model()
    if args.cuda:
        model.cuda(cuda_id)
    loss_sum = 0.0
    step_weight = 0.1
    pre_loss = 0.0

    for i in range(len(test_dataset)):
        x, y = test_dataset[i]
        x = torch.FloatTensor(x)
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
        meta_model = optimizer.meta_update(model, loss.data)

        f_x = meta_model(x)
        loss_new = F.nll_loss(f_x, y)

        loss_sum = loss_sum * step_weight + loss.data[0] - pre_loss
        pre_loss = loss.data[0]

    return loss_sum


def train(cuda_id = 0):
    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    meta_model = Model()
    if args.cuda:
        meta_model.cuda(cuda_id)
    meta_optimizer = MetaOptimizer(MetaModel(meta_model), args.num_layers, args.hidden_size)
    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)
    if args.cuda:
        meta_optimizer.cuda(cuda_id)
    min_loss = 100000

    for epoch in range(args.max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = iter(train_loader)
        for i in range(args.updates_per_epoch):

            # Sample a new model
            model = Model()
            if args.cuda:
                model.cuda(cuda_id)

            x, y = next(train_iter)
            if args.cuda:
                x, y = x.cuda(cuda_id), y.cuda(cuda_id)
            x, y = Variable(x), Variable(y)

            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = F.nll_loss(f_x, y)
            
            step_weight = 0.1

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                # Keep states for truncated BPTT
                meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda, cuda_id = cuda_id)

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda(cuda_id)
                for j in range(args.truncated_bptt_step):
                    x, y = next(train_iter)
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
                    meta_model = meta_optimizer.meta_update(model, loss.data)

                    # Compute a loss for a step the meta optimizer
                    x, y = next(train_iter)
                    if args.cuda:
                        x, y = x.cuda(cuda_id), y.cuda(cuda_id)
                    x, y = Variable(x), Variable(y)
                    f_x = meta_model(x)
                    loss = F.nll_loss(f_x, y)
                    loss_square_sum = loss_sum * 0.1 + (loss - Variable(prev_loss)) * (loss - Variable(prev_loss))
                    loss_sum += loss - Variable(prev_loss)     

                    prev_loss = loss.data

                
                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                sum_loss = loss_square_sum + loss_sum * 10 
                sum_loss.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            # Compute relative decrease in the loss function w.r.t initial
            # value
            loss_eval = loss_sum.data[0] / args.truncated_bptt_step
            decrease_in_loss += loss_eval / initial_loss.data[0]
            final_loss += loss_eval

        test_loss = test_optimizer(meta_optimizer, cuda_id)
        if min_loss > test_loss:
            torch.save(meta_optimizer, 'optimizer.pkl')
            min_loss = test_loss

        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}, test data loss: {}".format(epoch, final_loss / args.updates_per_epoch,
                                                                       decrease_in_loss / args.updates_per_epoch, test_loss))


def Test(cuda_id = 0):
    model = Model()
    train_iter = iter(train_loader)
    if args.cuda:
        model.cuda(cuda_id)
    optimizer = torch.load('optimizer.pkl')
    for i in range(args.train_steps):
        x, y = next(train_iter)
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
        meta_model = optimizer.meta_update(model, loss.data)

        f_x = model(x)
        loss_new = F.nll_loss(f_x, y)

        print(loss.data[0])


def Test_norm(cuda_id = 0):
    model = Model()
    train_iter = iter(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    if args.cuda:
        model.cuda(cuda_id)
    for i in range(1000):
        x, y = next(train_iter)
        if args.cuda:
            x, y = x.cuda(cuda_id), y.cuda(cuda_id)
        x, y = Variable(x), Variable(y)

        # First we need to compute the gradients of the model
        f_x = model(x)
        loss = F.nll_loss(f_x, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()


        print(loss.data[0])


if __name__ == "__main__":
    if args.mode == 0:
        train()
    elif args.mode == 1:
        Test()
    else:
        Test_norm()
