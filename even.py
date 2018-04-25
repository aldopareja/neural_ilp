import scipy.io
import numpy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb
from copy import deepcopy
import pdb

num_constants = 7
# Background
zero_extension = torch.zeros(1, num_constants)
zero_extension[0, 0] = 1
succ_extension = torch.eye(num_constants - 1, num_constants - 1)
succ_extension = torch.cat((torch.zeros(num_constants - 1, 1), succ_extension), 1)
succ_extension = torch.cat((succ_extension, torch.zeros(1, num_constants)), 0)

# Intensional Predicates
aux_extension = torch.zeros(num_constants, num_constants)
even_extension = torch.zeros(1, num_constants)

valuation_init = [Variable(zero_extension), Variable(succ_extension), Variable(aux_extension), Variable(even_extension)]

# Target
target = torch.zeros(num_constants, num_constants)
target[0] = torch.zeros(1, num_constants)
odd = [0, 2, 4, 6]
for integer in odd:
    target[0, integer] = 1


def decoder_efficient(valuation, step):
    ## Create valuation_new
    valuation_new = [deepcopy(valuation[0]), deepcopy(valuation[1]), Variable(torch.zeros(valuation[2].size())),
                     Variable(torch.zeros(valuation[3].size()))]

    ##Unifications
    rules_aux = torch.cat((rules[:, :num_feat], rules[:, num_feat:2 * num_feat], rules[:, 2 * num_feat:3 * num_feat]),
                          0)
    rules_aux = rules_aux.repeat(num_predicates, 1)
    embeddings_aux = embeddings.repeat(1, num_rules * 3).view(-1, num_feat)
    # unifs = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates,-1)

    unifs = F.pairwise_distance(embeddings_aux, rules_aux).view(num_predicates, -1)
    unifs = torch.exp(-unifs)
    unifs_sum = torch.sum(unifs, 0)
    unifs = unifs / unifs_sum

    ##Get_Valuations
    for predicate in intensional_predicates:
        if valuation[predicate].size()[0] == 1:
            for s in range(num_constants):
                valuation_aux = Variable(torch.Tensor([0]))
                for body1 in range(num_predicates):
                    for body2 in range(num_predicates):
                        ## Get nums
                        if valuation[body1].size()[0] == 1:
                            if valuation[body2].size()[0] == 1:
                                num = torch.min(valuation[body1][0, s], valuation[body2][0, s])
                            else:
                                num = torch.min(valuation[body1][0, :], valuation[body2][:, s])
                                num = torch.max(num)
                        else:
                            if valuation[body2].size()[0] == 1:
                                num = torch.min(valuation[body1][:, s], valuation[body2][0, s])
                                num = torch.max(num)
                            else:
                                num = 0

                        ## max across three rules
                        new = Variable(torch.Tensor([0]))
                        for rule in range(num_rules):
                            unif = unifs[predicate][rule] * unifs[body1][num_rules + rule] * unifs[body2][
                                2 * num_rules + rule]
                            new = torch.max(new, unif)

                        num = num * new
                        valuation_aux = torch.max(valuation_aux, num)
                valuation_new[predicate][0, s] = torch.max(valuation[predicate][0, s], valuation_aux)



        else:
            for s in range(num_constants):
                for o in range(num_constants):
                    valuation_aux = Variable(torch.Tensor([0]))
                    for body1 in range(num_predicates):
                        for body2 in range(num_predicates):
                            ## Get nums
                            if valuation[body1].size()[0] == 1:
                                if valuation[body2].size()[0] == 1:
                                    num = torch.min(valuation[body1][0, s], valuation[body2][0, o])
                                else:
                                    num = torch.min(valuation[body1][0, s], valuation[body2][s, o])
                                    # num = torch.max(num)
                            else:
                                if valuation[body2].size()[0] == 1:
                                    num = torch.min(valuation[body1][s, o], valuation[body2][0, o])
                                    # num = torch.max(num)
                                else:
                                    num = torch.min(valuation[body1][s, :], valuation[body2][:, o])
                                    num = torch.max(num)

                            ## max across three rules
                            new = Variable(torch.Tensor([0]))
                            for rule in range(num_rules):
                                unif = unifs[predicate][rule] * unifs[body1][num_rules + rule] * unifs[body2][
                                    2 * num_rules + rule]
                                new = torch.max(new, unif)
                                # could be amalgamate

                            num = num * new
                            valuation_aux = torch.max(valuation_aux, num)
                    valuation_new[predicate][s, o] = torch.max(valuation[predicate][s, o], valuation_aux)

    return valuation_new


def amalgamate(x, y):
    return x + y - x * y


def rbf(x, y):
    ans = (x - y) ** 2
    ans = torch.mean(ans, 1)
    ans = torch.exp(-ans)
    return ans


num_iters = 50
learning_rate = .1
steps = 4

num_feat = 4
num_rules = 3
num_predicates = 4
intensional_predicates = [2, 3]
num_intensional_predicates = len(intensional_predicates)

embeddings = Variable(torch.rand(num_predicates, num_feat), requires_grad=True)
# embeddings = Variable(torch.eye(4), requires_grad=True)

rules = Variable(torch.rand(num_rules, num_feat * 3), requires_grad=True)
# rule1 = torch.Tensor([0,0,0,1,1,0,0,0,0,1,0,0]).view(1,-1)
# rule2 = torch.Tensor([0,0,0,1,0,0,0,1,0,0,1,0]).view(1,-1)
# rule3 = torch.Tensor([0,0,1,0,0,1,0,0,0,1,0,0]).view(1,-1)
# rules = Variable(torch.cat((rule1,rule2,rule3),0), requires_grad=True)

optimizer = torch.optim.Adam([rules, embeddings], lr=learning_rate)

criterion = torch.nn.BCELoss(size_average=False)

for i in range(num_iters):
    optimizer.zero_grad()

    valuation = valuation_init
    for step in range(steps):
        valuation = decoder_efficient(valuation, step)
        print('step', step, 'valuation3', valuation[3], 'valuation2', valuation[2])
    loss = criterion(valuation[3][0, :], Variable(torch.Tensor(target[0, :])))

    print(i, 'lossssssssssssssssssssssssssss', loss.data[0])
    loss.backward()
    optimizer.step()


