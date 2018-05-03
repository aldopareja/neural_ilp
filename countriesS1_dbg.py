import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
from pudb import set_trace;

with open('data/countries_s1_dbg') as f:
    facts = f.read().splitlines()
facts = [el.split(',') for el in facts]
preds = [fact[0] for fact in facts]
subjs = [fact[1] for fact in facts]
objs = [fact[2] for fact in facts]

unique = sorted(list(set(preds)))
num_unique = len(unique)
num_predicates = num_unique
predsToIdx = dict(zip(unique,range(num_unique)))
idxToPreds = dict(zip(range(num_unique),unique))

unique = sorted(list(set(subjs+objs)))
num_unique = len(unique)
num_constants = num_unique
consToIdx = dict(zip(unique,range(num_unique)))
idxToCons = dict(zip(range(num_unique),unique))

facts = np.array([(predsToIdx[preds[i]],consToIdx[subjs[i]], consToIdx[objs[i]]) for i in range(len(facts))])
data = np.zeros((num_predicates,num_constants,num_constants))

#data: idx0->predicate, idx1->subj, idx2->obj
data[facts[:,0],facts[:,1],facts[:,2]] = 1
no_facts = int(np.sum(data))
data = torch.from_numpy(data)

predicates = torch.eye(num_predicates)
constants = torch.eye(num_constants)

knowledge_pos = (data==1).nonzero()
data_aux = knowledge_pos
knowledge_pos = torch.cat((predicates[knowledge_pos[:,0]],
                           constants[knowledge_pos[:,1]],
                           constants[knowledge_pos[:,2]]),dim=1)
print(knowledge_pos)
print(idxToPreds)
print(idxToCons)

####FORWARD CHAINING
def forward_step(facts):
    new_facts = facts.clone()
    i = 0
    for fact1 in facts:
        #rule 0
        p = fact1[-1].expand(1)
        # b1 with the body of fact1
        p = p*F.cosine_similarity(rules[0],fact1[:num_predicates],dim=0)
        new_fact = torch.cat((rules[0],
                              fact1[num_predicates+num_constants:-1],
                              fact1[num_predicates:num_predicates+num_constants],
                              p))
        max_prev, indi_prev =torch.max(F.cosine_similarity(new_fact[:-1].repeat(new_facts.size()[0],1)
                                                           ,new_facts[:,:-1])
                                       ,dim=0)
        set_trace()
        if max_prev.data[0] < 0.9: 
            new_facts = torch.cat((new_facts, new_fact.view(1,-1) ),0)
        elif p.data[0] > new_facts[indi_prev.data[0],-1].data[0]:
            new_facts[indi_prev.data[0]] = new_fact
        for fact2 in facts:
            #rule 1
            rule = rules[1]
            p = fact1[-1]*fact2[-1]
            #body 1
            p = p*F.cosine_similarity(rule[num_predicates:].view(1,-1), fact1[:num_predicates].view(1,-1))
            #body 2
            p = p*F.cosine_similarity(rule[num_predicates:].view(1,-1), fact2[:num_predicates].view(1,-1))
            #enforce same shared constant
            p = p*F.cosine_similarity(fact1[num_predicates+num_constants:-1].view(1,-1) , fact2[num_predicates:num_predicates+num_constants].view(1,-1))
            new_fact = torch.cat((rule[:num_predicates], fact1[num_predicates:num_predicates+num_constants],\
                                                         fact2[num_predicates+num_constants:-1], p), 0)
            max_prev, indi_prev = torch.max(F.cosine_similarity(new_fact[:-1].view(1,-1).expand(new_facts[:,:-1].size()),new_facts[:,:-1]),0)
            if max_prev.data[0] < 0.9: 
                new_facts = torch.cat(( new_facts, new_fact.view(1,-1) ),0)
            elif p.data[0] > new_facts[indi_prev.data[0],-1].data[0]:
                new_facts[indi_prev.data[0]] = new_fact
            # if new_facts.size()[0]>200:
            #     _ , index = torch.topk(new_facts[:,-1], K)
            #     index, _ = torch.sort(index)
            #     new_facts = torch.index_select(new_facts, 0, index)
            # i += 1
            # if i%1000 == 0:
            #     print(datetime.datetime.now())
    print(new_facts.size())
    _ , index = torch.topk(new_facts[:,-1], K)
    index, _ = torch.sort(index)
    new_facts = torch.index_select(new_facts, 0, index)
    return new_facts

####TRAINING
#added params
# core_rel = Variable(knowledge_pos[[0,1,4]])
# target = Variable(knowledge_pos[[2,3]])

core_rel = Variable(knowledge_pos)
target = Variable(knowledge_pos)

num_iters = 200
learning_rate = .01
drop=0

steps = 1
num_rules = 2
epsilon=.001

K = 7 ##For top K

#rules should be:
#r1(x,y) <- r1(y,x)
#r1(x,y) <- r2(x,z),r2(z,x)
rules = [Variable(torch.rand(num_predicates), requires_grad=True),
         Variable(torch.rand(2*num_predicates), requires_grad=True)]

optimizer = torch.optim.Adam([
        {'params': rules}], 
        lr = learning_rate)

criterion = torch.nn.MSELoss(size_average=False)

for epoch in range(num_iters):
    # ##sampling
    # core_rel = torch.randperm(no_facts)
    # target = core_rel[no_samples:]
    # core_rel = core_rel[:no_samples]

    # core_rel = Variable(knowledge_pos[core_rel])
    # target = Variable(knowledge_pos)
    
    
    optimizer.zero_grad()
    facts = torch.cat((core_rel, Variable(torch.ones(core_rel.size()[0], 1))), 1)
    
    for step in range(steps):
        facts = forward_step(facts)
    loss = 0
    for targ in target:
        _, indi = torch.max(F.cosine_similarity(targ.view(1,-1).expand(facts[:,:-1].size()),facts[:,:-1]),0)
        indi=indi.data[0]
        loss += criterion(facts[indi,:-1],targ)+(1-(facts[indi,-1]))
        #remove fact from predicted facts
        # if indi==0:
        #     facts = facts[1:,:]
        # elif indi+1 == facts.size()[0]:
        #     facts = facts[:indi,:]
        # else:
        #     facts = torch.cat((facts[:indi,:],facts[indi+1:,:]),dim=0)

    print(epoch, 'losssssssssssssssssssss',loss.data[0])
    loss.backward()
    optimizer.step()






