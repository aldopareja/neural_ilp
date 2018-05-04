import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
from pudb import set_trace;

with open('data/countries_s1_dbg_locin') as f:
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
num_facts = knowledge_pos.size()[0]
num_feats_per_fact = knowledge_pos.size()[1]

#helps removing repeated predicted facts -> same embeddings and constants, probably different scores
#this is of complexity K^3, could be optimized
def leaveTopK(preds,K):
    _,idx = torch.sort(preds[:,-1],descending=True)
    preds = preds[idx,:]
    out = preds[0,:].unsqueeze(0)
    for i in range(1,K):
        t = preds[i,:].unsqueeze(0)
        m,_ = torch.max(F.cosine_similarity(t[:,:-1].repeat(out.size()[0],1),out[:,:-1]),dim=0)
        if m<1:
            out = torch.cat((out,t),dim=0)
    return out

####FORWARD CHAINING
#input: some facts that can either be ground or predicted in previous steps
#output: predicted facts in this specific step (outer loop must gather all of them)
#computes the predictions of applying each rule to the facts, giving a score to each of them.
#leaves only topK scoring fact for each applied rule (not for the whole thing)
def forward_step(facts):
    num_facts = facts.size()[0]
    #rule 1
    # b1(x,y)<-b1(y,x)
    # rule_expanded = rules[0].expand(facts[:,:num_predicates].size())
    # preds_r1 = F.cosine_similarity(rule_expanded,facts[:,:num_predicates],dim=1)
    # preds_r1 = preds_r1*facts[:,-1]
    # preds_r1 = preds_r1.unsqueeze(1)
    # preds_r1 = torch.cat((rule_expanded,
    #                      facts[:,num_predicates+num_constants:-1],
    #                      facts[:,num_predicates:num_predicates+num_constants],
    #                      preds_r1),dim=1)
    # preds_r1 = leaveTopK(preds_r1,K)
    #rule 2
    # b1(x,y)<-b2(x,z),b2(z,y)
    body1 = facts.repeat((1,num_facts)).view(-1,num_feats_per_fact+1)
    body2 = facts.repeat((num_facts,1))
    rule_expanded = rules[1].repeat(body1.size()[0],1)
    #previous scores
    preds_r2 = body1[:,-1]*body2[:,-1]
    #predicate of body1 with predicate of rule
    preds_r2 = preds_r2*F.cosine_similarity(rule_expanded[:,num_predicates:],body1[:,:num_predicates],dim=1)
    #predicate of body2 with predicate of rule
    preds_r2 = preds_r2*F.cosine_similarity(rule_expanded[:,num_predicates:],body2[:,:num_predicates],dim=1)
    #similarity between shared constants
    preds_r2 = preds_r2*F.cosine_similarity(body1[:,num_predicates+num_constants:-1],
                                            body2[:,num_predicates:num_predicates+num_constants],dim=1)
    preds_r2 = preds_r2.unsqueeze(1)
    preds_r2 = torch.cat((rule_expanded[:,:num_predicates]
                         ,body1[:,num_predicates:num_predicates+num_constants]
                         ,body2[:,num_predicates+num_constants:-1]
                         ,preds_r2)
                        ,dim=1)
    #removing repeated facts and leaving ones with highest score
    preds_r2 = leaveTopK(preds_r2,K)
    # out = torch.cat((preds_r1,preds_r2),dim=0)
    # return out
    return preds_r2

    
####TRAINING
#dbg -> cherrypicked
core_rel = Variable(knowledge_pos[[0,1]])
target = Variable(knowledge_pos[2]).unsqueeze(0)

#####sampling
# target = Variable(knowledge_pos)
# no_samples = 100

num_iters = 200
learning_rate = .1
drop=0

steps = 1
num_rules = 2
epsilon=.001

K = 4 ##For top K

#rules should be:
#r1(x,y) <- r1(y,x)
#r1(x,y) <- r2(x,z),r2(z,x)
# rules = [Variable(torch.rand(num_predicates), requires_grad=True),
#          Variable(torch.rand(2*num_predicates), requires_grad=True)]
rules = [Variable(torch.rand(num_predicates), requires_grad=True),
         Variable(torch.Tensor([1, 1]), requires_grad=True)]


optimizer = torch.optim.Adam([
        {'params': rules}], 
        lr = learning_rate)

criterion = torch.nn.MSELoss(size_average=False)

for epoch in range(num_iters):
    # # ##sampling
    # core_rel = torch.randperm(no_facts)
    # # target = core_rel[no_samples:]
    # core_rel = core_rel[:no_samples]

    # core_rel = Variable(knowledge_pos[core_rel])
    # target = Variable(knowledge_pos)
    optimizer.zero_grad()
    facts = torch.cat((core_rel, Variable(torch.ones(core_rel.size()[0], 1))), 1)
    #will accumulate predictions separately to compare with target facts
    consequences = forward_step(facts)
    for step in range(1,steps):
        tmp = torch.cat((consequences,facts),dim=0)
        tmp = forward_step(tmp)
        consequences = torch.cat((consequences,tmp),dim=0)
    loss = 0
    for cons in consequences:
        m, indi = torch.max(F.cosine_similarity(cons[:-1].view(1,-1).expand(target.size()),target),0)
        indi=indi.data[0]
        loss += criterion(cons[:num_predicates],target[indi,:num_predicates]) + (1-cons[-1])*m
        #remove fact from predicted facts
        # if indi==0:
        #     facts = facts[1:,:]
        # elif indi+1 == facts.size()[0]:
        #     facts = facts[:indi,:]
        # else:
        #     facts = torch.cat((facts[:indi,:],facts[indi+1:,:]),dim=0)

    print(epoch, 'losssssssssssssssssssss',loss.data[0])
    print(rules)
    loss.backward()
    optimizer.step()


    



