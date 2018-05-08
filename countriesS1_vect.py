import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
from pudb import set_trace;

with open('data/countries_s1') as f:
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
print(idxToPreds)

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
print(knowledge_pos)

#helps removing repeated predicted facts -> same embeddings and constants, probably different scores
#this is of complexity K^3, could be optimized
def leaveTopK(preds,K):
    _,idx = torch.sort(preds[:,-1],descending=True)
    preds = preds[idx,:]
    out = preds[0,:].unsqueeze(0)
    for i in range(1,K):
        t = preds[i,:].unsqueeze(0)
        m,_ = torch.max(F.cosine_similarity(t[:,:-1].repeat(out.size()[0],1),out[:,:-1],dim=1),dim=0)
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
    # b1(x,y)<-b2(y,x)
    rule_expanded = rules[0].repeat(num_facts,1)
    preds_r1 = F.cosine_similarity(rule_expanded[:,num_predicates:],facts[:,:num_predicates],dim=1)
    preds_r1 = preds_r1*facts[:,-1]
    preds_r1 = preds_r1.unsqueeze(1)
    preds_r1 = torch.cat((rule_expanded[:,:num_predicates],
                         facts[:,num_predicates+num_constants:-1],
                         facts[:,num_predicates:num_predicates+num_constants],
                         preds_r1),dim=1)
    # print(preds_r1)

    preds_r1 = leaveTopK(preds_r1,K)
    #rule 2
    #b1(x,y)<-b2(x,z),b3(z,y)
    body1 = facts.repeat((1,num_facts)).view(-1,num_feats_per_fact+1)
    body2 = facts.repeat((num_facts,1))
    rule_expanded = rules[1].repeat(body1.size()[0],1)
    #previous scores
    preds_r2 = body1[:,-1]*body2[:,-1]
    #predicate of body1 with predicate of rule
    preds_r2 = preds_r2*F.cosine_similarity(rule_expanded[:,num_predicates:2*num_predicates],body1[:,:num_predicates],dim=1)
    #predicate of body2 with predicate of rule
    preds_r2 = preds_r2*F.cosine_similarity(rule_expanded[:,2*num_predicates:],body2[:,:num_predicates],dim=1)
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
    out = torch.cat((preds_r1,preds_r2),dim=0)
    return out
    # return preds_r2

    
####TRAINING
#dbg -> cherrypicked
# core_rel = Variable(knowledge_pos[[0,1,3]])
# target = Variable(knowledge_pos[2]).unsqueeze(0)
# target = Variable(knowledge_pos[[2,4],:])
#####sampling
target = Variable(knowledge_pos)
no_samples = 50

num_iters = 200
learning_rate = .1
lamb = 1

steps = 1
num_rules = 2
epsilon=.001

K = 30 ##For top K



#hyperparameter search
# lambdas = [1,2,5,0.3,0.8]
with open('mult_noEnforcement','w') as f:
    # for lamb in lambdas:
    suc_rate_neigh = 0
    suc_rate_locin = 0
    for _ in range(10):
        #rules should be:
        #r1(x,y) <- r2(y,x)
        #r1(x,y) <- r2(x,z),r3(z,x)
        rules = [Variable(torch.rand(2*num_predicates), requires_grad=True),
                 Variable(torch.rand(3*num_predicates), requires_grad=True)]
        # rules = [Variable(torch.rand(num_predicates), requires_grad=True),
        #          Variable(torch.Tensor([1, 1]), requires_grad=True)]
        optimizer = torch.optim.Adam([
                {'params': rules}], 
                lr = learning_rate)

        criterion = torch.nn.MSELoss(size_average=False)

        rules_tmp = [torch.zeros_like(rule) for rule in rules]
        for epoch in range(num_iters):
            for par in optimizer.param_groups:
                par['params'][1].data.clamp_(min=0.,max=1.)
                par['params'][0].data.clamp_(min=0.,max=1.)
            # # ##sampling
            core_rel = torch.randperm(no_facts)
            # # target = core_rel[no_samples:]
            core_rel = core_rel[:no_samples]

            core_rel = Variable(knowledge_pos[core_rel])
            # target = Variable(knowledge_pos)
            optimizer.zero_grad()
            facts = torch.cat((core_rel, Variable(torch.ones(core_rel.size()[0], 1))), 1)
            #will accumulate predictions separately to compare with target facts
            consequences = forward_step(facts)
            for step in range(1,steps):
                tmp = torch.cat((consequences,facts),dim=0)
                tmp = forward_step(tmp)
                consequences = torch.cat((consequences,tmp),dim=0)
            #LOSS
            loss = 0
            num_consequences = consequences.size()[0]
            num_targets = target.size()[0]
            # print(num_targets,num_consequences)
            #each consequence repeated by the number of targets
            tmp_c = consequences.repeat(1,target.size()[0]).view(-1,num_feats_per_fact+1)
            #all targets repeated number of consequences
            tmp_t = target.repeat(num_consequences,1)
            # print(tmp_c.size())
            # print(tmp_t.size())
            #for each consequence compute the similarity with all targets
            sim = F.cosine_similarity(tmp_c[:,:num_predicates],tmp_t[:,:num_predicates],dim=1)
            sim = sim * F.cosine_similarity(tmp_c[:,num_predicates:num_predicates+num_constants],
                                            tmp_t[:,num_predicates:num_predicates+num_constants],dim=1)
            sim = sim * F.cosine_similarity(tmp_c[:,num_predicates+num_constants:-1],
                                            tmp_t[:,num_predicates+num_constants:],dim=1)
            # sim = F.cosine_similarity(consequences[:,:-1],target)
            #for each consequence, get the maximum simlarity with the set of targets
            sim = sim.view(-1,num_targets)
            # print(sim.size())
            m, _ = torch.max(sim,dim=1)
            # print(m)
            # print(consequences[:,-1])
            #the loss is min(lamb*p,1-p*m)
            loss = torch.sum(lamb*consequences[:,-1]*(1- consequences[:,-1]*m))
            print(rules)
            print(epoch, 'losssssssssssssssssssss',loss.data[0])
            # print(sum([torch.sum(rules_tmp[i]-rules[i]) for i in range(num_rules)]))
            if loss < 10**-6 or sum([torch.sum(torch.abs(rules_tmp[i]-rules[i])) for i in range(num_rules)])<10**-5:
                break
            rules_tmp = [r.clone() for r in rules]
            loss.backward()
            optimizer.step()
        suc_neigh, suc_locIn = False,False
        if F.cosine_similarity(rules[0],torch.Tensor([0,1,0,1]),dim=0)>0.5:
            suc_neigh = True
        if F.cosine_similarity(rules[1],torch.Tensor([1,0,1,0,1,0]),dim=0)>0.5:
            suc_locIn = True
        if suc_neigh:
            suc_rate_neigh+=1
        if suc_locIn:
            suc_rate_locin+=1
        f.write('lamb '+str(lamb)+'\n')
        f.write('loss '+str(loss)+'\n')
        f.write('rules '+str(rules)+'\n')
        f.write('suc_neigh '+str(suc_neigh)+'\n')
        f.write('suc_locIn '+str(suc_locIn)+'\n')
        f.flush()
    f.write('#############RESULTS###############'+'\n')
    f.write('lamb '+str(lamb)+'\n')
    f.write('suc_rate_neigh '+str(suc_rate_neigh)+'\n')
    f.flush()
    f.write('suc_rate_locin '+str(suc_rate_locin)+'\n')
    f.write('####################################'+'\n')