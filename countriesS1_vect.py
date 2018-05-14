import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pudb import set_trace;
import time

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

num_feats_per_fact = knowledge_pos.size()[1]

#reading the test set
with open('data/s1_test') as f:
    test = f.read().splitlines()
test = [el.split(',') for el in test]
preds = [fact[0] for fact in test]
subjs = [fact[1] for fact in test]
objs = [fact[2] for fact in test]

test = np.array([(predsToIdx[preds[i]],consToIdx[subjs[i]], consToIdx[objs[i]]) for i in range(len(test))])
ts_data = np.zeros((num_predicates,num_constants,num_constants))
ts_data[test[:,0],test[:,1],test[:,2]] = 1
ts_data = torch.from_numpy(ts_data)
test = (ts_data==1).nonzero()
test = torch.cat((predicates[test[:,0]],
                   constants[test[:,1]],
                   constants[test[:,2]]),dim=1)


#helps removing repeated predicted facts -> same embeddings and constants, probably different scores
#this is of complexity K^3, could be optimized
def leaveTopK(preds,K):
    _,idx = torch.sort(preds[:,-1],descending=True)
    preds = preds[idx,:]
    out = preds[0,:].unsqueeze(0)
    for i in range(1,min(K,preds.size()[0])):
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
def forward_step(facts,K):
    num_facts = facts.size()[0]
    #rule 1
    # b1(x,y)<-b2(y,x)
    start_time = time.time()
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
    #similarity between shared constants
    preds_r2 = preds_r2*F.cosine_similarity(body1[:,num_predicates+num_constants:-1],
                                            body2[:,num_predicates:num_predicates+num_constants],dim=1)
    #remove 0 scores to improve speed - if constant isn't shared we don't compute anything else
    non_zero = preds_r2.nonzero().squeeze()
    preds_r2 = preds_r2[non_zero]
    rule_expanded = rule_expanded[non_zero,:]
    body1 = body1[non_zero,:]
    body2 = body2[non_zero,:]
    
    #predicate of body1 with predicate of rule
    preds_r2 = preds_r2*F.cosine_similarity(rule_expanded[:,num_predicates:2*num_predicates],body1[:,:num_predicates],dim=1)
    #predicate of body2 with predicate of rule
    preds_r2 = preds_r2*F.cosine_similarity(rule_expanded[:,2*num_predicates:],body2[:,:num_predicates],dim=1)
    
    preds_r2 = preds_r2.unsqueeze(1)
    preds_r2 = torch.cat((rule_expanded[:,:num_predicates]
                         ,body1[:,num_predicates:num_predicates+num_constants]
                         ,body2[:,num_predicates+num_constants:-1]
                         ,preds_r2)
                        ,dim=1)
    #removing repeated facts and leaving ones with highest score
    preds_r2 = leaveTopK(preds_r2,K)
    out = torch.cat((preds_r1,preds_r2),dim=0)
    print("fws took %s" % (time.time() - start_time))
    return out
    # return preds_r2

    
####TRAINING
#dbg -> cherrypicked
# core_rel = Variable(knowledge_pos[[0,1,3]])
# target = Variable(knowledge_pos[2]).unsqueeze(0)
# target = Variable(knowledge_pos[[2,4],:])
#####sampling
target = Variable(knowledge_pos)
no_samples = 200

num_iters = 100
learning_rate = .05
lamb = 1000

steps = 1
num_rules = 2
epsilon=.001

K = 100 ##For top K
#Find maximum similarity for each consequence in the set of facts contained in target
#if testing is true it finds the consequence with maximum similarity for each target
#if testing is true, returns the truth value of the matched predicted consequence for each target
#Inputs: 
#consequences: facts to be looked for maximum similarities across target
#target: set of facts that are assumed to be true
#testing: returns the probabilities of matched facts, a prediction is considered true if p>0.5
def find_max_similarities(consequences,target,testing=False):
    start_time = time.time()    
    num_consequences = consequences.size()[0]
    num_targets = target.size()[0]

    #each consequence repeated by the number of targets
    if testing:
        #for each target find max similarity across consequences
        tmp_c = consequences.repeat(num_targets,1)
        tmp_t = target.repeat(1,num_consequences).view(-1,num_feats_per_fact)
    else:
        #for each consequence compute the similarity with all targets
        tmp_c = consequences.repeat(1,num_targets).view(-1,num_feats_per_fact+1)
        tmp_t = target.repeat(num_consequences,1)

    sim = F.cosine_similarity(tmp_c[:,num_predicates:num_predicates+num_constants],
                              tmp_t[:,num_predicates:num_predicates+num_constants],dim=1)
    #only compute for non-zero values to speed up
    non_zero = sim.nonzero().squeeze()
    #if no unification return 0
    if non_zero.size()[0]==0:
        sim = torch.zeros_like(sim)
    else:
        sim[non_zero] = sim[non_zero] * F.cosine_similarity(tmp_c[non_zero,num_predicates+num_constants:-1],tmp_t[non_zero,num_predicates+num_constants:],dim=1)

    non_zero = sim.nonzero().squeeze()
    if non_zero.size()[0]==0:
        sim = torch.zeros_like(sim)
    else:
        sim[non_zero] = sim[non_zero] * F.cosine_similarity(tmp_c[non_zero,:num_predicates] ,tmp_t[non_zero,:num_predicates],dim=1)

    non_zero = sim.nonzero().squeeze()
    if non_zero.size()[0]==0:
        sim = torch.zeros_like(sim)
    else:
        sim[non_zero] = sim[non_zero] * tmp_c[non_zero,-1]
    #for each consequence/target, get the maximum simlarity with the set of targets/consequences
    if testing:
        sim = sim.view(-1,num_consequences)
    else:
        sim = sim.view(-1,num_targets)
    m, idx = torch.max(sim,dim=1)
    print("fms took %s" % (time.time() - start_time))
    if testing:
        return m, tmp_c[idx,-1]
    return m

#hyperparameter search
# lambdas = [1,2,5,0.3,0.8]
with open('test_acc_s1_neigh_sample_one_step','w') as f:
    # for lamb in lambdas:
    suc_rate_neigh = 0
    suc_rate_locin = 0
    accuracies = []
    for _ in range(10):
        K_tmp = K
        #rules should be:
        #r1(x,y) <- r2(y,x)
        #r1(x,y) <- r2(x,z),r3(z,x)
        rules = [Variable(torch.rand(2*num_predicates), requires_grad=True),
                 Variable(torch.rand(3*num_predicates), requires_grad=True)]
        f.write('initial random rules' + str(rules) +'\n')
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
            consequences = forward_step(facts,K_tmp)
            for step in range(1,steps):
                tmp = torch.cat((consequences,facts),dim=0)
                tmp = forward_step(tmp,K_tmp)
                consequences = torch.cat((consequences,tmp),dim=0)
            #LOSS
            loss = 0
            m,matches = find_max_similarities(consequences,core_rel,testing=True)
            print()
            loss = torch.sum(torch.min(lamb*matches,1 - matches*m))
            print(epoch, 'losssssssssssssssssssss',loss.data[0])
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
        f.write('train_loss '+str(loss)+'\n')
        f.write('rules '+str(rules)+'\n')
        f.write('suc_neigh '+str(suc_neigh)+'\n')
        f.write('suc_locIn '+str(suc_locIn)+'\n')

        #computing test results
        print('computing test results')
        K_tmp = 300
        facts = torch.cat((knowledge_pos, Variable(torch.ones(knowledge_pos.size()[0], 1))), 1)
        consequences = forward_step(facts,K_tmp)
        for step in range(1,steps):
            tmp = torch.cat((consequences,facts),dim=0)
            tmp = forward_step(tmp,K_tmp)
            consequences = torch.cat((consequences,tmp),dim=0)
        m,p = find_max_similarities(consequences,test,testing=True)
        true_positives = m[p>0.5]
        true_positives = (true_positives>0.5).nonzero()
        true_positives = true_positives.size()[0]
        ts_accuracy = true_positives/test.size()[0]
        accuracies.append(ts_accuracy)
        f.write('ts_accuracy '+str(ts_accuracy)+'\n')
        f.flush()
        
    f.write('#############RESULTS###############'+'\n')
    f.write('lamb '+str(lamb)+'\n')
    f.write('suc_rate_neigh '+str(suc_rate_neigh)+'\n')
    f.write('suc_rate_locin '+str(suc_rate_locin)+'\n')
    accuracies = np.array(accuracies)
    f.write('mean accuracy ' + str(np.mean(accuracies)) +'\n')
    f.write('std accuracy ' + str(np.std(accuracies))+'\n')
    f.write('####################################'+'\n')
    f.flush()