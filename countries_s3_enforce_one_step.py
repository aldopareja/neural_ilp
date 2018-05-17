import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
# from pudb import set_trace;
import time
from sklearn.metrics import precision_recall_curve as prc
from sklearn.metrics import auc

with open('data/countries_s3') as f:
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

# reading the test set
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

#reading the complete database (to sample false facts)
with open('data/countries_complete') as f:
    comp = f.read().splitlines()
comp = [el.split(',') for el in comp]
preds = [fact[0] for fact in comp]
subjs = [fact[1] for fact in comp]
objs = [fact[2] for fact in comp]

comp = np.array([(predsToIdx[preds[i]],consToIdx[subjs[i]], consToIdx[objs[i]]) for i in range(len(comp))])
ts_data = np.zeros((num_predicates,num_constants,num_constants))
ts_data[comp[:,0],comp[:,1],comp[:,2]] = 1
ts_data = torch.from_numpy(ts_data)
comp = (ts_data==1).nonzero()
comp = torch.cat((predicates[comp[:,0]],
                   constants[comp[:,1]],
                   constants[comp[:,2]]),dim=1)

#takes a fact in embedding form and prints its word equivalent
def print_fact(fact):
    p = int(fact[:num_predicates].nonzero().squeeze().numpy())
    p = idxToPreds[p]
    c1 = int(fact[num_predicates:num_predicates+num_constants].nonzero().squeeze().numpy())
    c1 = idxToCons[c1]
    c2 = int(fact[num_predicates+num_constants:num_predicates+2*num_constants].nonzero().squeeze().numpy())
    c2 = idxToCons[c2]
    print(p,c1,c2)

#gets a set of false facts for each test fact (s[i,j] by sampling c1, c2, c3 and c4 such that s[c1,j], s[i,c2] and s[c3,c4]
# do not belong to the complete dataset
def get_false_set(test,comp):
    test_tmp = test.numpy()
    comp_tmp = comp.numpy()
    false_set = torch.empty(0,num_feats_per_fact)
    for el in test_tmp:
        while True:
            c1 = constants[np.random.randint(0,num_constants),:]
            f1 = np.concatenate((el[:num_predicates],c1,el[num_predicates+num_constants:]))
            if not any((comp_tmp[:]==f1).all(1)):
                f1 = torch.from_numpy(f1).unsqueeze(0)
                break
        while True:
            c1 = constants[np.random.randint(0,num_constants),:]
            f2 = np.concatenate((el[:num_predicates+num_constants],c1))
            if not any((comp_tmp[:]==f2).all(1)):
                f2 = torch.from_numpy(f2).unsqueeze(0)
                break
        while True:
            c1 = constants[np.random.randint(0,num_constants),:]
            c2 = constants[np.random.randint(0,num_constants),:]
            f3 = np.concatenate((el[:num_predicates],c1,c2))
            if not any((comp_tmp[:]==f3).all(1)):
                f3 = torch.from_numpy(f3).unsqueeze(0)
                break
        false_set = torch.cat((false_set,f1,f2,f3),dim=0)
    return false_set


# sample num_samples as a connected subgraph of the input data
#   basically, perform num_samples steps, each adding one more fact
#   that is connected to at least one constant in the sample
# data: a tensor of the form num_preds,num_cons,num_cons where a 1
#   means that the fact composed of pred,cons1,cons2 is true
# num_samples: number of samples to be gotten
# returns sample: a tensor of the form num_samples*num_feats_per_fact
def sample_neighbors(num_samples,data):
    data_source_tmp = data.clone()
    data_tmp = torch.zeros_like(data)
    sample = torch.zeros(0,num_feats_per_fact,dtype=torch.long)
    #choose one random constant
    idx = torch.randperm(num_constants)[0].unsqueeze(0)
    for _ in range(num_samples):
    #     print('data_source',data_source_tmp)
        #subset your possible choices to where idx is subject or object
        data_tmp[:,idx,:] = data_source_tmp[:,idx,:]
        data_tmp[:,:,idx] = data_source_tmp[:,:,idx]
        #choose one at random
        new_fact = data_tmp.nonzero()
        if new_fact.size()[0] == 0:
            break
        chosen = torch.randperm(new_fact.size()[0])[0]
        new_fact = new_fact[chosen,:].unsqueeze(0)
        #add fact to sample
        sample = torch.cat((sample,new_fact),dim=0)
        #set chosen fact to zero (avoiding choosing it again)
        data_source_tmp[new_fact[:,0],new_fact[:,1],new_fact[:,2]] = 0
        #add new idx in the fact
        idx = torch.cat((idx,new_fact[:,1],new_fact[:,2]))
        idx = torch.unique(idx)
    sample = torch.cat((predicates[sample[:,0]],
                        constants[sample[:,1]],
                        constants[sample[:,2]]),dim=1)
    return sample



#helps removing repeated predicted facts -> same embeddings and constants, probably different scores
#this is of complexity K^3, could be optimized
def leaveTopK(preds,K):
    _,idx = torch.sort(preds[:,-4],descending=True)
    preds = preds[idx,:]
    out = preds[0,:].unsqueeze(0)
    for i in range(1,min(K,preds.size()[0])):
        t = preds[i,:].unsqueeze(0)
        if t[:,-4] == 0:
            break
        m,_ = torch.max(F.cosine_similarity(t[:,:-4].repeat(out.size()[0],1),out[:,:-4],dim=1),dim=0)
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
    facts = torch.cat((facts,torch.range(0,facts.size()[0]-1).unsqueeze(1)),dim=1)
    facts_tmp = facts.clone()
    facts = facts.repeat((1,num_predicates)).view(-1,num_feats_per_fact+2)
    #rule 1
    # b1(x,y)<-b2(y,x)
    preds_expanded = predicates.repeat((num_facts,1))
    start_time = time.time()
    rule_expanded = rules[0].repeat(facts.size()[0],1)
#     print(rule_expanded)
#     print(facts)
#     print(preds_expanded)
    #body unification
    preds_r1 = F.cosine_similarity(rule_expanded[:,:num_predicates],facts[:,:num_predicates],dim=1)
#     print('body', preds_r1)
    #previous score
    preds_r1 = preds_r1*facts[:,-2]
#     print('previous', preds_r1)
    #head unification (for each predicate)
    preds_r1 = preds_r1 * F.cosine_similarity(preds_expanded,rule_expanded[:,:num_predicates])
#     print('head', preds_r1)
#     print(preds_r1)
    preds_r1 = preds_r1.unsqueeze(1)
    preds_r1 = torch.cat((preds_expanded,
                         facts[:,num_predicates+num_constants:-2],
                         facts[:,num_predicates:num_predicates+num_constants],
                         preds_r1,
                         facts[:,-1].unsqueeze(1),
                         -torch.ones(facts.size()[0],1),
                         -torch.ones(facts.size()[0],1)),dim=1)
    # print(preds_r1)
    preds_r1 = leaveTopK(preds_r1,K)


    
    # #rule 2
    # #b1(x,y)<-b2(x,z),b3(z,y)
    # #plus 2 because the fact_id is now the last dimmension
    # body1 = facts.repeat((1,num_facts)).view(-1,num_feats_per_fact+2)
    # body2 = facts.repeat((num_facts,1))
    # rule_expanded = rules[1].repeat(body1.size()[0],1)
    # preds_expanded = predicates.repeat(num_facts**2,1)
    # #previous scores
    # preds_r2 = body1[:,-2]*body2[:,-2]
    # #similarity between shared constants
    # preds_r2 = preds_r2*F.cosine_similarity(body1[:,num_predicates+num_constants:-2],
    #                                         body2[:,num_predicates:num_predicates+num_constants],dim=1)
    # #remove 0 scores to improve speed - if constant isn't shared we don't compute anything else

    # non_zero = preds_r2.nonzero().squeeze()
    # if non_zero.size()[0]==0:
    #     preds_r2 = torch.zeros_like(preds_r2)
    # else:
    #     #predicate of body1 with predicate of rule\
    #     preds_r2[non_zero] = preds_r2[non_zero]*F.cosine_similarity(rule_expanded[non_zero,num_predicates:2*num_predicates],
    #                                                                 body1[non_zero,:num_predicates],dim=1)

    # if non_zero.size()[0]==0:
    #     preds_r2 = torch.zeros_like(preds_r2)
    # else:
    #     #predicate of body2 with predicate of rule
    #     preds_r2[non_zero] = preds_r2[non_zero]*F.cosine_similarity(rule_expanded[non_zero,num_predicates:2*num_predicates],
    #                                                                 body2[non_zero,:num_predicates],dim=1)
    # if non_zero.size()[0]==0:
    #     preds_r2 = torch.zeros_like(preds_r2)
    # else:
    #     #head of rule with the two predicates
    #     preds_r2[non_zero] = preds_r2[non_zero]*F.cosine_similarity(rule_expanded[non_zero,:num_predicates],
    #                                                                 preds_expanded[non_zero,:],dim=1)
    
    # preds_r2 = preds_r2.unsqueeze(1)
    # preds_r2 = torch.cat((preds_expanded
    #                      ,body1[:,num_predicates:num_predicates+num_constants]
    #                      ,body2[:,num_predicates+num_constants:-2]
    #                      ,preds_r2
    #                      ,body1[:,-1].unsqueeze(1)
    #                      ,body2[:,-1].unsqueeze(1))
    #                     ,dim=1)
    # #removing repeated facts and leaving ones with highest score
    # preds_r2 = leaveTopK(preds_r2,K)

    #rule 3
    #b1(x,y)<-b2(x,z),b3(z,w),b4(w,y)
    #plus 2 because the fact_id is now the last dimmension
    body1 = facts.repeat((1,num_facts)).view(-1,num_feats_per_fact+2)
    body2 = facts.repeat((num_facts,1))
    rule_expanded = rules[1].repeat(body1.size()[0],1)
    preds_expanded = predicates.repeat(num_facts**2,1)
    #previous scores
    preds_r3 = body1[:,-2]*body2[:,-2]
    #similarity between shared constants
    preds_r3 = preds_r3*F.cosine_similarity(body1[:,num_predicates+num_constants:-2],
                                            body2[:,num_predicates:num_predicates+num_constants],dim=1)
    #remove 0 scores to improve speed - if constant isn't shared we don't compute anything else

    non_zero = preds_r3.nonzero().squeeze()
    if non_zero.size()[0]==0:
        preds_r3 = torch.zeros_like(preds_r3)
    else:
        #predicate of body1 with predicate of rule\
        preds_r3[non_zero] = preds_r3[non_zero]*F.cosine_similarity(rule_expanded[non_zero,num_predicates:2*num_predicates],
                                                                    body1[non_zero,:num_predicates],dim=1)
        #predicate of body2 with predicate of rule
        preds_r3[non_zero] = preds_r3[non_zero]*F.cosine_similarity(rule_expanded[non_zero,2*num_predicates:3*num_predicates],
                                                                    body2[non_zero,:num_predicates],dim=1)
        #head of rule with the two predicates
        preds_r3[non_zero] = preds_r3[non_zero]*F.cosine_similarity(rule_expanded[non_zero,:num_predicates],
                                                                    preds_expanded[non_zero,:],dim=1)
    
    preds_r3 = preds_r3.unsqueeze(1)
    preds_r3 = torch.cat((preds_expanded
                         ,body1[:,num_predicates:num_predicates+num_constants]
                         ,body2[:,num_predicates+num_constants:-2]
                         ,preds_r3
                         ,body1[:,-1].unsqueeze(1)
                         ,body2[:,-1].unsqueeze(1),
                         -torch.ones(body1.size()[0],1))
                        ,dim=1)
    
    #removing repeated facts and leaving ones with highest score
    preds_r3 = leaveTopK(preds_r3,K)

#     #taking care of third atom
    no_preds_left = preds_r3.size()[0]
    body3 = facts_tmp.repeat((1,no_preds_left)).view(-1,num_feats_per_fact+2)

    preds_r4 = preds_r3.repeat((num_facts,1))

    rule_expanded = rules[1].repeat((preds_r4.size()[0],1))
    
    #unifying second shared constant
    p = preds_r4[:,-4] * F.cosine_similarity(preds_r4[:,num_predicates+num_constants:num_predicates+2*num_constants],
                                            body3[:,num_predicates:num_predicates+num_constants],dim=1)
    p = p * F.cosine_similarity(preds_r4[:,num_predicates+num_constants:num_predicates+2*num_constants],
                                            body3[:,num_predicates:num_predicates+num_constants],dim=1)
    
    #unifying third body predicate
    p = p * F.cosine_similarity(rule_expanded[:,3*num_predicates:],
                                            body3[:,:num_predicates],dim=1)
#     for i in (preds_r4[:,:num_predicates+num_constants]
#                          ,body3[:,num_predicates+num_constants:num_predicates+2*num_constants]
#                          ,p
#                          ,preds_r4[:,[-3,-2]]
#                          ,body3[:,-1].unsqueeze(1)):
#               print(i.size())
    preds_r5 = torch.cat((preds_r4[:,:num_predicates+num_constants]
                         ,body3[:,num_predicates+num_constants:num_predicates+2*num_constants]
                         ,p.unsqueeze(1)
                         ,preds_r4[:,[-3,-2]]
                         ,body3[:,-1].unsqueeze(1))
                        ,dim=1)
    
    #removing repeated facts and leaving ones with highest score
    preds_r5 = leaveTopK(preds_r5,K)
    #out = torch.cat((preds_r1,preds_r2,preds_r3),dim=0)
    out = torch.cat((preds_r1,preds_r5),dim=0)
#     print("fws took %s" % (time.time() - start_time))
    return out

#Find maximum similarity for each consequence in the set of facts contained in target
#if testing is true it finds the consequence with maximum similarity for each target
#otherwise finds the target with maximum similarity for each consequence
#Inputs: 
#consequences: result of unrolling the rules for the specified steps with the input facts
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
        tmp_c = consequences.repeat(1,num_targets).view(-1,num_feats_per_fact+4)
        tmp_t = target.repeat(num_consequences,1)

    #first constant
    sim = F.cosine_similarity(tmp_c[:,num_predicates:num_predicates+num_constants],
                              tmp_t[:,num_predicates:num_predicates+num_constants],dim=1)
    #only compute for non-zero values to speed up
#     non_zero = sim.nonzero()
#     if non_zero.size()[0]==0:
#         sim = torch.zeros_like(sim)
#     else:
#         non_zero = non_zero.squeeze()
#         sim[non_zero] = sim[non_zero] * F.cosine_similarity(tmp_c[non_zero,num_predicates+num_constants:-3]
#                                                            ,tmp_t[non_zero,num_predicates+num_constants:],dim=1)

#     non_zero = sim.nonzero()
#     if non_zero.size()[0]==0:
#         sim = torch.zeros_like(sim)
#     else:
#         non_zero = non_zero.squeeze()
#         sim[non_zero] = sim[non_zero] * F.cosine_similarity(tmp_c[non_zero,:num_predicates] 
#                                                            ,tmp_t[non_zero,:num_predicates],dim=1)
#         sim[non_zero] = sim[non_zero] + tmp_c[non_zero,-3]*lamb2
    sim = sim * F.cosine_similarity(tmp_c[:,num_predicates+num_constants:num_predicates+2*num_constants]
                                    ,tmp_t[:,num_predicates+num_constants:],dim=1)
    sim = sim * F.cosine_similarity(tmp_c[:,:num_predicates] 
                                    ,tmp_t[:,:num_predicates],dim=1)
    sim = sim + tmp_c[:,-3]*lamb2
    #for each consequence/target, get the maximum simlarity with the set of targets/consequences
    if testing:
        sim = sim.view(-1,num_consequences)
    else:
        sim = sim.view(-1,num_targets)
    m, idx = torch.max(sim,dim=1)
#     print("fms took %s" % (time.time() - start_time))
    if testing:
        return m, consequences[idx,:]
    return m,target[idx,:]



####TRAINING
#dbg -> cherrypicked
# core_rel = Variable(knowledge_pos[[0,1,3]])
# target = Variable(knowledge_pos[2]).unsqueeze(0)
# target = Variable(knowledge_pos[[2,4],:])
#####sampling
target = Variable(knowledge_pos)
no_samples = 40

num_iters = 50
learning_rate = .1
lamb = 1
lamb2 = 0

steps = 1
num_rules = 3
epsilon=.001

K = 400 ##For top K

#hyperparameter search
# lambdas = [1,2,5,0.3,0.8]
with open('s3_auc-pr','w') as f:
    # for lamb in lambdas:
    suc_rate_neigh = 0
    suc_rate_locin = 0
    accuracies = []
    for _ in range(10):
        K_tmp = K
        #rules should be:
        #r1(x,y) <- r2(y,x)
        #r1(x,y) <- r2(x,z),r3(z,x)
        rules = [Variable(torch.rand(1*num_predicates), requires_grad=True),
                 # Variable(torch.rand(2*num_predicates), requires_grad=True),
                 Variable(torch.rand(4*num_predicates), requires_grad=True)]
        # rules = [Variable(torch.rand(num_predicates), requires_grad=True),
        #          Variable(torch.Tensor([1, 1]), requires_grad=True)]
        f.write('random_rules' + str(rules) + '\n')
        optimizer = torch.optim.Adam([
                {'params': rules}], 
                lr = learning_rate)

        criterion = torch.nn.MSELoss(size_average=False)

        rules_tmp = [torch.zeros_like(rule) for rule in rules]
        for epoch in range(num_iters):
            for par in optimizer.param_groups:
#                 par['params'][2].data.clamp_(min=0.3,max=0.7)
                par['params'][1].data.clamp_(min=0.3,max=0.7)
                par['params'][0].data.clamp_(min=0.3,max=0.7)

            # core_rel = Variable(knowledge_pos[core_rel])
            core_rel = sample_neighbors(no_samples,data)
            # target = Variable(knowledge_pos)
            optimizer.zero_grad()
            facts = torch.cat((core_rel, Variable(torch.ones(core_rel.size()[0], 1))), 1)
            #will accumulate predictions separately to compare with target facts
            consequences = forward_step(facts,K_tmp)
            for step in range(1,steps):
                tmp = torch.cat((facts,consequences[:,:-2]),dim=0)
                tmp = forward_step(tmp,K_tmp)
                consequences = torch.cat((consequences,tmp),dim=0)
            #LOSS
            loss = 0
            m, matches = find_max_similarities(consequences,core_rel,testing=True)
            loss = torch.sum(m*(1 - matches[:,-4]))
#             print(epoch, 'losssssssssssssssssssss',loss.data[0])
            # print(sum([torch.sum(rules_tmp[i]-rules[i]) for i in range(num_rules)]))
#             if loss < 10**-6 or sum([torch.sum(torch.abs(rules_tmp[i]-rules[i])) for i in range(num_rules)])<10**-5:
#                 break
            rules_tmp = [r.clone() for r in rules]
            loss.backward()
            optimizer.step()
        print(rules)
        ###### printing and saving AUC
        K_tmp = 500
        facts = torch.cat((knowledge_pos, Variable(torch.ones(knowledge_pos.size()[0], 1))), 1)
        consequences = forward_step(facts,K_tmp)
        for step in range(1,steps):
            tmp = torch.cat((facts,consequences[:,:-3]),dim=0)
            tmp = forward_step(tmp,K_tmp)
            consequences = torch.cat((consequences,tmp),dim=0)
        false_set = get_false_set(test,comp)
        test_plus_false = torch.cat((test,false_set))
        m,matches = find_max_similarities(consequences,test_plus_false,testing=True)
        p = matches[:,-4]
        true_vals = np.ones(test.size()[0])
        true_vals = np.concatenate((true_vals,np.zeros(false_set.size()[0])))
        prec,rec,_ = prc(true_vals,(m*p).detach().numpy()) #assume that no unification is a score of 0
        auc_tmp = auc(rec,prec)
        print('auc',auc_tmp)
        f.write('rules' + str(rules) + '\n')
        f.write('auc' + str(auc_tmp) + '\n')
#######Writting results
    #     suc_neigh, suc_locIn = False,False
    #     if F.cosine_similarity(rules[0],torch.Tensor([0,1,0,1]),dim=0)>0.5:
    #         suc_neigh = True
    #     if F.cosine_similarity(rules[1],torch.Tensor([1,0,1,0,1,0]),dim=0)>0.5:
    #         suc_locIn = True
    #     if suc_neigh:
    #         suc_rate_neigh+=1
    #     if suc_locIn:
    #         suc_rate_locin+=1
    #     f.write('lamb '+str(lamb)+'\n')
    #     f.write('train_loss '+str(loss)+'\n')
    #     f.write('rules '+str(rules)+'\n')
    #     f.write('suc_neigh '+str(suc_neigh)+'\n')
    #     f.write('suc_locIn '+str(suc_locIn)+'\n')
    #     #computing test results
    #     print('computing test results')
    #     K_tmp = 250
    #     facts = torch.cat((knowledge_pos, Variable(torch.ones(knowledge_pos.size()[0], 1))), 1)
    #     consequences = forward_step(facts,K_tmp)
    #     for step in range(1,steps):
    #         tmp = torch.cat((facts,consequences[:,:-2]),dim=0)
    #         tmp = forward_step(tmp,K_tmp)
    #         consequences = torch.cat((consequences,tmp),dim=0)
    #     m,matches = find_max_similarities(consequences,test,testing=True)
    #     p = matches[:,-3]
    #     true_positives = m[p>0.5]
    #     true_positives = (true_positives>0.5).nonzero()
    #     true_positives = true_positives.size()[0]
    #     ts_accuracy = true_positives/test.size()[0]
    #     accuracies.append(ts_accuracy)
    #     f.write('ts_accuracy '+str(ts_accuracy)+'\n')
    #     f.flush()
    # f.write('#############RESULTS###############'+'\n')
    # f.write('lamb '+str(lamb)+'\n')
    # f.write('suc_rate_neigh '+str(suc_rate_neigh)+'\n')
    # f.write('suc_rate_locin '+str(suc_rate_locin)+'\n')
    # accuracies = np.array(accuracies)
    # f.write('mean accuracy ' + str(np.mean(accuracies)) +'\n')
    # f.write('std accuracy ' + str(np.std(accuracies))+'\n')
    # f.write('####################################'+'\n')
    # f.flush()