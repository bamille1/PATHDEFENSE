# opt_defense.py
# Code implementing the PATHDEFENSE algorithm from "Defense Against Shortest
# Path Attacks" by Benjamin A. Miller, Zohair Shafi, Wheeler Ruml, Yevgeniy
# Vorobeychik, Tina Eliassi-Rad, and Scott Alfeld, at SDM 2025.

# This material is based upon work supported by the United States Air Force
# under Air Force Contract No. FA8702-15-D-0001 and the Combat Capabilities
# Development Command Army Research Laboratory (under Cooperative Agreement
# Number W911NF-13-2-0045).  Any opinions, findings, conclusions or
# recommendations expressed in this material are those of the authors and do not
# necessarily reflect the views of the United States Air Force or Army Research
# Laboratory.

# Copyright (C) 2023
# Benjamin A. Miller, Zohair Shafi, Wheeler Ruml, Yevgeniy Vorobeychik, Tina
# Eliassi-Rad, and Scott Alfeld

# The software is provided to you on an As-Is basis

# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014).  Notwithstanding any copyright notice,
# U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above.  Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in
# this work

import networkx as nx
import numpy as np
import numpy.random as rand
from numpy import linalg as la
import random
import sys
import time
from scipy import stats
from scipy import sparse as sp
import pickle as pkl

import gurobipy as gp
from gurobipy import GRB


from numpy.random import MT19937

from numpy.random import RandomState, SeedSequence

# based on code from original PATHATTACK provided by the authors
def randomized_rounding(delta, costVec, constrMat):
    uniqueVals = np.unique(delta)
    entropy = -np.sum(np.nan_to_num(delta*np.log2(delta)))
    
    #sanity check: dimensions are correct
    nPaths, m = constrMat.shape
    assert(m==len(delta))
    
    if nPaths == 0:
        return [], 0, 0, 1
    nSamples = np.ceil(np.log(4*nPaths))
    slackFactor = 4*nSamples
    
    sample = np.zeros(delta.shape)
    numTries = 1
    for ii in range(int(nSamples)):
        r = rand.rand(m)
        sample = np.logical_or(sample, [(r[ii] < delta[ii]) for ii in range(m)])

    # check that constraints are satisfied
    constrNotSat = np.sum(constrMat@(sample.astype('float'))<=.999999) > 0
    tooBig = (costVec.dot(sample) >= (costVec.dot(delta)*slackFactor))
    
    while tooBig or constrNotSat:
        sample = np.zeros(delta.shape)
        numTries += 1
        if numTries > 100:
            print('failed after 100 tries, just use everything with a nonzero probability . . . ')
            sample = (delta > 0)
            constrNotSat = False
            tooBig = False
        else:
            for ii in range(int(nSamples)):
                r = rand.rand(m)
                sample = np.logical_or(sample, [(r[ii] < delta[ii]) for ii in range(m)])
            
            # check that constraints are satisfied
            constrNotSat = np.sum(constrMat@(sample.astype('float'))<=.999999) > 0
            tooBig = (costVec.dot(sample) >= (costVec.dot(delta)*slackFactor))
    
    includeInd = np.where(sample)[0]
    maxApproxFactor = np.nan_to_num(costVec.dot(sample)/costVec.dot(delta))
           
    return includeInd, entropy, numTries, maxApproxFactor

def get_path_length(G, p, weight='weight'):
    length = 0
    try:
        for i in range(len(p)-1):
            length += G.edges[p[i], p[i+1]][weight]
    except:
        leng = np.inf
    return length

def prob_attack(G, pStarDist, budgetDist):
    pAcc = 0
    pathNorm = 0
    for p in pStarDist:
        pStarEdges = [(p[k], p[k+1]) for k in range(len(p)-1)]
        result = PATHATTACK_LP(G, list(p), dontRemove=pStarEdges)
        
        pAcc += pStarDist[p]*(1-np.sum(budgetDist[:result['cost_removed']])/np.sum(budgetDist))
        pathNorm += pStarDist[p]
    return pAcc/pathNorm

def reconnect(G, removed_edges):
    CC = list(nx.connected_components(G))
    while len(CC) > 1:
        for e in removed_edges:
            if (e[0] in CC[0] and e[1] in CC[1]) or (e[1] in CC[0] and e[0] in CC[1]):
                G.add_edge(*e)
                removed_edges.remove(e)
                break
        CC = list(nx.connected_components(G))

def get_path_from_vector(C, x_pStar, sInd, tInd):
    colInd = np.where(x_pStar != 0)[0]
    Ctemp = C[:, colInd]@sp.diags(x_pStar[colInd])
    p = [sInd]
    while p[-1] != tInd:
        cEdge = Ctemp[p[-1]]
        ind1 = np.argmin(cEdge.data)
        if cEdge.data[ind1] >= 0:
            raise Exception('no edge from node')
        ind1 = cEdge.indices[ind1]

        col = sp.csr_matrix(Ctemp[:, ind1].transpose())
        ind2 = np.argmax(col.data)
        if col.data[ind2] <= 0:
            raise Exception('no edge from node')

        p.append(col.indices[ind2])
    return p

# PATHATTACK_LP: Run the linear-programming-based PATHATTACK algorithm
#   to identify edges to cut to make p* the shortest path, based on
#   the code provided by the authors of the original PATHATTACK paper
# Inputs:   G - an undirected networkx graph with weights and costs on
#               each edge
#           s - source vertex of the path
#           t - destination vertex of the path
#           pStar - a list of vertices denoting the order of the path
#           batch_size - the number of shortest paths to add before
#               retrying the optimization
# Outputs:  return_dict - a dictionary containing statistics of the
#               of the optimization, including running time and
#               costs of edges cut

def PATHATTACK_LP(G, pStar, batch_size=1, init_constraints = [], dontRemove=[]):
    
    start_time = time.time()
    ######################set up######################
    E = list(G.edges()) # list of edges
    M = len(E)          # number of edges
    
    # create a map from edges to indices in E
    eInd = dict()
    for i in range(len(E)):
        # edges are undirected, so map pairs in both orders
        eInd[E[i]] = i
        eInd[E[i][1], E[i][0]] = i
    
    s = pStar[0]
    t = pStar[-1]
    assert(len(pStar)==len(np.unique(pStar))) #no cycles
    #all edges are in the graph
    assert(np.prod([(((pStar[i], pStar[i+1]) in E) or ((pStar[i+1], pStar[i]) in E)) for i in range(len(pStar)-1)]))
    
    #make the p* vector
    xp = np.zeros((len(E)))
    for i in range(len(pStar)-1):
        xp[eInd[(pStar[i], pStar[i+1])]] = 1
    
    #get weights
    weights = nx.get_edge_attributes(G, "weight")
    w = np.array([weights[e] for e in E])
    
    #get costs
    costs = nx.get_edge_attributes(G, "cost")
    c = np.array([costs[e] for e in E])
    ##################################################
    
    
    zeros = np.zeros((M))       # vector of zeros
    ones = np.ones((M))         # vector of ones
    xp_ind = np.nonzero(xp)[0]  # indices of edges in p*
    max_length = np.dot(w, xp)  # length of p* (maximum length of paths to cut)
    
    # create a Gurobi model
    mod = gp.Model('minCut')
    # delta: indicator vector of edges to cut (continuous variable, apply
    # randomized rounding
    delta = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="delta")

    # constraints on delta (at most 1, at least 0, don't cut p*)
    mod.addConstr(delta <= ones, "deltaUpper")
    mod.addConstr(delta >= zeros, "deltaLower")
    if not dontRemove:
        mod.addConstr(xp@delta == 0)
    else:
        for p in dontRemove:
            x = np.zeros((len(E)))
            for i in range(len(p)-1):
                x[eInd[(p[i], p[i+1])]]=1
            mod.addConstr(x@delta==0)
    # objective: minimize cost
    mod.setObjective(delta@c, GRB.MINIMIZE)
    
    
    done = False    # loop until done
    nConst = 0      # count constraints
    if init_constraints:
        cols = [np.array([eInd[(x[i], x[i+1])] for i in range(len(x)-1)]) for x in init_constraints]
        rows = [i*np.ones((len(cols[i]))) for i in range(len(cols))]
        cols = np.concatenate(cols)
        rows = np.concatenate(rows)

        X = sp.csr_matrix((np.ones((len(cols))), (rows, cols)), shape=(len(init_constraints), M))
        mod.addConstr(X@delta >= 1, "initial path constraints")
        nConst = X.shape[0]
    else:
        X = sp.csr_matrix((0, M))
    final_constraints = init_constraints.copy()
    incremental_cost = []
        
        
        
    while not done:
        # perform minimization
        print('Minimize delta')
        mod.optimize()
        deltaHat = delta.X #get current value of delta
        
        cutInd, entropy, nTries, approxBound = randomized_rounding(deltaHat, c, X)
        current_cost = np.sum([G.edges[(E[i][0], E[i][1])]['cost'] for i in cutInd])
        while(len(incremental_cost) < len(final_constraints)):
            incremental_cost.append(None)
        incremental_cost.append(cutInd)
        
        
        
        print('find next constraint')
        # find new shortest path probably not cut
        Gtemp = G.copy()
        #cut edges where delta=1
        for i in cutInd:
            Gtemp.remove_edge(E[i][0], E[i][1])

        # iterate over paths from s to t
        P = nx.shortest_simple_paths(Gtemp, s, t, weight='weight')
        ctr = 0
        pConst = [] # list to hold new constraints
        while ctr < batch_size: # find batch_size new shortest paths
            try:
                p = next(P)
            except:
                print('no paths') # break if there are no more such paths
                break
            
            length = 0  # path length
            # loop over edges on the path
            for i in range(len(p)-1):
                ind = eInd[(p[i], p[i+1])]  # index of the current edge
                length += w[ind]        # add weight (length) at current edge
            
            # if the path is too long, don't consider anything further
            if length > max_length:
                break
            elif p == pStar: # if the path is p*, keep iterating
                print('we see p*')
                continue
            else:
                # if the path is not long enough, add a new constraint
                print('adding a new path constraint: '+str(p))
                pConst.append(p)
                ctr += 1

        if len(pConst) == 0: # done if no new constraints were added
            done = True
        else: # otherwise, add the new constraints to the linear program
            for p in pConst: # for each constraint
                # get the indicator vector for edges in the path
                xHat = np.zeros((M))
                for i in range(len(p)-1):
                    ind = eInd[(p[i], p[i+1])]
                    xHat[ind] = 1

                # add the new constraint to the model
                nConst += 1
                xHat = sp.csr_matrix(xHat)
                mod.addConstr(xHat@delta >= 1, "path constraint "+str(nConst))
                X = sp.vstack([X, xHat])
            final_constraints += pConst
                
    
    cutEdges = [E[i] for i in cutInd]
    totalCost = 0
    costRemoved = 0
    for e in G.edges:
        totalCost += G.edges[e]['cost']
        if (e in cutEdges) or ((e[1], e[0]) in cutEdges):
            costRemoved += G.edges[e]['cost']
    cut_graph = G.copy()
    for e in cutEdges:
        cut_graph.remove_edge(e[0], e[1])
    reconnect(cut_graph, cutEdges)
    while(len(incremental_cost) < len(final_constraints)):
        incremental_cost.append(None)
    incremental_cost.append(cutEdges)
    return_dict = dict()
    return_dict['algorithm_time'] = time.time() - start_time
    return_dict['cut_sp'] = nx.shortest_path(cut_graph, source=s, target=t, weight='weight')
    return_dict['edges_removed'] = cutEdges
    return_dict['num_edges_removed'] = len(cutEdges)
    return_dict['total_edges'] = G.number_of_edges()
    return_dict['cost_removed'] = costRemoved
    return_dict['inc_cost'] = incremental_cost
    return_dict['total_cost'] = totalCost
    return_dict['num_constraints'] = nConst
    return_dict['constraints'] = final_constraints
    return_dict['entropy'] = entropy
    return_dict['approx_bound'] = approxBound
    return_dict['num_randomized_rounding_tries'] = nTries
    
    return return_dict


def diffCost(x, negSlope=.5, posSlope=.05):
    if x < 0:
        retVal = -x*negSlope
    else:
        retVal = x*posSlope
    return retVal

def weightChangeCost(G, P, attacks, attack_prob, successCost, f0, f1):

    V = list(G.nodes())
    N = len(V)
    assert(len(attacks)==len(attack_prob))

    #cost = successCost*np.sum(attack_prob)
    success_cost = successCost*np.sum(attack_prob)
    length_cost = 0
    error_cost = 0

    APSP = dict(nx.all_pairs_dijkstra_path(G, weight='observed_weight'))
    p_no_attack = 1-np.sum(attack_prob)
    for i in range(N):
        for j in range(i+1, N):
            path = APSP[V[i]][V[j]]
            true_length = get_path_length(G, path, 'true_weight')
            obs_length = get_path_length(G, path, 'observed_weight')
            if obs_length > true_length:
                inc_cost = (obs_length-true_length)*f1
            else:
                inc_cost = (true_length-obs_length)*f0
            #inc_cost += true_length
            #cost += P[V[i]][V[j]]*p_no_attack*inc_cost
            length_cost += P[V[i]][V[j]]*p_no_attack*true_length
            error_cost += P[V[i]][V[j]]*p_no_attack*inc_cost
    
    for p in range(len(attacks)):
        G1 = G.copy()
        G1.remove_edges_from(attacks[p])
        APSP = dict(nx.all_pairs_dijkstra_path(G1, weight='observed_weight'))
        for i in range(N):
            for j in range(i+1, N):
                path = APSP[V[i]][V[j]]
                true_length = get_path_length(G1, path, 'true_weight')
                obs_length = get_path_length(G1, path, 'observed_weight')
                if obs_length > true_length:
                    inc_cost = (obs_length-true_length)*f1
                else:
                    inc_cost = (true_length-obs_length)*f0
                #inc_cost += true_length
                #cost += P[V[i]][V[j]]*attack_prob[p]*inc_cost
                length_cost += P[V[i]][V[j]]*attack_prob[p]*true_length
                error_cost += P[V[i]][V[j]]*attack_prob[p]*inc_cost
                
    return length_cost, error_cost, success_cost

def get_avg_budget(G, pathDict):
    bSum = 0
    bNorm = 0
    for p in pathDict:
        pEdges = [(p[i], p[i+1]) for i in range(len(p)-1)]
        result = PATHATTACK_LP(G, list(p), dontRemove=pEdges)
        bSum += result['cost_removed']
        bNorm += 1
    if bNorm == 0:
        bAvg = 0
    else:
        bAvg = bSum/bNorm

    return bAvg

def lower_bound_attack_probability(G, pDist, B):
    G1 = G.copy()
    for e in G1.edges:
        G1.edges[e]['weight'] = 0
    
    pAttack = 0
    for p in pDist:
        pEdges = [(p[i], p[i+1]) for i in range(len(p)-1)]
        result = PATHATTACK_LP(G1, list(p), dontRemove=pEdges)
        Gprime = G1.copy()
        Gprime.remove_edges_from(result['edges_removed'])
        CCs = list(nx.connected_components(Gprime))
        
        pAttack += pDist[p]*np.sum(B[np.arange(len(B)) >= result['num_edges_removed']-len(CCs)+1])
    
    return pAttack

def minimize_attack_probability(G, pStar):
    N = G.number_of_nodes()
    Gprime = G.copy()
    
    pStarEdges = [(pStar[i], pStar[i+1]) for i in range(len(pStar)-1)]
    s = pStar[0]
    t = pStar[-1]
    
    weights = nx.get_edge_attributes(G, "weight")
    w = np.array([weights[e] for e in weights])
    W = np.sum(np.sort(w)[-N:])+1
    
    for e in pStarEdges:
        Gprime.edges[e]['weight'] += W
    
    
    return (Gprime, PATHATTACK_LP(Gprime, pStar, dontRemove=pStarEdges))


def _tuples_using_path(path_edge_ind, pathsByEdge):
    
    #get the paths that use the edges from path_edges
    all_tuples = None
    for i in path_edge_ind:
        current_edge_paths = set(pathsByEdge[i])
        if all_tuples is None:
            all_tuples = current_edge_paths
        else:
            all_tuples = all_tuples.intersection(current_edge_paths)
    return list(all_tuples)


def _compute_path_weight(path_edge_ind, pathsByEdge, D, p_attack, V):
    all_tuples = _tuples_using_path(path_edge_ind, pathsByEdge)
    
    nPaths = len(p_attack)
    path_weight = 0
    for (p, i, j) in all_tuples:
        path_weight += p_attack[p]*D[V[i]][V[j]]
    return path_weight

def get_path_from_vector(C, x_pStar, sInd, tInd):
    colInd = np.where(x_pStar != 0)[0]
    Ctemp = C[:, colInd]@sp.diags(x_pStar[colInd])
    p = [sInd]
    while p[-1] != tInd:
        cEdge = Ctemp[p[-1]]
        ind1 = np.argmin(cEdge.data)
        if cEdge.data[ind1] >= 0:
            raise Exception('no edge from node')
        ind1 = cEdge.indices[ind1]

        col = sp.csr_matrix(Ctemp[:, ind1].transpose())
        ind2 = np.argmax(col.data)
        if col.data[ind2] <= 0:
            raise Exception('no edge from node')

        p.append(col.indices[ind2])
    return p

def get_path_through_target(G, target, s, t):
    
    E = list(G.edges)
    V = list(G.nodes)
    M = len(E)
    N = len(V)

    eInd = dict()
    for i in range(M):
        eInd[E[i]] = i
        eInd[(E[i][1], E[i][0])] = i

    vInd = dict()
    for i in range(N):
        vInd[V[i]] = i
    
    #get weights 
    weights = nx.get_edge_attributes(G, "weight")
    w = np.array([weights[e] for e in E])

    C = nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True)
    C = sp.csr_matrix(C)

    zeros = np.zeros((M))
    ones = np.ones((M))

    d1 = np.zeros((N))
    d1[vInd[s]] -= 1
    d1[vInd[target]] += 1

    d2 = np.zeros((N))
    d2[vInd[target]] -= 1
    d2[vInd[t]] += 1

    dAbs = 2*np.ones((N))
    dAbs[vInd[s]] = 1
    dAbs[vInd[t]] = 1

    x_target = np.zeros((M))


    mod = gp.Model('shortestPathThroughTarget')
    mod.setParam("Threads", 1)
    Cabs = np.abs(C)

    x1pos = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="x1pos")
    x2pos = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="x2pos")
    x1neg = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="x1neg")
    x2neg = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, name="x2neg")
    x1 = (x1pos, x1neg)
    x2 = (x2pos, x2neg)

    mod.addConstr(x1pos >= zeros, "x1posLower")
    mod.addConstr(x2pos >= zeros, "x2posLower")
    mod.addConstr(x1neg >= zeros, "x1negLower")
    mod.addConstr(x2neg >= zeros, "x2negLower")
    mod.addConstr(x1pos+x2pos+x1neg+x2neg <= ones, "xUpper")

    mod.addConstr(C@x1pos-C@x1neg  == d1, "s_to_target")
    mod.addConstr(C@x2pos-C@x2neg == d2, "target_to_t")
    print('d2')
    print(np.where(d2)[0])
    print(d2[np.where(d2)[0]])
    mod.addConstr(Cabs@x1pos+Cabs@x1neg+Cabs@x2pos+Cabs@x2neg+Cabs@x_target <= dAbs, "no repeats")

    mod.setObjective(x1pos@w + x2pos@w + x1neg@w + x2neg@w + x_target@w, GRB.MINIMIZE)
    mod.optimize()

    x1new = x1[0].X-x1[1].X

    p1 = get_path_from_vector(C, x1new, vInd[s], vInd[target])
    p1 = [V[v] for v in p1]

    x1Vec = np.zeros((M))
    for i in range(len(p1)-1):
        C1 = C[vInd[p1[i]], eInd[(p1[i], p1[i+1])]]
        C2 = C[vInd[p1[i+1]], eInd[(p1[i], p1[i+1])]]
        x1Vec[eInd[(p1[i], p1[i+1])]] = 1 - 2*(C1 > C2)
    if la.norm(x1new-x1Vec) > 0:
        # chose from several options
        print('warning: several paths returned')
        oldObj = mod.objVal

        tempConstraint = mod.addConstr(x1[0]-x1[1]==x1Vec, 'fix x1')
        mod.optimize()
        if mod.objVal > oldObj:
            print(f"WARNING: raised objective value from {oldObj} to {mod.objVal}")
        mod.remove(tempConstraint)
    x2new = x2[0].X-x2[1].X
    p2 = get_path_from_vector(C, x2new, vInd[target], vInd[t])
    p2 = [V[v] for v in p2]

    p = p1 + [p2[i] for i in range(1, len(p2))]

    return p

def _get_path_from_ind_list(ind_list, E, source, dest, directed=False):
    edges = [E[i] for i in ind_list]
    if directed:
        raise Exception('not implemented')
    else:
        #initialize
        edgesByNode = {}
        for e in edges:
            edgesByNode[e[0]] = []
            edgesByNode[e[1]] = []
        
        for e in edges:
            edgesByNode[e[0]].append(e)
            edgesByNode[e[1]].append(e)
        
        path = [source]
        current_end = source
        while current_end != dest:
            possible_edges = edgesByNode[path[-1]]
            if possible_edges[0][0] == current_end:
                other_node = possible_edges[0][1]
            else:
                other_node = possible_edges[0][0]
            if other_node in path:
                if possible_edges[1][0] == current_end:
                    other_node = possible_edges[1][1]
                else:
                    other_node = possible_edges[1][0]
            assert(other_node not in path)
            path.append(other_node)
            current_end = path[-1]
    return path


def _update_constraints(mod, min_length_tuple, alt_path, path_dict, diffVec, path_diff_constr, trueDist, path_len_constr, wPrime, w, pair_index, V, E,  eInd, pathsByEdge):
    
    p, i, j = min_length_tuple
    M = len(E)
    current_sp = path_dict[p][i][j]
    current_path_ind = [eInd[(current_sp[k], current_sp[k+1])] for k in range(len(current_sp)-1)]
    k = len(current_path_ind)
    current_path_vec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), current_path_ind)), (1, M))
    
    # case 1: force alt_path to be longer than the current shortest path
    
    alt_path_ind = [eInd[(alt_path[k], alt_path[k+1])] for k in range(len(alt_path)-1)]
    k = len(alt_path_ind)
    alt_path_vec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), alt_path_ind)), (1, M))
    
    temp_constraint = mod.addConstr(alt_path_vec@wPrime >= current_path_vec@wPrime)
    
    
    #measure the objective for case 1
    try:
        mod.optimize()
        obj1 = mod.ObjVal
    except:
        obj1 = np.inf
    
    #case 2: make alt_path_ind the current shortest path
    mod.remove(temp_constraint)
    all_tuples = _tuples_using_path(current_path_ind, pathsByEdge)
    new_path = {}
    for p, i, j in all_tuples:
        ctr = pair_index[(i, j)]
        
        #create new shortest path
        old_path = path_dict[p][i][j]
        old_path_ind = [eInd[(old_path[k], old_path[k+1])] for k in range(len(old_path)-1)]
        new_path_ind = np.union1d(np.setdiff1d(old_path_ind, current_path_ind), alt_path_ind)
        k = len(new_path_ind)
        new_path_vec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), new_path_ind)), (1, M))
        new_path[(p, i, j)] = new_path_ind
        
        mod.remove(path_diff_constr[p][i][j])
        mod.remove(path_len_constr[p][i][j])
        path_diff_constr[p][i][j] = mod.addConstr(new_path_vec@wPrime - new_path_vec@w == diffVec[p][ctr])
        path_len_constr[p][i][j] = mod.addConstr(new_path_vec@w == trueDist[p][ctr])
    
    mod.optimize()
    obj2 = mod.ObjVal
    
    print(f"objective in case 1: {obj1}")
    print(f"objective in case 2: {obj2}")
    
    if obj2 < obj1:
        # update paths in dictionary
        for p, i, j in all_tuples:
            src = V[i]
            dst = V[j]
            path_dict[p][i][j] = _get_path_from_ind_list(new_path[(p, i, j)], E, src, dst)
            for e in current_path_ind:
                pathsByEdge[e].remove((p, i, j))
            for e in new_path[(p, i, j)]:
                pathsByEdge[e].append((p, i, j))
            
            
    else:
        # put back the old constraints
        for p, i, j in all_tuples:
            ctr = pair_index[(i, j)]
            
            path = path_dict[p][i][j]
            path_ind = [eInd[(path[k], path[k+1])] for k in range(len(path)-1)]
            k = len(path_ind)
            pathVec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), path_ind)), (1, M))
            
            mod.remove(path_diff_constr[p][i][j])
            mod.remove(path_len_constr[p][i][j])
            path_diff_constr[p][i][j] = mod.addConstr(pathVec@wPrime - pathVec@w == diffVec[p][ctr])
            path_len_constr[p][i][j] = mod.addConstr(pathVec@w == trueDist[p][ctr])
        #and add the new one:
        mod.addConstr(alt_path_vec@wPrime >= current_path_vec@wPrime)


def _minimize_cost_keep_paths(G, E, V, w, P, constraints_and_removed_edges, p_attack, success_cost, D, f0, f1):
    
    start_time = time.time()
    #E = list(G.edges())
    #V = list(G.nodes())
    M = len(E)
    N = G.number_of_nodes()
    eInd = {}
    wPrime_init = np.zeros((M))
    for i in range(M):
        e = E[i]
        eInd[e] = i
        eInd[(e[1], e[0])] = i
        wPrime_init[i] = G.edges[e]['weight']
    nPairs = int(N*(N-1)/2)
    vInd = {}
    for i in range(N):
        vInd[V[i]] = i
    
    pathList = list(P.keys())
    nPaths = len(pathList)
    #eps = [1 for _ in range(nPaths+1)]
    
    assert(len(constraints_and_removed_edges)==nPaths)
    assert(len(p_attack)==nPaths)
    
    d = np.zeros((nPairs)) #proportion of trips between node pairs
    ctr = 0
    min_path = {}
    pair_ind = {}
    random_pairs = []
    for i in range(N):
        min_path[i] = {}
        for j in range(i+1, N):
            d[ctr] = D[V[i]][V[j]]
            pair_ind[(i, j)] = ctr
            ctr += 1
            min_path[i][j] = None
            random_pairs.append((i, j))
    
    #initialize linear program
    mod = gp.Model('defenderModel')
    attack_obj = mod.addMVar(shape=nPaths+1, vtype=GRB.CONTINUOUS, lb=0.0, ub=float('inf'))
    wPrime = mod.addMVar(shape=M, vtype=GRB.CONTINUOUS, lb=0.0, ub=float('inf'))
    diffVec = []
    trueDist = []
    overEst = []
    underEst = []
    for p in range(nPaths+1):
        
        diffVec.append(mod.addMVar(shape=nPairs, vtype=GRB.CONTINUOUS, lb=-float('inf')))
        trueDist.append(mod.addMVar(shape=nPairs, vtype=GRB.CONTINUOUS, lb=0.0))
        overEst.append(mod.addMVar(shape=nPairs, vtype=GRB.CONTINUOUS, lb=0.0))
        underEst.append(mod.addMVar(shape=nPairs, vtype=GRB.CONTINUOUS, lb=0.0))
        mod.addConstr(overEst[p] - underEst[p] == diffVec[p])
        selectVec = np.zeros((nPaths+1))
        selectVec[p] = 1
        #mod.addConstr(attack_obj[p] == d@trueDist[p] + f1*(d@overEst[p]) + f0*(d@underEst[p]))
        mod.addConstr(selectVec@attack_obj == d@trueDist[p] + (f1*d)@overEst[p] + (f0*d)@underEst[p])
        
    
    pAttack_vec = np.array([p_attack[p]*P[pathList[p]] for p in range(nPaths)])
    pAttack = np.sum(pAttack_vec)
    pAttack_vec = np.append(pAttack_vec, [1-pAttack])
    mod.setObjective(pAttack_vec@attack_obj + pAttack*success_cost, GRB.MINIMIZE)
    
    
    path_dict = {}
    for p in range(nPaths+1):
        if p < nPaths:
            constraint_paths, edges_removed = constraints_and_removed_edges[p]
            pStar = list(pathList[p])
            
            #set pStar edges to 0 weight
            pStarEdges = [(pStar[i], pStar[i+1]) for i in range(len(pStar)-1)]
            pStarVec = np.zeros((M))
            for e in pStarEdges:
                pStarVec[eInd[e]] = 1
            
            #first constraints: constraint_paths need to compete with p*
            for path in constraint_paths:
                pathVec = np.zeros((M))
                for i in range(len(path)-1):
                    pathVec[eInd[(path[i], path[i+1])]] = 1
                mod.addConstr(pathVec@wPrime <= pStarVec@wPrime)
        else:
            constraint_paths, edges_removed = [], []
            

        path_dict[p] = {}
        Gprime = G.copy()
        Gprime.remove_edges_from(edges_removed)
        APSP = dict(nx.all_pairs_dijkstra_path(Gprime, weight='weight'))
        
        for i in range(N):
            path_dict[p][i] = {}
            for j in range(i+1, N):
                path = APSP[V[i]][V[j]]
                pathInd = [eInd[(path[k], path[k+1])] for k in range(len(path)-1)]
                k = len(pathInd)

                pathVec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), pathInd)), (1, M))
                path_dict[p][i][j] = path
                ctr = pair_ind[(i, j)]
                mod.addConstr(pathVec@wPrime - pathVec@w == diffVec[p][ctr])
                mod.addConstr(pathVec@w == trueDist[p][ctr])
                if Gprime.has_edge(V[i], V[j]) and (len(path) > 2):
                    # if direct edge is not shortest path, add constraint
                    edgeVec = sp.csr_matrix((np.ones((1)), (np.zeros((1)), [eInd[(V[i], V[j])]])), (1, M))
                    mod.addConstr(pathVec@wPrime <= edgeVec@wPrime)
    wPrimeVec = np.zeros((M))
    for e in E:
        wPrimeVec[eInd[e]] = G.edges[e]['weight']
    #mod.addConstr(wPrime == wPrimeVec)
    # this is for debugging: is the initial point feasible?
    tempConstr = mod.addConstr(wPrime == wPrime_init)
    mod.optimize()
    #mod.computeIIS()
    #idx = rand.randint(1e7)
    #mod.write("model"+str(idx)+".ilp")
    #get all the initial values
    initial_obj = mod.ObjVal
    attack_obj_init = attack_obj.X
    diffVec_init = []
    trueDist_init = []
    overEst_init = []
    underEst_init = []
    for p in range(nPaths+1):
        
        diffVec_init.append(diffVec[p].X)
        trueDist_init.append(trueDist[p].X)
        overEst_init.append(overEst[p].X)
        underEst_init.append(underEst[p].X)
    mod.remove(tempConstr)
    
    done = False
    #mod.setParam("TimeLimit", 600.0);
    mod.setParam("OptimalityTol", 1e-4);
    best_obj = 0
    while (not done) and (time.time()-start_time < 24*60*60):
        wPrime.Start = wPrime_init
        attack_obj.Start = attack_obj_init
        del tempConstr
        tempConstr = []
        tempConstr.append(mod.addConstr(wPrime == wPrime_init))
        tempConstr.append(mod.addConstr(attack_obj == attack_obj_init))
        for p in range(nPaths+1):
            diffVec[p].Start = diffVec_init[p]
            trueDist[p].Start = trueDist_init[p]
            overEst[p].Start = overEst_init[p]
            underEst[p].Start = underEst_init[p]
            tempConstr.append(mod.addConstr(diffVec[p] == diffVec_init[p]))
            tempConstr.append(mod.addConstr(trueDist[p] == trueDist_init[p]))
            tempConstr.append(mod.addConstr(overEst[p] == overEst_init[p]))
            tempConstr.append(mod.addConstr(underEst[p] == underEst_init[p]))
        mod.optimize()
        #mod.computeIIS()
        #idx = rand.randint(1e7)
        #mod.write("model"+str(idx)+".ilp")
        print(f"re-initialized with optimal value {mod.ObjVal}")
        mod.remove(tempConstr)
        mod.optimize()
        best_obj = mod.ObjVal
        if mod.ObjVal > initial_obj*0.9999:
            break
        
        G1 = G.copy()
        for e in G1.edges:
            G1.edges[e]['weight'] = np.maximum(wPrime.X[eInd[e]], 0.0)
        
        min_length_diff = np.inf
        min_length_tuple = ()
        min_length_hops = np.inf
        min_length_weight = 0
        alt_path = []
        random_paths = list(range(nPaths+1))
        rand.shuffle(random_paths)
        new_constraint = False
        for p in random_paths:
            #apply attack
            if p < nPaths:
                constraint_paths, edges_removed = constraints_and_removed_edges[p]
                G1prime = G1.copy()
                G1prime.remove_edges_from(edges_removed)
                pStar = list(pathList[p])
                sps = nx.shortest_simple_paths(G1prime, pStar[0], pStar[-1], 'weight')
                path = next(sps)
                constr_path = None
                if path == pStar:
                    try:
                        next_path = next(sps)
                        length_diff = get_path_length(G1prime, next_path) - get_path_length(G1prime, pStar)
                    except:
                        length_diff = np.inf
                    #if length_diff == 0:
                    if length_diff < 0.000001:
                        constr_path = next_path
                else:
                    length_diff = get_path_length(G1prime, pStar) - get_path_length(G1prime, path)
                    constr_path = path
                if constr_path is not None:
                    constr_path_ind = [eInd[(constr_path[k], constr_path[k+1])] for k in range(len(constr_path)-1)]
                    k = len(constr_path_ind)
                    constr_path_vec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), constr_path_ind)), (1, M)) 
                    pStar_ind = [eInd[(pStar[k], pStar[k+1])] for k in range(len(pStar)-1)]
                    k = len(pStar_ind)
                    pStar_vec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), pStar_ind)), (1, M)) 
                    eps = get_path_length(G, constr_path) - get_path_length(G, pStar)
                    assert(eps > 0)
                    mod.addConstr(pStar_vec@wPrime <= constr_path_vec@wPrime - eps/2)
                    new_constraint = True
            else:
                constraint_paths, edges_removed = [], []
        if new_constraint:
            print('adding new attack constraint')
            continue
        
        print('checking APSP')
        for p in random_paths:
            #apply attack
            if p < nPaths:
                constraint_paths, edges_removed = constraints_and_removed_edges[p]
            else:
                constraint_paths, edges_removed = [], []
            G1prime = G1.copy()
            G1prime.remove_edges_from(edges_removed)
            
            #get shortest paths
            APSP_new = dict(nx.all_pairs_dijkstra_path(G1prime, weight='weight'))
            
            #compare shortest path length to chosen path length
            #choose shortest true distance with greatest usage
            #for i in range(N):
            #    for j in range(i+1, N):
            rand.shuffle(random_pairs)
            done = True
            for (i, j) in random_pairs:
                path_old = path_dict[p][i][j]
                path_new = APSP_new[V[i]][V[j]]
                length_diff = get_path_length(G1prime, path_old) - get_path_length(G1prime, path_new)
                #if length_diff > 0:
                if length_diff >= 0.000001:
                    path_old_ind = [eInd[(path_old[k], path_old[k+1])] for k in range(len(path_old)-1)]
                    k = len(path_old_ind)
                    path_old_vec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), path_old_ind)), (1, M))
                    path_new_ind = [eInd[(path_new[k], path_new[k+1])] for k in range(len(path_new)-1)]
                    k = len(path_new_ind)
                    path_new_vec = sp.csr_matrix((np.ones((k)), (np.zeros((k)), path_new_ind)), (1, M))
                    mod.addConstr(path_old_vec@wPrime <= path_new_vec@wPrime)
                    done = False
                    break
            
            if not done:
                break
    if mod.ObjVal >= initial_obj+0.000001:
        raise Exception("objective increased by "+str(mod.ObjVal-initial_obj))
    wPrimeVec = wPrime.X
    Gprime = G.copy()
    if done:
        for e in E:
            # rounding errors may result in small negative values
            Gprime.edges[e]['weight'] = np.maximum(wPrimeVec[eInd[e]], 0.0)
    return Gprime, initial_obj, best_obj


def zero_sum_heuristic(G, D, B, P, f0, f1):
    nPath = len(P)
    pathList = list(P.keys())
    p_attack = []
    for p in pathList:
        pAttack = 1
        Eremove = []
        G0 = G.copy()
        for e in G0.edges:
            G0.edges[e]['observed_weight'] = G0.edges[e]['weight']
        while pAttack > 1e-6/nPath:
            temp_result, temp_diff, temp_edges = attack_and_get_difference(G0, list(p))
            Eremove = temp_result['edges_removed']
            if not temp_edges:
                pAttack = 0
            else:
                pAttack = np.sum(B[np.arange(len(B)) >= temp_result['num_edges_removed']])
                if pAttack > 1e-6/nPath:
                    G0.edges[temp_edges[0]]['observed_weight'] += max(temp_diff, 1.0)
                
        p_attack.append(P[p]*np.sum(B[np.arange(len(B)) >= len(Eremove)]))
    
    G0 = G.copy()
    for e in G0.edges:
        G0.edges[e]['observed_weight'] = G0.edges[e]['weight']
    for ind in np.argsort(p_attack):
        p = pathList[ind]
        pAttack = 1
        Eremove = []
        while pAttack > 1e-6/nPath:
            temp_result, temp_diff, temp_edges = attack_and_get_difference(G0, list(p))
            Eremove = temp_result['edges_removed']
            if not temp_edges:
                pAttack = 0
            else:
                pAttack = np.sum(B[np.arange(len(B)) >= temp_result['num_edges_removed']])
                if pAttack > 1e-6/nPath:
                    G0.edges[temp_edges[0]]['observed_weight'] += max(temp_diff, 1.0)

    E = list(G.edges())
    V = list(G.nodes())
    weights = nx.get_edge_attributes(G, "weight")
    w = np.array([weights[e] for e in E])

    p_attack_vec = []
    constraints_and_removed_edges = []
    attacks = []
    p_attack = []
    for p in pathList:
        temp_result, temp_diff, temp_edges = attack_and_get_difference(G0, list(p))
        constraints_and_removed_edges.append((temp_result['constraints'], temp_result['edges_removed']))
        attacks.append(temp_result['edges_removed'])
        p_attack_vec.append(np.sum(B[np.arange(len(B)) >= temp_result['num_edges_removed']]))
        p_attack.append(p_attack_vec[-1]*P[p])
    #return _minimize_cost_new(G, P, constraints_and_removed_edges,
    #                          pAttack, 0, D, f0, f1)
    for e in G0.edges:
        G0.edges[e]['weight'] = G0.edges[e]['observed_weight']
    print('starting local minimization')
    Gfinal, costUB, costLB = _minimize_cost_keep_paths(G0, E, V, w, P, constraints_and_removed_edges, p_attack_vec, 0, D, f0, f1)
    return (lower_bound_attack_probability(G, P, B), Gfinal,
            weightChangeCost(Gfinal, D, attacks, p_attack, 1, f0, f1), costUB, costLB) 
        

def attack_and_get_difference(G, pStar):
    
    pStarEdges = [(pStar[k], pStar[k+1]) for k in range(len(pStar)-1)]
    Gprime = G.copy()
    for u, v, data in Gprime.edges(data=True):
        data['weight'] = data.pop('observed_weight')
    result = PATHATTACK_LP(Gprime, pStar, dontRemove=pStarEdges)

    Gprime.remove_edges_from(result['edges_removed'])
    s = pStar[0]
    t = pStar[-1]
    sps = nx.shortest_simple_paths(Gprime, s, t, 'weight')
    p = next(sps)
    assert(p == pStar)
    try:
        p = next(sps)
        len_diff = get_path_length(Gprime, p)-get_path_length(Gprime, pStar)
        p_edges = [(p[k], p[k+1]) for k in range(len(p)-1)]
        inc_edges = [e for e in pStarEdges if e not in p_edges]
    except:
        len_diff = np.inf
        inc_edges = []
    
    return result, len_diff, inc_edges


def increment_edge_heuristic(G, D, B, P, f0, f1, success_cost, cost_threshold, max_iter=500, prob_threshold=1e-6,  min_inc=1):
    start_time = time.time()
    E = list(G.edges())
    V = list(G.nodes())
    weights = nx.get_edge_attributes(G, "weight")
    w = np.array([weights[e] for e in E])
    #get all edges on a target path
    G1 = G.copy()
    ctr = 0
    for e in G1.edges:
        G1.edges[e]['observed_weight'] = G1.edges[e]['weight']
        G1.edges[e]['true_weight'] = G1.edges[e]['weight']
    
    all_edges = set()
    results = []
    len_diff = []
    diff_edges = []
    pList = list(P.keys())
    path_prob = []
    for p in pList:
        for k in range(len(p)-1):
            if ((p[k], p[k+1]) not in all_edges) and ((p[k+1], p[k]) not in all_edges):
                all_edges.add((p[k], p[k+1]))
        temp_result, temp_diff, temp_edges = attack_and_get_difference(G1, list(p))
        results.append(temp_result)
        len_diff.append(temp_diff)
        diff_edges.append(temp_edges)
        path_prob.append(P[p])
    p_attack = [np.sum(B[np.arange(len(B)) >= r['cost_removed']]) for r in results]
    p_attack = [p_attack[i]*path_prob[i] for i in range(len(pList))]
    attacks = [r['edges_removed'] for r in results]
    p_attack_LB = lower_bound_attack_probability(G, P, B)
    print(f'lower bound attack probability: {p_attack_LB}')

    #current_cost = weightChangeCost(G1, D, attacks, p_attack, success_cost, f0, f1)
    Lc, Le, Ls = weightChangeCost(G1, D, attacks, p_attack, success_cost, f0, f1)
    current_cost = Lc + Le + Ls
    
    cost_sequence = [[Lc, Le, Ls]]
    probs_sequence = [np.sum(p_attack)]
    bestCost = current_cost
    bestWeights = w
    while ctr < max_iter and (np.sum(p_attack) >= prob_threshold) and (current_cost >= cost_threshold) and (time.time()-start_time < 24*60*60):
        min_prob = 2
        min_prob_edge = None
        min_prob_diff = None
        min_prob_length = -1
        inc = {}
        #initialize all possible edges
        all_edges = set()
        for p in range(len(pList)):
            for e in diff_edges[p]:
                all_edges.add(e)
        all_edges = list(all_edges)
        rand.shuffle(all_edges)
        for e in all_edges:
            inc[e] = np.inf
        for p in range(len(pList)):
            for e in diff_edges[p]:
                inc[e] = min(inc[e], max(len_diff[p], min_inc))
        for e in all_edges:
            if inc[e] == np.inf:
                inc[e] = 0
        
        for e in all_edges:
            if inc[e] > 0:
                G1.edges[e]['observed_weight'] += inc[e]
                results = []
                len_diff = []
                diff_edges = []
                length_sum = 0
                for p in pList:
                    temp_result, temp_diff, temp_edges = attack_and_get_difference(G1, list(p))
                    results.append(temp_result)
                    len_diff.append(temp_diff)
                    diff_edges.append(temp_edges)
                    length_sum += get_path_length(G1, p, weight='observed_weight')
                #p_mask = (np.arange(len(B)) >= r['cost_removed'])
                p_attack = [np.sum(B[np.arange(len(B)) >= r['cost_removed']]) for r in results]
                p_attack = [p_attack[i]*path_prob[i] for i in range(len(pList))]
                attacks = [r['edges_removed'] for r in results]
                #temp_cost = weightChangeCost(G1, D, attacks, p_attack, success_cost, f0, f1)
                if np.sum(p_attack) < min_prob or (np.sum(p_attack) == min_prob and min_prob_length < length_sum):
                    min_prob = np.sum(p_attack)
                    min_prob_edge = e
                    min_prob_diff = len_diff
                    min_prob_cost = weightChangeCost(G1, D, attacks, p_attack, success_cost, f0, f1)
                    min_prob_length = length_sum
                    min_diff_edges = diff_edges
                    min_results = results
                    
                G1.edges[e]['observed_weight'] -= inc[e]
        if min_prob_edge is None:
            break
        G1.edges[min_prob_edge]['observed_weight'] += inc[min_prob_edge]
        len_diff = min_prob_diff
        print(f"iteration {ctr}, cost={min_prob_cost}, Pr(attack)={min_prob}")
        ctr += 1
        constraints_and_removed_edges = []
        for p in pList:
            
            constraints_and_removed_edges.append([(r['constraints'], r['edges_removed']) for r in min_results])
        p_attack = [np.sum(B[np.arange(len(B)) >= r['cost_removed']]) for r in min_results]
        
        
        
        p_attack = [p_attack[i]*path_prob[i] for i in range(len(pList))]
        min_prob = np.sum(p_attack)
        attacks = [r['edges_removed'] for r in min_results]
        Lc, Le, Ls = weightChangeCost(G1, D, attacks, p_attack, success_cost, f0, f1)
        min_prob_cost = Lc + Le + Ls
        cost_sequence.append([Lc, Le, Ls])
        probs_sequence.append(min_prob)
        print(f"iteration {ctr-0.5}, cost={min_prob_cost}, Pr(attack)={min_prob}")
        if bestCost > min_prob_cost:
            bestCost = min_prob_cost
            current_weights = nx.get_edge_attributes(G1, "weight")
            bestWeights = np.array([current_weights[e] for e in E])
    
    return ([bestWeights, bestCost], cost_sequence, probs_sequence)



if __name__ == "__main__":

    graphName = sys.argv[1]
    trialInd = int(sys.argv[2])
    method = sys.argv[3]
    num_paths = int(sys.argv[4])
    terminals = sys.argv[5]
    success_factor = float(sys.argv[6])
    budget_factor = float(sys.argv[7])
    resultDir = argv[8]
    dataDir = argv[9]

    random.seed(81238.2345+9235.893456*trialInd)
    rand.seed(892358293+27493463*trialInd)
    print(f"Gurobi version: {gp.gurobi.version()}")
    assert(terminals in ['same', 'different', 'community'])
    if terminals=='community':
        assert(graphName in ['as', 'sbm'])


    if graphName == 'er':
        G = nx.erdos_renyi_graph(250, .048)
    elif graphName == 'ba':
        G = nx.barabasi_albert_graph(250, 6)
    elif graphName == 'ws':
        G = nx.watts_strogatz_graph(250, 12, .05)
    elif graphName == 'sbm':
        G = nx.stochastic_block_model([200, 50], [[0.06, 0.005], [0.005, .2]])
        comm1 = np.arange(200, dtype=int)
        comm2 = np.arange(50, dtype=int)+200
    elif graphName == 'as':
        with open(dataDir+'/as19991211.pkl', 'rb') as f:
            G = pkl.load(f)
        comm1 = [7, 44, 53, 57, 59, 62, 63, 66, 67, 125, 129, 165, 166, 211, \
                 212, 253, 281, 306, 307, 308, 309, 357, 376, 377, 378, 382, \
                 383, 385, 386, 387, 390, 397, 402, 403, 404, 405, 446, 447, \
                 453, 475, 476, 477, 513, 541, 542, 543, 544, 545, 767, 768, \
                 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 781, \
                 782, 783, 784, 785, 786, 787, 788, 789, 795, 808, 827, 828, \
                 829, 868, 911, 912, 913, 914, 916, 917, 918, 919, 920, 924, \
                 925, 926, 927, 928, 929, 930, 978, 993, 994, 995, 996, 997, \
                 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1009, 1015, \
                 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1029, 1031, 1048, \
                 1099, 1118, 1141, 1142, 1143, 1144, 1145, 1164, 1165, 1166, \
                 1167, 1169, 1171, 1172, 1173, 1176, 1177, 1178, 1179, 1180, \
                 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, \
                 1191, 1192, 1193, 1194, 1195, 1262, 1263, 1264, 1265, 1266, \
                 1267, 1268, 1269, 1283, 1302, 1303, 1304, 1305, 1306, 1307, \
                 1308, 1309, 1311, 1312, 1314, 1315, 1316, 1317, 1318, 1319, \
                 1321, 1322, 1323, 1328, 1378, 1379, 1381, 1382, 1383, 1429, \
                 1448, 1450, 1451, 1452, 1454, 1456]
        comm2 = [96, 102, 104, 106, 122, 167, 168, 169, 180, 181, 206, 219, \
                 220, 221, 237, 238, 239, 240, 410, 491, 529, 707, 708, 711, \
                 885, 894, 897, 1007, 1038, 1110, 1115, 1116, 1203, 1246, \
                 1258, 1274, 1349, 1350, 1354, 1355, 1356, 1357, 1358, 1359, \
                 1364, 1384, 1385, 1386, 1387, 1446, 1447, 1457, 1458, 1459, \
                 1460, 1465]
    elif graphName == 'airport':
        with open(dataDir+'/airport.pkl', 'rb') as f:
            G = pkl.load(f)
    elif graphName == 'ht':
        with open(dataDir+'/ht09.pkl', 'rb') as f:
            G = pkl.load(f)
    elif graphName == 'metro':
        with open(dataDir+'/metro.pkl', 'rb') as f:
            G = pkl.load(f)
    else:
        raise Exception('bad graph name')
    
    filename = resultDir+'/defense_'+graphName+'_trial'+str(trialInd)+'_method'+method+'_'+str(num_paths)+'paths_'+terminals+'Terminals_cost'+str(success_factor)+'_budgetFactor'+str(budget_factor)+'.pkl'

    if graphName not in ['airport', 'ht', 'metro']:
        for e in G.edges:
            G.edges[e]['cost'] = 1
            G.edges[e]['weight'] = 1+rand.poisson(20)
    else:
        for e in G.edges:
            G.edges[e]['cost'] = 1

    lcc = list(max(nx.connected_components(G), key=len))
    G = nx.subgraph(G, lcc)

    pathDict = {}
    if graphName == 'test':
        lcc = list(max(nx.connected_components(G), key=len))
        pathDict = {}
        for ii in range(3):
            st = rand.choice(lcc, size=2, replace=False)
            s = st[0]
            t = st[1]
            ctr = 0
            for p in nx.shortest_simple_paths(G, s, t, 'weight'):
                ctr += 1
                if ctr >= 10:
                    break
            while (tuple(p) in pathDict) or (ctr < 5+2*ii):
                st = rand.choice(lcc, size=2, replace=False)
                s = st[0]
                t = st[1]
                ctr = 0
                for p in nx.shortest_simple_paths(G, s, t, 'weight'):
                    ctr += 1
                    if ctr >= 10:
                        break
            pathDict[tuple(p)] = (1/3)
    elif terminals=='same':
        st = rand.choice(lcc, size=2, replace=False)
        s = st[0]
        t = st[1]
        ctr = 0
        temp_paths = []
        for p in nx.shortest_simple_paths(G, s, t, 'weight'):
            ctr += 1
            temp_paths.append(p)
            if ctr >= 5+2*(num_paths-1):
                break
        while ctr < 5+2*(num_paths-1):
            st = rand.choice(lcc, size=2, replace=False)
            s = st[0]
            t = st[1]
            ctr = 0
            temp_paths = []
            for p in nx.shortest_simple_paths(G, s, t, 'weight'):
                ctr += 1
                temp_paths.append(p)
                if ctr >= 5+2*(num_paths-1):
                    break
        for ii in range(num_paths):
            pathDict[tuple(temp_paths[5+2*ii-1])] = 1/num_paths
    elif terminals=='different':
        for ii in range(num_paths):
            st = rand.choice(lcc, size=2, replace=False)
            s = st[0]
            t = st[1]
            ctr = 0
            for p in nx.shortest_simple_paths(G, s, t, 'weight'):
                ctr += 1
                if ctr >= 5+2*ii:
                    break
            if ctr < 5+2*ii:
                print(f'warning: fewer than {5+2*ii} paths considered')

            while (tuple(p) in pathDict) or (ctr < 5+2*ii):
                st = rand.choice(lcc, size=2, replace=False)
                s = st[0]
                t = st[1]
                ctr = 0
                for p in nx.shortest_simple_paths(G, s, t, 'weight'):
                    ctr += 1
                    if ctr >= 5+2*ii:
                        break
            pathDict[tuple(p)] = 1/num_paths
    elif terminals=='community':
        for ii in range(num_paths):
            st = rand.choice(comm1, size=2, replace=False)
            s = st[0]
            t = st[1]
            target = rand.choice(comm2)
            try:
                p = get_path_through_target(G, target, s, t)
            except:
                p = []
            while (not p) or (tuple(p) in pathDict):
                st = rand.choice(comm1, size=2, replace=False)
                s = st[0]
                t = st[1]
                target = rand.choice(comm2)
                try:
                    p = get_path_through_target(G, target, s, t)
                except:
                    p = []
            pathDict[tuple(p)] = 1/num_paths
    
    avg_budget = get_avg_budget(G, pathDict)
    avg_budget *= budget_factor
    
    D = {}
    deg = nx.degree(G)
    pairNorm = 0

    impt_nodes = []
    for p in pathDict:
        for v in p:
            impt_nodes.append(v)
        p1 = nx.shortest_path(G, p[0], p[-1], 'weight')
        for v in p1:
            impt_nodes.append(v)
    impt_nodes = np.unique(impt_nodes)
    scale = len(impt_nodes)*(len(impt_nodes)-1)/2
    scale = (len(G)*(len(G)-1)/2-scale)/scale

    for ii in G.nodes():
        D[ii] = {}
        di = nx.degree(G, ii)
        for jj in G.nodes():
            if ii==jj:
                D[ii][jj] = 0
            elif (ii in impt_nodes) and (jj in impt_nodes):
                D[ii][jj] = scale
            else:
                D[ii][jj] = 1
            pairNorm += D[ii][jj]
    for ii in G.nodes():
        for jj in G.nodes():
                D[ii][jj] /= pairNorm
    
    B = stats.poisson.pmf(np.arange(200), avg_budget)
    eInd = {}
    edges = list(G.edges())
    for i in range(len(edges)):
        eInd[edges[i]] = i
        eInd[(edges[i][1], edges[i][0])] = i
    w = np.zeros((len(edges)))
    for i in range(len(edges)):
        w[i] = G.edges[edges[i]]['weight']
    f0 = .8
    f1 = .1
    lmda = 10
        
    
    for e in G.edges:
        G.edges[e]['true_weight'] = G.edges[e]['weight']
        G.edges[e]['observed_weight'] = G.edges[e]['weight']
    results = []
    path_prob = []
    for p in pathDict:
        temp_result, _1, _2 = attack_and_get_difference(G, list(p))
        results.append(temp_result)
        path_prob.append(pathDict[p])
    p_attack = [np.sum(B[np.arange(len(B)) >= r['cost_removed']]) for r in results]
    p_attack = [p_attack[i]*path_prob[i] for i in range(len(p_attack))]
    attacks = [r['edges_removed'] for r in results]
    Lc, _1, _2 = weightChangeCost(G, D, [[]], [0], 0, f0, f1)
    lmda = Lc*success_factor
    
    if method=='zs':
        results = zero_sum_heuristic(G, D, B, pathDict, f0, f1)
    elif method=='heur':
        results = increment_edge_heuristic(G, D, B, pathDict, f0, f1, lmda, Lc*1.001)

    with open(filename, 'wb') as f:
        pkl.dump((results, Lc), f)


