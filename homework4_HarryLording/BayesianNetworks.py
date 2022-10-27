import numpy as np
import pandas as pd
from functools import reduce

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## factor1, factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors


def joinFactors(factor1, factor2):

    factor1Cols = list(factor1)
    factor1Cols.remove('probs')
    factor2Cols = list(factor2)
    factor2Cols.remove('probs')

    allColumns = factor2Cols + factor1Cols

    commonCols = []
    for element in factor1Cols:
        if element in factor2Cols:
            commonCols.append(element)


    if commonCols:
        newTable = pd.DataFrame()
        newTable = pd.merge(factor1, factor2, how='inner', on=commonCols)
        newTable['probs'] = newTable['probs_x'] * newTable['probs_y']
        newTable =newTable.drop(columns=['probs_x','probs_y'])

        return newTable

    else:
        newTableProbs = []
        factor1Probs = factor1["probs"]
        factor2Probs = factor2["probs"]
        for probability in factor2Probs:
            newProbs = factor1Probs*probability
            newTableProbs.extend(newProbs)

        outcomesList = []

        for col in factor1Cols:
            outcomeRange = []
            r = factor1[col][0]
            for i in range(r + 1):
                outcomeRange.append(i)

            outcomeRange.reverse()
            outcomesList.append(outcomeRange)
        for col in factor2Cols:
            outcomeRange = []
            r = factor2[col][0]
            for i in range(r + 1):
                outcomeRange.append(i)

            outcomeRange.reverse()
            outcomesList.append(outcomeRange)

        return readFactorTable(allColumns, newTableProbs, outcomesList)
    return

## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):

    if hiddenVar not in factorTable.columns:
        return factorTable
    factors = list(factorTable)
    factors.remove('probs')
    factors.remove(hiddenVar)

    if len(factors) == 0:
        return factorTable
    new = factorTable.groupby(factors).sum().reset_index()
    new.drop(columns=hiddenVar, axis=1, inplace=True)
    return new

## Marginalize a list of variables
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    for var in hiddenVar:
        marginalizeTabs = []
        unaffectedTabs = []
        for table in bayesNet:
            if var in table.columns:
                marginalizeTabs.append(table)
            else:
                unaffectedTabs.append(table)

        if len(marginalizeTabs) == 0:
            continue
        newTable = marginalizeTabs[0]
        marginalizeTabs.pop(0)
        while marginalizeTabs:
            newTable = joinFactors(newTable, marginalizeTabs.pop(0))

        newTable = marginalizeFactor(newTable, var)

        unaffectedTabs.append(newTable)
        bayesNet = unaffectedTabs

    return bayesNet

## Update BayesNet for a set of evidence variables
## bayesNet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):

    newBayes = []
    for table in bayesNet:

        table = table.reset_index()
        table.drop(columns='index', inplace=True)
        newBayes.append(table)


    bayesNet = newBayes

    loops = 0
    for var in evidenceVars:
        for table in bayesNet:
            if var in table.columns:
                index = 0
                for value in table[var]:
                    if value != evidenceVals[loops]:
                        table.drop(index, axis=0, inplace=True)
                    index = index + 1
            else:
                continue
        loops = loops + 1
        newBayes = []
        for table in bayesNet:
            table = table.reset_index()
            table.drop(columns='index', inplace=True)
            newBayes.append(table)
        bayesNet = newBayes

    return bayesNet


## Run inference on a Bayesian network
## bayesNet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using
## join and marginalization of the sets of variables.
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    # your code
    bayesNet2 = bayesNet.copy()
    bayesNet2 = evidenceUpdateNet(bayesNet2, evidenceVars, evidenceVals)
    bayesNet2 = marginalizeNetworkVariables(bayesNet2, hiddenVar)
    singleTable = bayesNet2[0]
    bayesNet2.pop(0)
    while bayesNet2:
        singleTable = joinFactors(singleTable, bayesNet2.pop(0))
    totalProbability = np.sum(singleTable['probs'])
    singleTable['probs'] = np.divide(singleTable['probs'],(totalProbability))

    return singleTable
