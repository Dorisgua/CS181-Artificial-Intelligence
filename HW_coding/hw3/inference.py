# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from typing import List, Dict, Tuple
import random
import util
from bayesNet import Factor, BayesNet
from factorOperations import joinFactorsByVariableWithCallTracking, joinFactors
from factorOperations import eliminateWithCallTracking, normalize

def inferenceByEnumeration(bayesNet: BayesNet, queryVariables: List[str], evidenceDict: Dict):
    """
    An inference by enumeration implementation provided as reference.
    This function performs a probabilistic inference query that
    returns the factor:

    P(queryVariables | evidenceDict)

    bayesNet:       The Bayes Net on which we are making a query.
    queryVariables: A list of the variables which are unconditioned in
                    the inference query.
    evidenceDict:   An assignment dict {variable : value} for the
                    variables which are presented as evidence
                    (conditioned) in the inference query.
    作为参考提供的枚举推理实现。
    该函数执行概率推理查询，并
    返回因子：

    P(queryVariables | evidenceDict)

    bayesNet：       我们正在查询的贝叶斯网。
    queryVariables： 查询变量： 推理查询中无条件变量的列表。
                    的变量列表。
    evidenceDict：   一个赋值独占项 {变量：值
                    变量的赋值
                    (条件）的变量的赋值独占项。
    """
    callTrackingList = []
    joinFactorsByVariable = joinFactorsByVariableWithCallTracking(callTrackingList)
    eliminate = eliminateWithCallTracking(callTrackingList)

    # initialize return variables and the variables to eliminate# 初始化返回变量和要消除的变量
    evidenceVariablesSet = set(evidenceDict.keys())
    queryVariablesSet = set(queryVariables)
    # 计算要消除的变量集合（贝叶斯网络变量集合减去证据变量和查询变量）
    eliminationVariables = (bayesNet.variablesSet() - evidenceVariablesSet) - queryVariablesSet
    # print(eliminationVariables)

    # grab all factors where we know the evidence variables (to reduce the size of the tables)
    # 获取所有包含给定证据变量的因子（用于减少表的大小）
    currentFactorsList = bayesNet.getAllCPTsWithEvidence(evidenceDict)

    # join all factors by variable# 根据变量逐步连接所有因子
    for joinVariable in bayesNet.variablesSet():
        currentFactorsList, joinedFactor = joinFactorsByVariable(currentFactorsList, joinVariable)
        currentFactorsList.append(joinedFactor)

    # currentFactorsList should contain the connected components of the graph now as factors, must join the connected components
    # currentFactorsList 应该包含图的连接组件作为因子，需要连接这些连接组件
    fullJoint = joinFactors(currentFactorsList)

    # marginalize all variables that aren't query or evidence
    incrementallyMarginalizedJoint = fullJoint
    for eliminationVariable in eliminationVariables:
        incrementallyMarginalizedJoint = eliminate(incrementallyMarginalizedJoint, eliminationVariable)

    fullJointOverQueryAndEvidence = incrementallyMarginalizedJoint

    # normalize so that the probability sums to one
    # the input factor contains only the query variables and the evidence variables, 
    # both as unconditioned variables
    queryConditionedOnEvidence = normalize(fullJointOverQueryAndEvidence)
    # now the factor is conditioned on the evidence variables

    # the order is join on all variables, then eliminate on all elimination variables
    return queryConditionedOnEvidence

def inferenceByVariableEliminationWithCallTracking(callTrackingList=None):

    def inferenceByVariableElimination(bayesNet: BayesNet, queryVariables: List[str], evidenceDict: Dict, eliminationOrder: List[str]):
        """
        Question 6: Your inference by variable elimination implementation

        This function should perform a probabilistic inference query that
        returns the factor:

        P(queryVariables | evidenceDict)

        It should perform inference by interleaving joining on a variable
        and eliminating that variable, in the order of variables according
        to eliminationOrder.  See inferenceByEnumeration for an example on
        how to use these functions.

        You need to use joinFactorsByVariable to join all of the factors 
        that contain a variable in order for the autograder to 
        recognize that you performed the correct interleaving of 
        joins and eliminates.

        If a factor that you are about to eliminate a variable from has 
        only one unconditioned variable, you should not eliminate it 
        and instead just discard the factor.  This is since the 
        result of the eliminate would be 1 (you marginalize 
        all of the unconditioned variables), but it is not a 
        valid factor.  So this simplifies using the result of eliminate.

        The sum of the probabilities should sum to one (so that it is a true 
        conditional probability, conditioned on the evidence).

        bayesNet:         The Bayes Net on which we are making a query.
        queryVariables:   A list of the variables which are unconditioned
                          in the inference query.
        evidenceDict:     An assignment dict {variable : value} for the
                          variables which are presented as evidence
                          (conditioned) in the inference query. 
        eliminationOrder: The order to eliminate the variables in.

        Hint: BayesNet.getAllCPTsWithEvidence will return all the Conditional 
        Probability Tables even if an empty dict (or None) is passed in for 
        evidenceDict. In this case it will not specialize any variable domains 
        in the CPTs.
         问题 6：您的变量消除推理实现

        此函数应执行概率推理查询，以
        返回因子：

        P(queryVariables | evidenceDict)

        它应通过在一个变量上交错加入
        并消除该变量的推理。
        消除顺序来执行推理。 有关如何使用这些函数的示例，请参见 inferenceByEnumeration。
        如何使用这些函数的示例。

        您需要使用 joinFactorsByVariable 将包含变量的所有因子
        才能使自动分析仪识别您执行了正确的交错计算。
        识别出您执行了正确的
        连接和消除。

        如果您要消除的因子中只有一个无条件变量
        只有一个无条件变量，则不应删除该变量，而应直接舍弃该因数。
        而是直接舍弃因子。 这是因为
        剔除的结果将是 1（将所有无条件变量边际化
        所有无条件变量），但它不是一个有效因子。
        因数。 因此，这就简化了消除结果的使用。

        概率的总和应为 1（因此它是一个真正的条件概率，以证据为条件）。
        条件概率）。

        贝叶斯网         我们正在查询的贝叶斯网络。
        queryVariables: 查询变量：   推理查询中无条件变量的列表。
        evidenceDict：     赋值指针
        消除顺序： 消除变量的顺序。

        提示：BayesNet.getAllCPTsWithEvidence 将返回所有的条件
        概率表，即使为 evidenceDict。在这种情况下，它不会专门化 CPT 中的任何变量域。

        Useful functions:
        BayesNet.getAllCPTsWithEvidence
        normalize
        eliminate
        joinFactorsByVariable
        joinFactors
        """

        # this is for autograding -- don't modify
        joinFactorsByVariable = joinFactorsByVariableWithCallTracking(callTrackingList)
        eliminate = eliminateWithCallTracking(callTrackingList)
        if eliminationOrder is None: # set an arbitrary elimination order if None given
            eliminationVariables = bayesNet.variablesSet() - set(queryVariables) -\
                                   set(evidenceDict.keys())
            eliminationOrder = sorted(list(eliminationVariables))

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        currentFactorsList = bayesNet.getAllCPTsWithEvidence(evidenceDict)
        # print("currentFactorsList")
        # print(currentFactorsList)
        for joinVar in eliminationOrder:
            currentFactorsList, joinedFactor = joinFactorsByVariable(currentFactorsList, joinVar)
            # 如果因子有多个未条件化变量（未消除），则消除其中一个变量并将其添加到列表中
            if len(joinedFactor.unconditionedVariables()) > 1:
                eliminated = eliminate(joinedFactor, joinVar)
                currentFactorsList.append(eliminated)
            # 对已清除变量不进行消除操作
        return normalize(joinFactors(currentFactorsList))
        "*** END YOUR CODE HERE ***"


    return inferenceByVariableElimination

inferenceByVariableElimination = inferenceByVariableEliminationWithCallTracking()

def sampleFromFactorRandomSource(randomSource=None):
    if randomSource is None:
        randomSource = random.Random()

    def sampleFromFactor(factor, conditionedAssignments=None):
        """
        Sample an assignment for unconditioned variables in factor with
        probability equal to the probability in the row of factor
        corresponding to that assignment.

        factor:                 The factor to sample from.
        conditionedAssignments: A dict of assignments for all conditioned
                                variables in the factor.  Can only be None
                                if there are no conditioned variables in
                                factor, otherwise must be nonzero.

        Useful for inferenceByLikelihoodWeightingSampling

        Returns an assignmentDict that contains the conditionedAssignments but 
        also a random assignment of the unconditioned variables given their 
        probability.
        """
        if conditionedAssignments is None and len(factor.conditionedVariables()) > 0:
            raise ValueError("Conditioned assignments must be provided since \n" +
                            "this factor has conditionedVariables: " + "\n" +
                            str(factor.conditionedVariables()))

        elif conditionedAssignments is not None:
            conditionedVariables = set([var for var in conditionedAssignments.keys()])

            if not conditionedVariables.issuperset(set(factor.conditionedVariables())):
                raise ValueError("Factor's conditioned variables need to be a subset of the \n"
                                    + "conditioned assignments passed in. \n" + \
                                "conditionedVariables: " + str(conditionedVariables) + "\n" +
                                "factor.conditionedVariables: " + str(set(factor.conditionedVariables())))

            # Reduce the domains of the variables that have been
            # conditioned upon for this factor 
            newVariableDomainsDict = factor.variableDomainsDict()
            for (var, assignment) in conditionedAssignments.items():
                newVariableDomainsDict[var] = [assignment]

            # Get the (hopefully) smaller conditional probability table
            # for this variable 
            CPT = factor.specializeVariableDomains(newVariableDomainsDict)
        else:
            CPT = factor
        
        # Get the probability of each row of the table (along with the
        # assignmentDict that it corresponds to)
        assignmentDicts = sorted([assignmentDict for assignmentDict in CPT.getAllPossibleAssignmentDicts()])
        assignmentDictProbabilities = [CPT.getProbability(assignmentDict) for assignmentDict in assignmentDicts]

        # calculate total probability in the factor and index each row by the 
        # cumulative sum of probability up to and including that row
        currentProbability = 0.0
        probabilityRange = []
        for i in range(len(assignmentDicts)):
            currentProbability += assignmentDictProbabilities[i]
            probabilityRange.append(currentProbability)

        totalProbability = probabilityRange[-1]

        # sample an assignment with probability equal to the probability in the row 
        # for that assignment in the factor
        pick = randomSource.uniform(0.0, totalProbability)
        for i in range(len(assignmentDicts)):
            if pick <= probabilityRange[i]:
                return assignmentDicts[i]

    return sampleFromFactor

sampleFromFactor = sampleFromFactorRandomSource()
