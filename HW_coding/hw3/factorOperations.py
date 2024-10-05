# factorOperations.py
# -------------------
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


from typing import List
from bayesNet import Factor
import operator as op
import util
import functools

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factor failed joinFactorsByVariable typecheck: ", factors)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()


def joinFactors(factors: List[Factor]):
    """
    Question 3: Your join implementation 

    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict

    问题 3：您的连接实施

    输入因子是一个因子列表。

    您应计算无条件变量集和有条件变量集。
    变量。

    返回一个包含这些变量的新因子，其概率项
    是输入因子相应行的乘积。

    您可以假设所有输入因子的变量域 Dict
    因子的变量域数据是相同的，因为它们来自同一个贝叶斯网络。

    joinFactors 只允许无条件变量出现在一个输入因子中（因此它们的连接是无条件的）。
    一个输入因子中出现（因此它们的连接定义良好）。

    提示：将赋值描述符作为输入的因子方法
    (的因子方法（如 getProbability 和 setProbability）可以处理
    赋值变量多于因子中变量的赋值 dictionary。

    有用的函数
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factor failed joinFactors typecheck: ", factors)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))


    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    factors_list = list(factors)
    # print(factors)#dict_values([Factor({'D0'}, {'W0'}, {'W0': ['sun', 'rain'], 'D0': ['wet', 'dry']}), Factor({'W0'}, set(), {'W0': ['sun', 'rain'], 'D0': ['wet', 'dry']})])
    conditioned_vars = set()
    unconditioned_vars = set()

    for factor in factors_list:
        con = factor.conditionedVariables()
        uncon = factor.unconditionedVariables()
        conditioned_vars = conditioned_vars.union(con)
        unconditioned_vars = unconditioned_vars.union(uncon)

    # 移除有条件变量和无条件变量之间的重叠部分
    for var in unconditioned_vars:
        if var in conditioned_vars:
            conditioned_vars.remove(var)

    # 创建新的因子，其中包含连接后的有条件变量和无条件变量
    result_factor = Factor(unconditioned_vars, conditioned_vars, factors_list[0].variableDomainsDict())
    # print(factors[0].variableDomainsDict())#{'W0': ['sun', 'rain'], 'D0': ['wet', 'dry']}
    # print(conded_vars)#[]
    # print(unconded_vars)#['D0', 'W0']
    # print(res_factor)#P(D0, W0)
    # 计算新因子中每个赋值的概率项
    for assignment in result_factor.getAllPossibleAssignmentDicts():
        probability = 1
        for factor in factors_list:
            # print(factor)
            probability *= factor.getProbability(assignment)
        # print(assignment)
        result_factor.setProbability(assignment, probability)
    # print(result_factor)
    return result_factor
    "*** END YOUR CODE HERE ***"

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Question 4: Your eliminate implementation 

        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.
        问题 4：您的消除实施

        输入 factor 是单因子。
        输入 eliminationVariable 是要从因子中消除的变量。
        eliminationVariable 必须是因子中的无条件变量。

        您应计算消除该变量后得到的因子的无条件变量集和有条件变量集。
        变量。
        消除变量。

        返回一个新因子，其中所有提及
        消除变量的行与符合其他变量赋值的行相加。
        的行相加。
        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # autograder tracking -- don't remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))


        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # print(eliminationVariable)
        # 计算消除变量后的无条件变量集
        unconditioned_vars = factor.unconditionedVariables()
        unconditioned_vars.remove(eliminationVariable)
        conditioned_vars = factor.conditionedVariables()
        variableDomainsDict=factor.variableDomainsDict()
        # print(factor)
        # 新增加一个factor
        result_factor = Factor(unconditioned_vars, conditioned_vars, variableDomainsDict)
        # print(factor.conditionedVariables())
        # print('factor.getAllPossibleAssignmentDicts()')
        # print(factor.getAllPossibleAssignmentDicts())
        # print(res_factor)
        # 遍历每一条assignment，去除掉eliminationVariable，在原来的未完整的assignment上加
        for assignment in factor.getAllPossibleAssignmentDicts():
            probability = factor.getProbability(assignment)
            # print(assignment)
            assignment.pop(eliminationVariable)  # remove eliminated
            # print(assignment)
            # print(result_factor.getProbability(assignment))
            probability += result_factor.getProbability(assignment)
            result_factor.setProbability(assignment, probability)
        # print(result_factor)
        return result_factor
        "*** END YOUR CODE HERE ***"

    return eliminate

eliminate = eliminateWithCallTracking()


def normalize(factor: Factor):
    """
    Question 5: Your normalize implementation 

    Input factor is a single factor.

    The set of conditioned variables for the normalized factor consists 
    of the input factor's conditioned variables as well as any of the 
    input factor's unconditioned variables with exactly one entry in their 
    domain.  Since there is only one entry in that variable's domain, we 
    can either assume it was assigned as evidence to have only one variable 
    in its domain, or it only had one entry in its domain to begin with.
    This blurs the distinction between evidence assignments and variables 
    with single value domains, but that is alright since we have to assign 
    variables that only have one value in their domain to that single value.

    Return a new factor where the sum of the all the probabilities in the table is 1.
    This should be a new factor, not a modification of this factor in place.

    If the sum of probabilities in the input factor is 0,
    you should return None.

    This is intended to be used at the end of a probabilistic inference query.
    Because of this, all variables that have more than one element in their 
    domain are assumed to be unconditioned.
    There are more general implementations of normalize, but we will only 
    implement this version.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    # typecheck portion
    variableDomainsDict = factor.variableDomainsDict()
    for conditionedVariable in factor.conditionedVariables():
        if len(variableDomainsDict[conditionedVariable]) > 1:
            print("Factor failed normalize typecheck: ", factor)
            raise ValueError("The factor to be normalized must have only one " + \
                            "assignment of the \n" + "conditional variables, " + \
                            "so that total probability will sum to 1\n" + 
                            str(factor))

    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    conditioned_vars = set()
    unconditioned_vars = set()
    # print(variableDomainsDict)
    # print(factor.conditionedVariables())
    # print(factor.unconditionedVariables())
    # 想法是重新放进去
    for var in variableDomainsDict:
        if var in factor.conditionedVariables():
            if len(variableDomainsDict[var]) == 1:#只有一个值，视为条件变量
                conditioned_vars.add(var)
            else:
                unconditioned_vars.add(var)
        if var in factor.unconditionedVariables():
            if len(variableDomainsDict[var]) == 1:
                conditioned_vars.add(var)
            else:
                unconditioned_vars.add(var)

    result_factor = Factor(unconditioned_vars, conditioned_vars, variableDomainsDict)
    # 开始归一化
    normalize_sum = 0
    for assignment in factor.getAllPossibleAssignmentDicts():
        normalize_sum += factor.getProbability(assignment)

    #If the sum of probabilities in the input factor is 0,
    # you should return None.
    if normalize_sum == 0:
        return None
    for assign in result_factor.getAllPossibleAssignmentDicts():
        result_factor.setProbability(assign, factor.getProbability(assign) / normalize_sum)

    return result_factor
    "*** END YOUR CODE HERE ***"

