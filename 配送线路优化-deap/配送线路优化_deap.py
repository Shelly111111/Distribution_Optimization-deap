## 环境设定
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import random
import matplotlib
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from PyQt5 import QtCore, QtWidgets, QtGui
#from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pylab import mpl
from matplotlib.figure import Figure
from tkinter import *
import pickle
import json
import threading

params = {
    'font.family': 'serif',
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)
train_opt={
    'ngen' : 400,#迭代次数
    'cxpb' : 0.8,#遗传概率
    'mutpb' : 0.1,#变异概率
    'data_path' : './graph.json',
    'log_file' : './log',
    'popsize' : 200,#种群大小
    'checkpoint' : False,#检查点
    'max' : 9999999,
    'file_path' : './checkpoint.pkl',
    'dataDict' : {
        'Vehicle_type_num' : 2,
        0 : {
            'MaxLoad' : 30.0,#车辆最大负载
            'MaxMileage' : 400,#最大巡回里程
            'Num' : 18
            },
        1 : {
            'MaxLoad' : 12.0,
            'MaxMileage' : 400,#最大巡回里程
            'Num' : 10
            },
        'ServiceTime' : 5,#服务时间，单位分钟
        'MaxServiceTime' : 480,#单位分钟
        'speed' : 60#单位千米/小时
    }
}
from copy import deepcopy


#-----------------------------------
def dijkstra(s):
    V = len(train_opt['dataDict']['Node'])
    # 标记数组：used[v]值为False说明改顶点还没有访问过，在S中，否则在U中！
    used = [False for _ in range(V)]
    distance = [train_opt['max'] for _ in range(V)]
    distance[s] = 0
    while True:
        # v在这里相当于是一个哨兵，对包含起点s做统一处理！
        v = -1
        # 从未使用过的顶点中选择一个距离最小的顶点
        for u in range(V):
            if not used[u] and (v == -1 or distance[u] < distance[v]):
                v = u
        if v == -1:
            # 说明所有顶点都维护到S中了！
            break

        # 将选定的顶点加入到S中, 同时进行距离更新
        used[v] = True
        # 更新U中各个顶点到起点s的距离。之所以更新U中顶点的距离，是由于上一步中确定了k是求出最短路径的顶点，从而可以利用k来更新其它顶点的距离；例如，(s,v)的距离可能大于(s,k)+(k,v)的距离。
        for u in range(V):
            distance[u] = min(distance[u], distance[v] + train_opt['dataDict']['edges'][v][u])
    for i in range(V):
        train_opt['dataDict']['edges'][s][i]=distance[i]

## 个体编码
# 用字典存储所有参数 -- 配送中心坐标、顾客坐标、顾客需求、到达时间窗口、服务时间、车型载重量
def create_Data_dict(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        js=json.load(f)
        train_opt['dataDict']['Node']=[]
        train_opt['dataDict']['Demand']=[]
        for vex in list(js['vertexes']):
            train_opt['dataDict']['Node'].append(vex['id'])
            train_opt['dataDict']['Demand'].append(vex['need'])
        _l=len(train_opt['dataDict']['Node'])
        train_opt['dataDict']['edges']=np.ones((_l,_l),int) * train_opt['max']
        for edge in list(js['edges']):
            train_opt['dataDict']['edges'][edge['pointId1']][edge['pointId2']]=edge['distance']
            train_opt['dataDict']['edges'][edge['pointId2']][edge['pointId1']]=edge['distance']
    for i in range(_l):
        dijkstra(i)

def genInd(dataDict = train_opt['dataDict']):
    '''生成个体， 对我们的问题来说，困难之处在于车辆数目是不定的'''
    nCustomer = len(dataDict['Node']) - 1 # 顾客数量
    perm = np.random.permutation(nCustomer) + 1 # 生成顾客的随机排列,注意顾客编号为1--n
    pointer = 0 # 迭代指针
    lowPointer = 0 # 指针指向下界
    vehicle_num = 0
    vehicle_type = 0
    permSlice = []
    # 当指针不指向序列末尾时
    while pointer < nCustomer -1:
        vehicleLoad = 0
        # 当不超载时，继续装载
        while vehicleLoad < dataDict[vehicle_type]['MaxLoad'] and (pointer < nCustomer -1):
            vehicleLoad += dataDict['Demand'][perm[pointer]]
            pointer += 1
        vehicle_num += 1
        if lowPointer + 1 < pointer:#在负载上限上只取部分，生成一条路径
            tempPointer = np.random.randint(lowPointer+1, pointer)
            permSlice.append(perm[lowPointer:tempPointer].tolist())
            lowPointer = tempPointer
            pointer = tempPointer
        else:
            pointer = nCustomer
            permSlice.append(perm[lowPointer:pointer].tolist())
            break
        if vehicle_num >= dataDict[vehicle_type]['Num']:
            vehicle_num = 0
            vehicle_type += 1
        if vehicle_type >= dataDict['Vehicle_type_num']:
            permSlice.append(perm[pointer:nCustomer -1].tolist())
            break
    # 将路线片段合并为染色体
    ind = [0]
    for eachRoute in permSlice:
        ind = ind + eachRoute + [0]
    return ind
#-----------------------------------
## 评价函数
# 染色体解码
## 评价函数
# 染色体解码
def decodeInd(ind):
    '''从染色体解码回路线片段，每条路径都是以0为开头与结尾'''
    indCopy = np.array(deepcopy(ind)) # 复制ind，防止直接对染色体进行改动
    idxList = list(range(len(indCopy)))
    zeroIdx = np.asarray(idxList)[indCopy == 0]
    routes = []
    for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
        routes.append(ind[i:j]+[0])
    return routes

def calDist(pos1, pos2):
    return train_opt['dataDict']['edges'][pos1][pos2]

def loadPenalty(routes):
    '''辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚'''
    vehicle_num = 0
    vehicle_type = 0
    # 计算每条路径的负载
    for eachRoute in routes:
        if vehicle_type >= train_opt['dataDict']['Vehicle_type_num']:
            return train_opt['max']
        routeLoad = np.sum([train_opt['dataDict']['Demand'][i] for i in eachRoute])
        if routeLoad > train_opt['dataDict'][vehicle_type]['MaxLoad']:
            return train_opt['max']
        vehicle_num += 1
        if vehicle_num >= train_opt['dataDict'][vehicle_type]['Num']:
            vehicle_num = 0
            vehicle_type += 1
    return 0

def calRouteLen(routes,dataDict=train_opt['dataDict']):
    '''辅助函数，返回给定路径的总长度，总服务时长'''
    totalDistance = 0 # 记录各条路线的总长度
    totalServiceTime = 0
    paraServiceTime = 0
    vehicle_num = 0
    vehicle_type = 0
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        if vehicle_type >= dataDict['Vehicle_type_num']:
            return train_opt['max'],train_opt['max']
        paraDistance = 0
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            paraDistance += calDist(dataDict['Node'][i], dataDict['Node'][j])
        paraServiceTime = (len(eachRoute)-2) * dataDict['ServiceTime'] + int(paraDistance / dataDict['speed'] * 60)
        if paraDistance > dataDict[vehicle_type]['MaxMileage'] or paraServiceTime > dataDict['MaxServiceTime']:
            return train_opt['max'],train_opt['max']
        vehicle_num += 1
        if vehicle_num >= dataDict[vehicle_type]['Num']:
            vehicle_num = 0
            vehicle_type += 1
        totalDistance += paraDistance
        totalServiceTime += paraServiceTime
    return totalDistance , round(totalServiceTime / 60 , 2)

def evaluate(ind):
    '''评价函数，返回解码后路径的总长度，'''
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance , totalServiceTime = calRouteLen(routes)
    return (totalDistance + totalServiceTime + loadPenalty(routes)),
    
def _min(ls):
    '''最小值函数，返回符合约束的最小值'''
    lst=[ _dist for _dist in ls if _dist[0] < train_opt['max']]
    return None if len(lst)==0 else np.min(lst)
def _avg(ls):
    '''平均值函数，返回符合约束的平均值'''
    lst=[ _dist for _dist in ls if _dist[0] < train_opt['max']]
    return None if len(lst)==0 else np.mean(lst)
def _std(ls):
    '''标准差值函数，返回符合约束的标准差值'''
    lst=[ _dist for _dist in ls if _dist[0] < train_opt['max']]
    return None if len(lst)==0 else np.std(lst)
#-----------------------------------
## 交叉操作
def genChild(ind1, ind2, nTrail=5):
    '''参考《基于电动汽车的带时间窗的路径优化问题研究》中给出的交叉操作，生成一个子代'''
    # 在ind1中随机选择一段子路径subroute1，将其前置
    routes1 = decodeInd(ind1) # 将ind1解码成路径
    numSubroute1 = len(routes1) # 子路径数量
    subroute1 = routes1[np.random.randint(0, numSubroute1)]
    # 将subroute1中没有出现的顾客按照其在ind2中的顺序排列成一个序列
    unvisited = set(ind1) - set(subroute1) # 在subroute1中没有出现访问的顾客
    unvisitedPerm = [digit for digit in ind2 if digit in unvisited] # 按照在ind2中的顺序排列
    # 多次重复随机打断，选取适应度最好的个体
    bestRoute = None # 容器
    bestFit = np.inf
    for _ in range(nTrail):
        # 将该序列随机打断为numSubroute1-1条子路径
        breakPos = [0]+random.sample(range(1,len(unvisitedPerm)),numSubroute1-2) # 产生numSubroute1-2个断点
        breakPos.sort()
        breakSubroute = []
        for i,j in zip(breakPos[0::], breakPos[1::]):
            breakSubroute.append([0]+unvisitedPerm[i:j]+[0])
        breakSubroute.append([0]+unvisitedPerm[j:]+[0])
        # 更新适应度最佳的打断方式
        # 将先前取出的subroute1添加入打断结果，得到完整的配送方案
        breakSubroute.append(subroute1)
        # 评价生成的子路径
        totalDistance,totalServiceTime = calRouteLen(breakSubroute)
        routesFit = totalDistance + totalServiceTime + loadPenalty(breakSubroute)
        if routesFit < bestFit:
            bestRoute = breakSubroute
            bestFit = routesFit
    # 将得到的适应度最佳路径bestRoute合并为一个染色体
    child = []
    for eachRoute in bestRoute:
        child += eachRoute[:-1]
    return child+[0]

def crossover(ind1, ind2):
    '''交叉操作'''
    ind1[:], ind2[:] = genChild(ind1, ind2), genChild(ind2, ind1)
    return ind1, ind2

#-----------------------------------
## 突变操作
def opt(route,dataDict=train_opt['dataDict'], k=2):
    # 用2-opt算法优化路径
    # 输入：
    # route -- sequence，记录路径
    # 输出： 优化后的路径optimizedRoute及其路径长度
    nCities = len(route) # 城市数
    optimizedRoute = route # 最优路径
    minDistance , minServiceTime = calRouteLen([route]) # 最优路径长度
    for i in range(1,nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
            reversedRouteDist , reversedServiceTime = calRouteLen([reversedRoute])
            # 如果翻转后路径更优，则更新最优解
            if  reversedRouteDist < minDistance:
                minDistance = reversedRouteDist
                minServiceTime = reversedServiceTime
                optimizedRoute = reversedRoute
    return optimizedRoute

def mutate(ind):
    '''用2-opt算法对各条子路径进行局部优化'''
    routes = decodeInd(ind)
    optimizedAssembly = []
    for eachRoute in routes:
        optimizedRoute = opt(eachRoute)
        optimizedAssembly.append(optimizedRoute)
    # 将路径重新组装为染色体
    child = []
    for eachRoute in optimizedAssembly:
        child += eachRoute[:-1]
    ind[:] = child+[0]
    return ind,

## 遗传算法主程序
## 遗传算法主程序

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring

def eaMuPlusLambda(gui, start_gen, population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])#打印的头部

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=start_gen, nevals=len(invalid_ind), **record)
    log = open(train_opt['log_file'],'w')
    if verbose:
        gui.f_plot.clear()
        minFit = logbook.select('min')
        avgFit = logbook.select('avg')
        gui.f_plot.plot(minFit, 'b-', label='Minimum Fitness')
        gui.f_plot.plot(avgFit, 'r-', label='Average Fitness')
        gui.canvs.draw()
        stream=logbook.stream
        log.write(stream)
        print(stream)

    # Begin the generational process
    for gen in range(start_gen+1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            minFit = logbook.select('min')
            avgFit = logbook.select('avg')
            if gen % 1==0:
                gui.f_plot.clear()
                gui.f_plot.plot(minFit, 'b-', label='Minimum Fitness')
                gui.f_plot.plot(avgFit, 'r-', label='Average Fitness')
                gui.canvs.draw()
            stream=logbook.stream
            log.write(stream)
            print(stream)
        if gen % 20 == 0:
            cp = dict(population=population, generation=gen, halloffame=halloffame,rndstate=random.getstate())
            with open(train_opt['file_path'], "wb") as cp_file:
                pickle.dump(cp, cp_file,protocol = pickle.HIGHEST_PROTOCOL)

    log.close()
    return population, logbook

from pprint import pprint

def calLoad(routes):
    loads = []
    for eachRoute in routes:
        routeLoad = np.sum([train_opt['dataDict']['Demand'][i] for i in eachRoute])
        loads.append(round(routeLoad,2))
    return loads


def Genetic(gui):
    #-----------------------------------
    ## 问题定义
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 最小化问题
    # 给个体一个routes属性用来记录其表示的路线
    creator.create('Individual', list, fitness=creator.FitnessMin)

    #-----------------------------------
    ## 注册遗传算法操作
    toolbox = base.Toolbox()
    toolbox.register('individual', tools.initIterate, creator.Individual, genInd)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mate', crossover)
    toolbox.register('mutate', mutate)

    ## 生成初始族群
    toolbox.popSize = train_opt['popsize']
    if train_opt['checkpoint']:
        with open(train_opt['file_path'], "rb") as cp_file:
            cp = pickle.load(cp_file)
        pop = cp["population"]
        start_gen = cp["generation"]
        hallOfFame = cp["halloffame"]
        random.setstate(cp["rndstate"])
    else:
        pop = toolbox.population(toolbox.popSize)
        start_gen = 0
        hallOfFame = tools.HallOfFame(maxsize=1)

    ## 记录迭代数据
    stats=tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('min', _min)
    stats.register('avg', _avg)
    stats.register('std', _std)

    ## 遗传算法参数
    toolbox.ngen = train_opt['ngen']
    toolbox.cxpb = train_opt['cxpb']
    toolbox.mutpb = train_opt['mutpb']

    print("-----------------------------")

    pop,logbook=eaMuPlusLambda(gui, start_gen, pop, toolbox, mu=toolbox.popSize, 
                   lambda_=toolbox.popSize,cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
                   ngen=toolbox.ngen ,stats=stats, halloffame=hallOfFame, verbose=True)



    #pop,logbook=algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.popSize, 
    #                   lambda_=toolbox.popSize,cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
    #                   ngen=toolbox.ngen ,stats=stats, halloffame=hallOfFame, verbose=True)
    print('-----------------------------')

    bestInd = hallOfFame.items[0]
    distributionPlan = decodeInd(bestInd)
    bestFit = bestInd.fitness.values
    print('最佳运输计划为：')
    pprint(distributionPlan)
    print('最佳适应距离为：')
    print(bestFit)
    print('最短运输距离、服务时长为：')
    print(calRouteLen(distributionPlan))
    print('各辆车上负载为：')
    print(calLoad(distributionPlan))

#GUI窗口类
class Control():
    #定义GUI界面
    def __init__(self, master, fuc):
        self.parent = master
        self.parent.title("配送线路优化")
        frame = Frame(master)
        self.f = Figure(figsize=(5, 4))
        self.f_plot = self.f.add_subplot(111)#111表示1行1列第1个
        self.Btn=Button(frame, text='最小损失', command=fuc).pack()
        self.canvs = FigureCanvasTkAgg(self.f, self.parent)
        frame.pack()

#具体功能类
class ThreadClient():
    def __init__(self, master):
        self.master = master 
        self.gui = Control(master, self.starting) #将我们定义的GUI类赋给服务类的属性，将执行的功能函数作为参数传入
    def starting(self):
        self.thread = threading.Thread(target = Genetic(self.gui))
        self.thread.start()


create_Data_dict(train_opt['data_path'])

root = Tk()

tool = ThreadClient(root)

tool.gui.canvs.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


#canvs.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

root.mainloop()

"""
# 画出迭代图
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()
"""