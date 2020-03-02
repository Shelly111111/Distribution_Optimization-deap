## 环境设置
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from deap import creator, base, tools, algorithms
import random
import json
from copy import deepcopy

params = {
    'font.family': 'serif',
    'figure.figsize': [4.0, 3.0],
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)

train_opt={
    'ngen' : 400,#迭代次数
    'cxpb' : 0.3,#遗传概率
    'mutpb' : 0.7,#变异概率
    'data_path' : './map.json',
    'log_file' : './log',
    'popsize' : 100,#种群大小
    'checkpoint' : False,#检查点
    'max' : 9999999,
    'file_path' : './checkpoint.pkl',
    'dataDict' : {
        'MaxLoad' : 5.0,#车辆最大负载
        'MaxMileage' : 35,#最大巡回里程
        'ServiceTime' : 0#服务时间
    }
}

#-----------------------------------
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
        train_opt['dataDict']['edges']=np.zeros((len(train_opt['dataDict']['Node']),len(train_opt['dataDict']['Node'])),int)
        for edge in list(js['edges']):
            train_opt['dataDict']['edges'][edge['pointId1']][edge['pointId2']]=edge['distance']
            train_opt['dataDict']['edges'][edge['pointId2']][edge['pointId1']]=edge['distance']
create_Data_dict(train_opt['data_path'])
#--------------------------------
## 定义TSP中的基本元素
def genInd(dataDict = train_opt['dataDict']):
    '''生成个体， 对我们的问题来说，困难之处在于车辆数目是不定的'''
    nCustomer = len(dataDict['Node']) - 1 # 顾客数量
    perm = np.random.permutation(nCustomer) + 1 # 生成顾客的随机排列,注意顾客编号为1--n
    pointer = 0 # 迭代指针
    lowPointer = 0 # 指针指向下界
    permSlice = []
    # 当指针不指向序列末尾时
    while pointer < nCustomer -1:
        vehicleLoad = 0
        # 当不超载时，继续装载
        while (vehicleLoad < dataDict['MaxLoad']) and (pointer < nCustomer -1):
            vehicleLoad += dataDict['Demand'][perm[pointer]]
            pointer += 1
        if lowPointer+1 < pointer:
            tempPointer = np.random.randint(lowPointer+1, pointer)
            permSlice.append(perm[lowPointer:tempPointer].tolist())
            lowPointer = tempPointer
            pointer = tempPointer
        else:
            permSlice.append(perm[lowPointer::].tolist())
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
    return train_opt['max'] if train_opt['dataDict']['edges'][pos1][pos2]==0 else train_opt['dataDict']['edges'][pos1][pos2]

#
def loadPenalty(routes):
    '''辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚'''
    penalty = 0
    # 计算每条路径的负载，取max(0, routeLoad - maxLoad)计入惩罚项
    for eachRoute in routes:
        routeLoad = np.sum([train_opt['dataDict']['Demand'][i] for i in eachRoute])
        if max(0, routeLoad - train_opt['dataDict']['MaxLoad'])!=0:
            penalty += train_opt['max']
    return penalty

def calRouteLen(routes,dataDict=train_opt['dataDict']):
    '''辅助函数，返回给定路径的总长度'''
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        paraDistance=0
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            paraDistance += calDist(dataDict['Node'][i], dataDict['Node'][j])
        paraDistance if paraDistance <= dataDict['MaxMileage'] else train_opt['max']
        totalDistance+=paraDistance
    return totalDistance

def evaluate(ind):
    '''评价函数，返回解码后路径的总长度，'''
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    return (totalDistance + loadPenalty(routes)),

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
        routesFit = calRouteLen(breakSubroute) + loadPenalty(breakSubroute)
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
    minDistance = calRouteLen([route]) # 最优路径长度
    for i in range(1,nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
            reversedRouteDist = calRouteLen([reversedRoute])
            # 如果翻转后路径更优，则更新最优解
            if  reversedRouteDist < minDistance:
                minDistance = reversedRouteDist
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

def min(ls):
    '''最小值函数，返回符合约束的最小值'''
    lst=[ _dist for _dist in ls if _dist[0] < train_opt['max']]
    return None if len(lst)==0 else np.min(lst).astype(int)
def avg(ls):
    '''平均值函数，返回符合约束的平均值'''
    lst=[ _dist for _dist in ls if _dist[0] < train_opt['max']]
    return None if len(lst)==0 else np.mean(lst)
def std(ls):
    '''标准差值函数，返回符合约束的标准差值'''
    lst=[ _dist for _dist in ls if _dist[0] < train_opt['max']]
    return None if len(lst)==0 else np.std(lst)
# 用[n,2]的np.array存储城市坐标；每行存储一个城市
def genCity(n, Lb = 100 ,Ub = 999):
    # 生成城市坐标
    # 输入：n -- 需要生成的城市数量
    # 输出: nx2 np array 每行是一个城市的[X,Y]坐标
    np.random.seed(42) # 保证结果的可复现性
    return np.random.randint(low = Lb, high = Ub, size=(n,2))

# 计算并存储城市距离矩阵
def cityDistance(cities):
    # 生成城市距离矩阵 distMat[A,B] = distMat[B,A]表示城市A，B之间距离
    # 输入：cities -- [n,2] np array， 表示城市坐标
    # 输出：nxn np array， 存储城市两两之间的距离
    return distance.cdist(cities, cities, 'euclidean')

def completeRoute(individual):
    # 序列编码时，缺少最后一段回到原点的线段
    return individual + [individual[0]] # 不要用append
    
# 计算给定路线的长度
def routeDistance(route):
    # 输入：
    #      route -- 一条路线，一个sequence
    # 输出：routeDist -- scalar，路线的长度
    if route[0] != route[-1]:
        route = completeRoute(route)
    routeDist = 0
    for i,j in zip(route[0::],route[1::]):
        routeDist += cityDist[i,j] # 这里直接从cityDist变量中取值了，其实并不是很安全的写法，单纯偷懒了
    return (routeDist), # 注意DEAP要求评价函数返回一个元组

# 路径可视化
def plotTour(tour, cities, style = 'bo-'):
    if len(tour)>1000: plt.figure(figsize = (15,10))
    start = tour[0:1]
    for i,j in zip(tour[0::], tour[1::]):
        plt.plot([cities[i,0],cities[j,0]], [cities[i,1],cities[j,1]], style)
    plt.plot(cities[start,0],cities[start,1],'rD')
    plt.axis('scaled')
    plt.axis('off')
    plt.show()
    
#--------------------------------
## 设计GA算法
nCities = 30
cities = genCity(nCities) # 随机生成nCities个城市坐标

# 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 最小化问题
creator.create('Individual', list, fitness=creator.FitnessMin)

# 定义个体编码
toolbox = base.Toolbox()
#toolbox.register('indices', random.sample, range(nCities), nCities) # 创建序列
#toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register('individual', tools.initIterate, creator.Individual, genInd)

# 生成族群
N_POP = train_opt['popsize']
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(N_POP)

# 注册所需工具
cityDist = cityDistance(cities)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', crossover)
toolbox.register('mutate', mutate)
#toolbox.register('evaluate', routeDistance)
#toolbox.register('select', tools.selTournament, tournsize = 2)
#toolbox.register('mate', tools.cxOrdered)
#toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.2)

# 数据记录
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', min)
stats.register('avg', avg)
stats.register('std', std)

# 调用内置的进化算法
resultPop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = train_opt['cxpb'], mutpb = train_opt['mutpb'], 
                                         ngen = train_opt['ngen'], stats = stats, verbose = True)

tour = tools.selBest(resultPop, k=1)[0]
tourDist = tour.fitness
tour = completeRoute(tour)
print('最优路径为:'+str(tour))
print('最优路径距离为：'+str(tourDist))
plotTour(tour, cities)
