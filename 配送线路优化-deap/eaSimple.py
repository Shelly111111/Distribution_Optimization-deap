## 环境设置
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from deap import creator, base, tools, algorithms
import random

params = {
    'font.family': 'serif',
    'figure.figsize': [4.0, 3.0],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)
#--------------------------------
## 定义TSP中的基本元素

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
toolbox.register('indices', random.sample, range(nCities), nCities) # 创建序列
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)

# 生成族群
N_POP = 100
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(N_POP)

# 注册所需工具
cityDist = cityDistance(cities)
toolbox.register('evaluate', routeDistance)
toolbox.register('select', tools.selTournament, tournsize = 2)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.2)

# 数据记录
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)

# 调用内置的进化算法
resultPop, logbook = algorithms.eaSimple(pop, toolbox, cxpb = 0.5, mutpb = 0.2, ngen = 200, stats = stats, verbose = True)

tour = tools.selBest(resultPop, k=1)[0]
tourDist = tour.fitness
tour = completeRoute(tour)
print('最优路径为:'+str(tour))
print('最优路径距离为：'+str(tourDist))
plotTour(tour, cities)
