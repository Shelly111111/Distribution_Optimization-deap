import json
import random
import time
import numpy as np
from copy import deepcopy


train_opt={
    'epoch' : 4,#迭代次数
    'data_path' : './graph.json',
    'log_file' : './log',
    'checkpoint' : False,#检查点
    'max' : 9999999,
    'file_path' : './checkpoint.json',
    'dataDict' : {
        'Vehicle_type_num' : 2,
        0 : {
            'MaxLoad' : 30.0,#车辆最大负载
            'MaxMileage' : 400,#最大巡回里程
            'Num' : 12
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


class Tabu:
    def __init__(self):
        if train_opt['checkpoint']:
            with open(train_opt['file_path'],'r',encoding = 'utf-8') as f_reader:
                checkpoint = json.load(f_reader)
                self.Need = checkpoint['Need']
                global start_epoch
                start_epoch = checkpoint['start_epoch']
                self.city = {}
                for id, need in zip(checkpoint['cityID'], checkpoint['need']):
                    self.city[id] = need
                self.edges = checkpoint['edges']
                self.route = checkpoint['curroute']
                self.bestroute = checkpoint['bestroute']
                self.bestcost = checkpoint['bestcost']
                self.curcost = checkpoint['curcost']
        else:
            self.Need = []
            self.curcost = 0
            self.city,self.edges = self.loadcity()
            self.route=self.randomroute()
            self.bestcost=self.evaluate(self.route)
            self.bestroute=self.route

    def dijkstra(self,s,edges,city):
        V = len(city)
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
                distance[u] = min(distance[u], distance[v] + edges[v][u])
        for i in range(V):
           edges[s][i]=distance[i]
        return edges

    def loadcity(self,json_path="./graph.json"):
        city = {}
        with open(json_path,'r',encoding='utf-8') as f:
            js=json.load(f)
            for vex in list(js['vertexes']):
                city[vex['id']] = vex['need']
                if vex['need'] > 0.0:
                    self.Need.append(vex['id'])
        _l=len(city)
        edges = np.ones((_l,_l),int) * train_opt['max']
        with open(json_path,'r',encoding='utf-8') as f:
            js=json.load(f)
            for edge in list(js['edges']):
                edges[edge['pointId1']][edge['pointId2']]=edge['distance']
                edges[edge['pointId2']][edge['pointId1']]=edge['distance']
        for i in range(_l):
            edges = self.dijkstra(i,edges,city)
        print('结点总数：{1}，需配送结点数目：{0}'.format(len(self.Need),_l))
        return city,edges

    def decodeInd(self, route):
        '''解码回路线片段，每条路径都是以0为开头与结尾'''
        indCopy = np.array(deepcopy(route)) # 复制ind，防止直接对染色体进行改动
        idxList = list(range(len(indCopy)))
        zeroIdx = np.asarray(idxList)[indCopy == 0]
        routes = []
        for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
            routes.append(route[i:j]+[0])
        while [0,0] in routes:
            routes.remove([0,0])
        return routes

    def costroad(self,routes):
        #计算当前路径的长度 与原博客里的函数功能相同
        distance = 0
        vehicle_type = 0
        vehicle_num = 0
        for eachRoute in routes:
            if vehicle_type >= train_opt['dataDict']['Vehicle_type_num']:
                return train_opt['max']
            paraDistance = 0
            for i,j in zip(eachRoute[0::], eachRoute[1::]):
                paraDistance += self.edges[i][j]
            if paraDistance > train_opt['dataDict'][vehicle_type]['MaxMileage']:
                return train_opt['max']
            distance += paraDistance
            vehicle_num += 1
            if vehicle_num >= train_opt['dataDict'][vehicle_type]['Num']:
                vehicle_num = 0
                vehicle_type += 1
        return distance

    def loadPenalty(self,routes):
        '''辅助函数，对不合负载要求的个体进行惩罚'''
        vehicle_num = 0
        vehicle_type = 0
        # 计算每条路径的负载
        for eachRoute in routes:
            if vehicle_type >= train_opt['dataDict']['Vehicle_type_num']:
                return train_opt['max']
            routeLoad = np.sum([self.city[i] for i in eachRoute])
            if routeLoad > train_opt['dataDict'][vehicle_type]['MaxLoad']:
                return train_opt['max']
            vehicle_num += 1
            if vehicle_num >= train_opt['dataDict'][vehicle_type]['Num']:
                vehicle_num = 0
                vehicle_type += 1
        return 0

    def evaluate(self,road):
        '''评价函数，返回解码后路径的总损失'''
        routes = self.decodeInd(road) # 将个体解码为路线
        totalDistance = self.costroad(routes)
        totalPenalty = self.loadPenalty(routes)
        if totalDistance < train_opt['max'] and totalPenalty < train_opt['max']:
            return totalDistance + totalPenalty
        else:
            return train_opt['max']

    def randomroute(self):
        #产生一条随机的路
        nCustomer = len(self.Need) # 顾客数量
        perm = self.Need.copy()
        np.random.shuffle(perm)
        pointer = 0 # 迭代指针
        lowPointer = 0 # 指针指向下界
        perdistance = 0
        vehicle_num = 0
        vehicle_type = 0
        permSlice = []
        # 当指针不指向序列末尾时
        while pointer < nCustomer -1:
            vehicleLoad = 0
            Mileage = 0
            curcity = 0
            # 当不超载时，继续装载
            while vehicleLoad + self.city[perm[pointer]] < train_opt['dataDict'][vehicle_type]['MaxLoad'] and pointer < nCustomer -1 \
                and Mileage + self.edges[curcity][perm[pointer]] + self.edges[perm[pointer]][0] < \
                train_opt['dataDict'][vehicle_type]['MaxMileage']:
                vehicleLoad += self.city[perm[pointer]]
                Mileage += self.edges[curcity][perm[pointer]]
                curcity = perm[pointer]
                pointer += 1
            vehicle_num += 1
            if vehicle_num >= train_opt['dataDict'][vehicle_type]['Num']:
                vehicle_num = 0
                vehicle_type += 1
            if vehicle_type >= train_opt['dataDict']['Vehicle_type_num']:
                permSlice.append(perm[pointer:nCustomer -1])
                break
            else:
                permSlice.append(perm[lowPointer:pointer])
                lowPointer = pointer
        # 将路线片段合并为染色体
        route = [0]
        for eachRoute in permSlice:
            route += eachRoute + [0]
        return route

    def opt(self, route, k=2):
        # 用2-opt算法优化路径
        # 输入：
        # route -- sequence，记录路径
        # 输出： 优化后的路径optimizedRoute及其路径长度
        nCities = len(route) # 城市数
        optimizedRoute = route # 最优路径
        minDistance = self.evaluate(route) # 最优路径长度
        for i in range(1,nCities-2):
            for j in range(i+k, nCities):
                if j-i == 1:
                    continue
                reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
                reversedRouteDist = self.evaluate(reversedRoute)
                # 如果翻转后路径更优，则更新最优解
                if  reversedRouteDist < minDistance:
                    minDistance = reversedRouteDist
                    optimizedRoute = reversedRoute
                    #return optimizedRoute
        return optimizedRoute

    def step(self):
        #搜索一步路找出当前下应该搜寻的下一条路
        routes=self.route
        pre_routes = self.opt(routes)
        pre_cal = self.evaluate(pre_routes)
        self.bestcost = pre_cal
        print('step:{0},curcost:{1},bestcost:{2}'.format(0,pre_cal,self.bestcost))
        v = 1
        while pre_cal != self.curcost:    #产生不在禁忌表中的路径
            self.curcost = pre_cal
            pre_routes = self.opt(pre_routes)
            pre_cal = self.evaluate(pre_routes)
            if pre_cal < self.bestcost:
                self.bestcost = pre_cal
                self.bestroute = pre_routes.copy()      #如果他比最好的还要好，那么记录下来
            print('step:{0},curcost:{1},bestcost:{2}'.format(v,pre_cal,self.bestcost))
            v += 1

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

start_epoch = 0
t=Tabu()
np.random.seed(0)
start = time.time()
for i in range(start_epoch, train_opt['epoch']):
    t.route = t.randomroute()
    t.step()
    with open(train_opt['file_path'], 'w', encoding = 'utf-8') as f_write:
        checkpoint = {}
        checkpoint['start_epoch'] = i
        checkpoint['Need'] = t.Need
        checkpoint['curcost'] = t.curcost
        checkpoint['bestcost'] = t.bestcost
        checkpoint['cityID'] = list(t.city.keys())
        checkpoint['need'] = list(t.city.values())
        checkpoint['edges'] = t.edges
        checkpoint['curroute'] = t.route
        checkpoint['bestroute'] = t.bestroute
        json.dump(checkpoint, f_write, cls = MyEncoder)
print(t.decodeInd(t.bestroute))
print(t.bestcost)
print('耗时：{0}'.format(time.time()-start))
"""
import matplotlib.pyplot as plt

x = range(train_opt['epoch'])
plt.title('Tabu')
plt.plot(x, bestcost, label = 'bestcost')
plt.plot(x, curcost, label = 'curcost')
plt.show()

bestroute = [[0, 26, 178, 129, 155, 139, 56, 97, 121, 25, 38, 58, 48, 0],
[0, 176, 158, 79, 168, 148, 11, 31, 171, 107, 191, 7, 103, 99, 1, 0],
[0, 6, 106, 166, 16, 150, 45, 131, 177, 61, 161, 9, 102, 160, 92, 0],
[0, 12, 138, 17, 183, 37, 72, 90, 167, 70, 62, 74, 20, 111, 41, 0],
[0, 146, 147, 54, 84, 43, 194, 197, 193, 179, 174, 180, 108, 122, 0],
[0, 133, 152, 145, 75, 22, 82, 123, 100, 89, 40, 186, 19, 173, 134, 0],
[0, 124, 14, 98, 15, 49, 73, 188, 67, 3, 13, 33, 165, 198, 93, 182, 199, 0],
[0, 189, 52, 34, 2, 55, 156, 94, 44, 181, 132, 53, 0],
[0, 66, 196, 172, 60, 57, 91, 86, 157, 78, 164, 141, 144, 96, 28, 0],
[0, 21, 76, 64, 10, 110, 85, 77, 187, 119, 117, 36, 170, 136, 0],
[0, 105, 125, 175, 112, 5, 4, 114, 128, 153, 95, 130, 29, 0],
[0, 63, 71, 88, 190, 154, 39, 23, 104, 184, 118, 127, 140, 8, 0],
[0, 185, 81, 135, 101, 47, 0],
[0, 87, 69, 0],
[0, 143, 35, 169, 24, 0]]
bestcost = 2292
"""