import json
import random
import numpy as np
from copy import deepcopy

class Tabu:
    def __init__(self,tabulen=100,preparelen=20):
        self.tabulen=tabulen
        self.preparelen=preparelen
        #self.city,self.cityids,self.stid=self.loadcity2()  #我直接把他的数据放到代码里了
        self.city,self.edges = self.loadcity()
        
        self.route=self.randomroute()
        self.tabu=[]
        self.prepare=[]
        self.curroute=self.route.copy()
        self.bestcost=self.costroad(self.route)
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
        #cityid=[]
        with open(json_path,'r',encoding='utf-8') as f:
            js=json.load(f)
            for vex in list(js['vertexes']):
                city[vex['id']] = vex['need']
        _l=len(city)
        edges = np.ones((_l,_l),int) * train_opt['max']
        with open(json_path,'r',encoding='utf-8') as f:
            js=json.load(f)
            for edge in list(js['edges']):
                edges[edge['pointId1']][edge['pointId2']]=edge['distance']
                edges[edge['pointId2']][edge['pointId1']]=edge['distance']
        for i in range(_l):
            edges = self.dijkstra(i,edges,city)


        """
        for line in open(f): 
            place,lon,lat = line.strip().split(" ") 
            city[int(place)]=float(lon),float(lat) #导入城市的坐标 
            cityid.append(int(place))
        return city,cityid,stid
        """
        return city,edges

    def decodeInd(self, route):
        '''解码回路线片段，每条路径都是以0为开头与结尾'''
        indCopy = np.array(deepcopy(route)) # 复制ind，防止直接对染色体进行改动
        idxList = list(range(len(indCopy)))
        zeroIdx = np.asarray(idxList)[indCopy == 0]
        routes = []
        for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
            routes.append(route[i:j]+[0])
        return routes

    def costroad(self,road):
        #计算当前路径的长度 与原博客里的函数功能相同
        routes = self.decodeInd(road)
        distance = 0
        for eachRoute in routes:
            for i,j in zip(eachRoute[0::], eachRoute[1::]):
                distance += self.edges[i][j]
        """
        d=-1
        st=0,0
        cur=0,0
        city=self.city
        for v in road:
            if d==-1:
                st=city[v]
                cur=st
                d=0
            else:
                d+=((cur[0]-city[v][0])**2+(cur[1]-city[v][1])**2)**0.5 #计算所求解的距离，这里为了简单，视作二位平面上的点，使用了欧式距离
                cur=city[v]
        d+=((cur[0]-st[0])**2+(cur[1]-st[1])**2)**0.5
        """
        return distance
    def randomroute(self):
        #产生一条随机的路
        """
        stid=self.stid
        rt=list(self.city.keys().copy())
        random.shuffle(rt)
        rt.pop(rt.index(stid))
        rt.insert(0,stid)
        """
        nCustomer = len(self.city) - 1 # 顾客数量
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
            while vehicleLoad < train_opt['dataDict'][vehicle_type]['MaxLoad'] and (pointer < nCustomer -1):
                vehicleLoad += self.city[perm[pointer]]
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
            if vehicle_num >= train_opt['dataDict'][vehicle_type]['Num']:
                vehicle_num = 0
                vehicle_type += 1
            if vehicle_type >= train_opt['dataDict']['Vehicle_type_num']:
                permSlice.append(perm[pointer:nCustomer -1].tolist())
                break
        # 将路线片段合并为染色体
        route = [0]
        for eachRoute in permSlice:
            route = route + eachRoute + [0]
        return route
    def opt(self, route, k=2):
        # 用2-opt算法优化路径
        # 输入：
        # route -- sequence，记录路径
        # 输出： 优化后的路径optimizedRoute及其路径长度
        nCities = len(route) # 城市数
        optimizedRoute = route # 最优路径
        minDistance = self.costroad(route) # 最优路径长度
        for i in range(1,nCities-2):
            for j in range(i+k, nCities):
                if j-i == 1:
                    continue
                reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
                reversedRouteDist = self.costroad(reversedRoute)
                # 如果翻转后路径更优，则更新最优解
                if  reversedRouteDist < minDistance:
                    minDistance = reversedRouteDist
                    optimizedRoute = reversedRoute
        return optimizedRoute
    def step(self):
        #搜索一步路找出当前下应该搜寻的下一条路
        routes=self.curroute
        i=0
        pre_routes = self.opt(routes)
        if int(self.costroad(pre_routes)) not in self.tabu:    #产生不在禁忌表中的路径
            self.prepare.append(pre_routes.copy())
            i += 1
        while i < self.preparelen:     #产生候选路径
            pre_routes = self.randomroute()
            v = 0
            while int(self.costroad(pre_routes)) in self.tabu:
                pre_routes = sele.randomroute()
                print('随机生成：第{0}条路径'.format(v))
                v += 1
            self.prepare.append(pre_routes.copy())
            i+=1
        cal=[]
        for route in self.prepare:
            cal.append(self.costroad(route))
        mincal=min(cal)
        min_route=self.prepare[cal.index(mincal)]     #选出候选路径里最好的一条
        if mincal<self.bestcost:    
            self.bestcost=mincal
            self.bestroute=min_route.copy()      #如果他比最好的还要好，那么记录下来
        self.tabu.append(mincal)#int(mrt))    #这里本来要加 mrt的 ，可是mrt是路径，要对比起来麻烦，这里假设每条路是由长度决定的
                                    #也就是说 每个路径和他的长度是一一对应，这样比对起来速度快点，当然这样可能出问题，更好的有待研究
        self.curroute=min_route   #用候选里最好的做下次搜索的起点
        self.prepare=[]
        if len(self.tabu)>self.tabulen:
            self.tabu.pop(0)

import timeit
train_opt={
    'epoch' : 400,#迭代次数
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
t=Tabu()
#print('ok')
#print(t.city)
#print(t.route)
print(t.bestcost)
print(t.curroute)
for i in range(200):
    t.step()
    if i%2==0:
        print(t.bestcost)
        print(t.bestroute)
        #print(t.curroute)

print('ok')
#print(timeit.timeit(stmt="t.step()", number=1000,globals=globals()))
print('ok')
"""
from matplotlib import pyplot as plt

x=[]
y=[]
print("最优路径长度:",t.bestcost)
for i in t.bestroute:
    x0,y0=t.city[i]
    x.append(x0)
    y.append(y0)
x.append(x[0])
y.append(y[0])
plt.plot(x,y)
plt.scatter(x,y)
plt.show()
"""