import random
import json
import numpy as np
import matplotlib.pyplot as plt

node = 200
weights=[1.3,0,1.0,2,2.5,3.0,3.3,1.7,2.4]
DataDict = {}
DataDict['edges'] = []
DataDict['vertexes'] = []

def calDist(pos1, pos2):
    '''计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
    输入： pos1, pos2 -- (x,y)元组
    输出： 欧几里得距离'''
    return np.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0]) + (pos1[1] - pos2[1])*(pos1[1] - pos2[1]))/50

arr1 = np.arange(0, 1000)
random.shuffle(arr1)
arr1 = arr1[:node]
arr2 = np.arange(0,1000)
random.shuffle(arr2)
arr2 = arr2[:node]
city = []
for i in range(node):
    city.append(tuple([arr1[i],arr2[i]]))
edges = np.zeros((node,node),dtype=int)
for i in range(500):
    _a = random.randint(0, node-1)
    _b = random.randint(0, node-1)
    while _a == _b or edges[_a][_b] != 0:
        _a = random.randint(0, node-1)
        _b = random.randint(0, node-1)
    edges[_a][_b] = int(calDist(city[_a],city[_b]))
    edges[_b][_a] = int(calDist(city[_a],city[_b]))
flag = np.zeros(node, dtype = int)
ls = []

def dfs(start):
    for i in range(node):
        if flag[i] == 0 and edges[start][i] != 0:
            flag[i] = 1
            ls.append(i)
            dfs(i)
flag[0] = 1
ls.append(0)
dfs(0)
for i in range(node):
    if flag[i] == 0:
        _a = random.choice(ls)
        _b = random.choice(ls)
        while _a == _b:
            _a = random.choice(ls)
            _b = random.choice(ls)
        edges[_a][i] = int(calDist(city[_a],city[i]))
        edges[i][_a] = int(calDist(city[_a],city[i]))
        edges[i][_b] = int(calDist(city[_b],city[i]))
        edges[_b][i] = int(calDist(city[_b],city[i]))
        flag[i] = 1
        ls.append(i)
ls.clear()
for i in range(node):
    ls.append(city[i])
with open('data.txt','w') as f:
    f.write(str(ls))
    f.write('\n ----------------------------------- \n')
    for i in range(node):
        for j in range(node):
            f.write('%d,'%(edges[i][j]))
        f.write('\n')
print(ls)
print('---------------------')
print(edges)
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

for i in range(node):
    vex = {}
    vex['id'] = i
    vex['need'] = random.choice(weights)
    for j in range(i,node):
        if edges[i][j] != 0:
            plt.plot([city[i][0],city[j][0]], [city[i][1],city[j][1]],'bo-')
            edge={}
            edge['distance'] = edges[i][j]
            edge['pointId1'] = i
            edge['pointId2'] = j
            edge['note'] = "备注"
            DataDict['edges'].append(edge)
    DataDict['vertexes'].append(vex)
DataDict['vertexes'][0]['need'] = 0.0
with open('graph.json','w', encoding='utf-8') as js:
    json.dump(DataDict,js,cls=MyEncoder)
plt.axis('scaled')
plt.axis('off')
plt.show()