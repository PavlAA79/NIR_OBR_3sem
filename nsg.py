import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
from multiprocessing import Pool


class NSG():
    matr_dist = []
    def __init__(self, data,K,l,m,num_processes=0,is_pool = False, init = 'nndescent'):
        self.data = []
        self.indexes = []
        self.neighs = []
        self.neighs_NSG = []
        self.navigating_node = 0
        for i,d in enumerate(data):
            self.indexes.append(i)
            self.data.append(d)
            self.neighs.append([[],[]])  #исходящие связи и входящие (для NNdescent)
            self.neighs_NSG.append([])
        num_elements =len(self.indexes)
        self.matr_dist_Q = []
        self.matr_dist = []
        for i in range(num_elements):
            self.matr_dist.append([])
            for j in range(num_elements):
                 self.matr_dist[i].append(-1)
        print('init---->')
        delta = 0.003
        if init == 'nndescent':
            self.NNDescent(K,delta) #инициализация
        elif init == 'hyrec':
            self.HyRec(K, m)
        else:
            print('There is no init method whith such name. NNDescent will be used')
            self.NNDescent(K,delta) 
        if is_pool:   #построение NSG
            print('build---->')
            self.manage_NSGbuild_pool(l,m,K,num_processes)
        else:
            print('build---->')
            self.NSGBuild(l,m,K)
        n = self.navigating_node
        print('DFS---->')
        self.DFS(n)  #проверка достижимости

    
    def l2_distance1(self,a,q, q_is_vector=False):
        if q_is_vector:
            if self.matr_dist_Q[a] == -1:
                result =  np.linalg.norm(self.data[a] - q)
                self.matr_dist_Q[a] = result
            else:
                result =self.matr_dist_Q[a]
        else:
            if (self.matr_dist[a][q] == -1) or (self.matr_dist[q][a] == -1):
                result = np.linalg.norm(self.data[a] - self.data[q])
                self.matr_dist[a][q] = result 
                self.matr_dist[q][a] = result
            else:
                result = self.matr_dist[a][q]
        return result    

    def l2_distance(self,a,q,q_is_vector=False):
        if q_is_vector:
            if self.matr_dist_Q[a] == -1:
                square = []
                for a1, q1 in zip(self.data[a], q):  
                    square.append(np.square(a1 - q1))
                sum_square = np.sum(square)
                distance = np.sqrt(sum_square)
                self.matr_dist_Q[a] = distance
            else:
                distance = self.matr_dist_Q[a]
        else:
            if (self.matr_dist[a][q] == -1) or (self.matr_dist[q][a] == -1):
                square = []
                for a1, q1 in zip(self.data[a], self.data[q]):  
                    square.append(np.square(a1 - q1))
                sum_square = np.sum(square)
                distance = np.sqrt(sum_square)
                self.matr_dist[a][q] = distance
                self.matr_dist[q][a] = distance
            else:
                distance = self.matr_dist[a][q]
        return distance
    
       
    def HyRec(self,K,m):
        print('hyrec:')
        for v in self.indexes:
            print('v: ', v)
            indexes_without_v = self.indexes.copy()
            indexes_without_v.remove(v)
            print('indexes_without_v: ', indexes_without_v)
            neighbors = list(np.random.choice(indexes_without_v,K, False))
            print('neighbors: ', neighbors)
            for n in neighbors:
                heapq.heappush(self.neighs[v][0], (self.l2_distance(v,n,False),n))
                print('added to ', v, ' neighs: ', n)
            print('neighs ', v , ' are: ', self.neighs[v][0])
        print('PART TWO')
        for v in self.indexes:
            print('v: ', v)
            candidates = []
            for n in self.neighs[v][0]:
                candidates.append(n)
                neighs_neighs = self.neighs[n[1]][0]
                for nn in neighs_neighs:
                    if nn[1] != v:
                        candidates.append((self.l2_distance(v,nn[1],False),nn[1]))
            indexes_without_v = self.indexes.copy()
            indexes_without_v.remove(v)
            #k = round(len(self.indexes)/5)
            k = m
            rand_neighs = list(np.random.choice(indexes_without_v,k, False))
            for r in rand_neighs:
                candidates.append((self.l2_distance(v,r,False),r))
            candidates = list(set(candidates))
            print('candidates = ', candidates)
            heapq.heapify(candidates)
            self.neighs[v][0]=heapq.nsmallest(K,candidates)
            print('neighs v = ', self.neighs[v][0])
            self.neighs_NSG[v] = self.neighs[v][0]
            print('neighs v = ',self.neighs_NSG[v])
    
    #Мультипроцессное построение NSG
    def manage_NSGbuild_pool(self,l,m,K,num_processes):
        border=round(len(self.indexes)/num_processes)
        first = 0
        end = border
        args = []
        neighs_NSG= self.neighs_NSG[:]
        for i in range(num_processes):
            if i == num_processes-1:
                end = len(self.indexes)
            args.append([l,m,K,first,end,neighs_NSG])
            first = first + border
            end = end + border                
        pool = Pool(processes=num_processes)
        resul = pool.map(self.NSGBuild_pool, args)
        neighs_pool = [[]]*(len(self.indexes))
        self.navigating_node = resul[0][1]
        for r in [x[0] for x in resul]:
            for i,neighs in enumerate(r):
                neighs_pool[i] = neighs_pool[i] + neighs
        for i in range(len(neighs_pool)):
            neighs_pool[i] = list(set(neighs_pool[i]))
            if len(neighs_pool[i]) > m:
                neighs_pool[i]  = heapq.nsmallest(m,neighs_pool[i])
        for v in self.indexes:
            self.neighs_NSG[v] = neighs_pool[v]

    #Построение NSG
    def NSGBuild_pool(self,args):
        l = args[0]
        m = args[1]
        K = args[2]
        first = args[3]
        end = args[4]
        neighs_NSG = args[5]
        c = list(np.mean(self.data,axis=0)) #центроид
        r = np.random.choice(self.indexes)
        hops = 0
        n, hops = self.search_on_graph(r,c,l,K=1,q_is_vector=True) #навигационная вершина
        n = n[0][1]
        self.navigating_node = n
        for v in range(first,end):
            E_temp, hops = self.search_on_graph(n,v,l,K,True,False)
            #E_temp = self.search_on_graph_NSGBuild(n,v,l,K,True)
            if v in E_temp:
                E_temp.remove(v)
            for neigh in neighs_NSG[v]:
                if neigh[1] not in E_temp and neigh[1]!=v:
                    E_temp.append(neigh[1])
            E = []
            for e in E_temp:
                heapq.heappush(E,(self.l2_distance(v,e,False),e))
            E = heapq.nsmallest(len(E),E)
            R = []
            p0 = E[0]
            heapq.heappush(R,p0)
            while len(E)!=0 and len(R)<m:
                p = E[0]
                E.remove(p)
                flag_conflict = False
                for r in R:
                    pv = p[0]
                    pr = self.l2_distance(p[1],r[1],False)
                    vr = self.l2_distance(v,r[1],False)
                    if pv > pr and pv > vr:
                        flag_conflict = True
                        break
                if flag_conflict == False:
                    if (pv,p[1]) not in R:
                        heapq.heappush(R,(pv,p[1]))
            neighs_NSG[v] = R
        return neighs_NSG,n
        #self.DFS(n)  
    
    #Построение NSG
    def NSGBuild(self,l,m,K):
        c = list(np.mean(self.data,axis=0)) #центроид
        r = np.random.choice(self.indexes)
        hops = 0
        n, hops = self.search_on_graph(r,c,l,K=1,q_is_vector=True) #навигационная вершина
        n = n[0][1]
        self.navigating_node = n
        for v in self.indexes:
            E_temp, hops = self.search_on_graph(n,v,l,K,True,False)
            #E_temp = self.search_on_graph_NSGBuild(n,v,l,K,True)
            if v in E_temp:
                E_temp.remove(v)
            for neigh in self.neighs_NSG[v]:
                if neigh[1] not in E_temp and neigh[1]!=v:
                    E_temp.append(neigh[1])
            E = []
            for e in E_temp:
                heapq.heappush(E,(self.l2_distance(v,e,False),e))
            E = heapq.nsmallest(len(E),E)
            R = []
            p0 = E[0]
            heapq.heappush(R,p0)
            while len(E)!=0 and len(R)<m:
                p = E[0]
                E.remove(p)
                flag_conflict = False
                for r in R:
                    pv = p[0]
                    pr = self.l2_distance(p[1],r[1],False)
                    vr = self.l2_distance(v,r[1],False)
                    if pv > pr and pv > vr:
                        flag_conflict = True
                        break
                if flag_conflict == False:
                    if (pv,p[1]) not in R:
                        heapq.heappush(R,(pv,p[1]))
            self.neighs_NSG[v] = R
        #self.DFS(n)
    
    #Инициализация с помощью NNDescent
    def NNDescent(self,K,delta):
        neighs_all = []
        for v in self.indexes:
            neighs_all.append([])
            indexes_without_v = self.indexes.copy()
            indexes_without_v.remove(v)
            neighbors = list(np.random.choice(indexes_without_v,K, False))
            for n in neighbors:
                heapq.heappush(self.neighs[v][0], (self.l2_distance(v,n,False),n))
        while True:
            for v in self.indexes:
                R = self.reverse(v)
                self.neighs[v][1] =  R
            for v in self.indexes: #!!!!!!!!
                neighs_all[v] = list(set(self.neighs[v][0] + self.neighs[v][1]))
            c = 0
            k = 0
            for ver in self.indexes:
                p = 0
                neighs_neighs = []
                for u in  neighs_all[ver]:
                    neighs_neighs = neighs_neighs + neighs_all[u[1]]
                neighs_neighs2 = []
                for n in neighs_neighs:
                    neighs_neighs2.append(n[1])
                neighs_neighs2 = list(set(neighs_neighs2))
                for u2 in neighs_neighs2:
                    if u2 != ver:
                        l = self.l2_distance(ver, u2,False)
                        p = p + self.updateNN(ver, (l,u2),K)
                        k = k + self.updateNN(ver, (l,u2),K)
                if c != 0:
                    c = c + 1
            if k == 0: #if c == 0 
                break
            if k < delta*K*len(self.indexes):
                break
        for i, neiset in enumerate(self.neighs):
            self.neighs_NSG[i] = neiset[0]
        self.neighs = []

    #Обновление списка соседей с индикацией изменения    
    def updateNN(self, ind, neigh,K):
        flag = 0
        if neigh not in self.neighs[ind][0] and len(self.neighs[ind][0])<K:
            heapq.heappush(self.neighs[ind][0],neigh)
            flag = 1
        elif neigh not in self.neighs[ind][0] and len(self.neighs[ind][0])==K:
            temp = heapq.nsmallest(K,self.neighs[ind][0])
            temp2 = self.neighs[ind][0]
            heapq.heappush(temp2,neigh)
            self.neighs[ind][0] = heapq.nsmallest(K,temp2)
            if self.neighs[ind][0]== temp:
                flag = 0
            else:
                flag = 1
        return flag
    
    #Метод отображения параметров объекта NSG
    def show_parameters(self):
        print ('NSG:')
        for ind in self.indexes:
            print(ind, ': ', self.neighs_NSG[ind]) 
    
    #Метод получения входящих связей
    def reverse(self, ind):
        R = []
        for j in self.indexes:
            if ind in self.neighs[j][0][1]:
                heapq.heappush(R, (self.l2_distance(j, ind,False),j))
        return R
    
    #Отображение графа NSG
    def show_graph(self):
        edges = []
        for i, neigh in enumerate(self.neighs_NSG):      
            for n in neigh:
                edges.append((i,n[1]))
        nodes = self.indexes
        G = nx.DiGraph()  # создаём объект графа
        # добавляем информацию в объект графа
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        # рисуем граф и отображаем его
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()

    #Поиск по графу от вершины с индексом и от вектора
    def search_on_graph(self,p,q,l,K,flag_constr=False,q_is_vector=True,path_len = False):
        self.matr_dist_Q = []
        for i in range(len(self.indexes)):
            self.matr_dist_Q.append(-1)
        i = 0
        S = [] #пул кандидатов
        heapq.heappush(S,[self.l2_distance(p,q,q_is_vector),p,False]) #пусть p - индекс
        checked = []
        checked.append(p)
        while i < l:
            for j,cand in enumerate(S):
                if cand[2] == False:
                    i = j
                    break
            S[i][2] = True
            for neigh in self.neighs_NSG[S[i][1]]:
                if neigh[1] not in [x[1] for x in S]:
                    dist = self.l2_distance(neigh[1],q,q_is_vector)
                    heapq.heappush(S,[dist, neigh[1],False])
                    checked.append(neigh[1])
            S_new = heapq.nsmallest(len(S),S)
            if len(S_new) > l:
                S_new = heapq.nsmallest(l,S_new)
        knns = heapq.nsmallest(K,S_new)
        knns = [(x[0],x[1]) for x in knns]
        if flag_constr:
            knns = checked
        if path_len:
            print('Hops: ', len(checked))
        return knns, len(checked)
    
    #Поиск в глубину для проверки достижимости вершин из навигационной
    def DFS(self,n):
        while True:
            visited = []
            for i in range (len(self.indexes)):
                visited.append(False)    
            candidates = []
            #n - индекс навигационной вершины
            candidates.append(n)
            while len(candidates) != 0:
                node = candidates.pop()
                if visited[node] == False:
                    visited[node] = True
                    for neigh in self.neighs_NSG[node]:
                        if visited[neigh[1]] == False and neigh[1] not in candidates:
                            candidates.append(neigh[1])
            declined = []
            hops = 0
            for i,v in enumerate(visited):
                if v == False:
                    declined.append(i)
            if len(declined) == 0:
                break
            for d in declined:
                neigh, hops = self.search_on_graph(n,d,7,2,False,False)
                if neigh[0][1] == d:
                    neigh = neigh[1]
                else:
                    neigh = neigh[0]
                if d not in [x[1] for x in self.neighs_NSG[neigh[1]]]:
                    self.neighs_NSG[neigh[1]].append((self.l2_distance(neigh[1],d,False),d))
                if neigh[1] not in [x[1] for x in self.neighs_NSG[d]]:
                    self.neighs_NSG[d].append((self.l2_distance(d,neigh[1],False),neigh[1]))
    
    #Метод получения навигационной вершины
    def get_navigating_node(self):
        return self.navigating_node
    