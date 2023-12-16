import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import tracemalloc
from pympler import asizeof
import time
from multiprocessing import Pool
from sklearn.datasets import make_blobs


class HNSW():
    matr_dist = []
    def __init__(self,data,M,mL,Mmax,efConstruction,num_processes=0,is_pool = False, add_hyrec=True, select = 'simple'):
        self.data = []
        self.indexes = []
        self.neighs = []
        for i,d in enumerate(data):
            self.data.append(d)
            self.indexes.append(i)
            self.neighs.append([])
        num_elements = len(self.indexes)
        self.matr_dist = []
        for i in range(num_elements):
            self.matr_dist.append([])
            for j in range(num_elements):
                 self.matr_dist[i].append(-1)
        self.layers = [[]]
        for ind in self.indexes:
            l = round((-1)*np.log10(np.random.uniform(0,1))*mL)
            for lc in range(l+1):
                self.neighs[ind].append([])
            while len(self.layers) <(l+1):
                self.layers.append([])
            for i in range(0,l+1):
                self.layers[i].append(ind)
        self.enter_point = self.layers[-1][0]
        self.initial_build(M,Mmax)
        if is_pool:
            if add_hyrec:
                print('pre_HyRec  --->')
                self.manage_hyrec_pool(num_processes,M,Mmax,efConstruction)
                #self.preHyRec(M,Mmax,efConstruction) 
            print('build_hnsw --->')
            self.pool_arrange(num_processes,M,Mmax,efConstruction,select)
            print('DFS --->')
            self.DFS_pool()
        else:
            if add_hyrec:
                print('pre_HyRec --->')
                self.preHyRec(M,Mmax,efConstruction)
            print('build_hnsw --->')
            self.build_HNSW_main(M,Mmax,efConstruction,select)
            print('DFS --->')
            self.DFS()
        print('num layers: ', len(self.layers))

    #Инициализация HNSW
    def initial_build(self,K,Mmax):
        print('initial_build --->')
        for lay_ind, layer in enumerate(self.layers):     # уровень
            for node_ind in layer:    # индекс вершины на уровне
                if len(layer)<= K and len(layer)!=1:  # num_neighs - степень вершины на уровне
                    num_neighs = len(layer) - 1
                elif len(layer)==1:
                    num_neighs = 1
                else:
                    num_neighs = K
                layer_cut = layer.copy()
                layer_cut.remove(node_ind)  # список без рассматриваемой вершины
                if len(layer_cut) != 0:   #добавление соседей в обе стороны
                    self.neighs[node_ind][lay_ind] = self.neighs[node_ind][lay_ind]+list(np.random.choice(layer_cut,num_neighs,False))
                    self.neighs[node_ind][lay_ind] = list(set(self.neighs[node_ind][lay_ind]))
                    if len(self.neighs[node_ind][lay_ind])> Mmax:
                        self.neighs[node_ind][lay_ind] = self.neighs[node_ind][lay_ind][:Mmax]
                    for nei in self.neighs[node_ind][lay_ind]:
                        if node_ind not in self.neighs[nei][lay_ind]:
                            self.neighs[nei][lay_ind].append(node_ind)
                        if len(self.neighs[nei][lay_ind])> Mmax:
                            self.neighs[nei][lay_ind] = self.neighs[nei][lay_ind][:Mmax]
    
    #Поиск в глубину для проверки доступности вершин из входной вершины
    def DFS(self):
        L = len(self.layers) - 1
        for lc in range(0,L):
            self.DFS_check(lc)
    
    #Мультипроцессный поиск в глубину
    def DFS_pool(self):
        L = len(self.layers)
        args = []
        for lc in range(0,L):
            args.append(lc)
        pool = Pool(processes=len(args))
        resul = pool.map(self.DFS_check_pool, args)
        for lc,r in enumerate(resul):
            for v in self.layers[lc]:
                self.neighs[v][lc] = r[v]

    def DFS_check_pool(self, lc):
        layer_neighs = [[]]*len(self.indexes)
        for v in self.layers[lc]:
            layer_neighs[v] = self.neighs[v][lc]
        iter = 0
        while(True):
            iter = iter + 1
            visited = []    
            candidates = []
            n = self.enter_point
            candidates.append(n)
            while len(candidates) != 0:
                node = candidates.pop()
                if node not in visited:
                    visited.append(node)
                    for neigh in layer_neighs[node]:
                        if neigh not in visited and neigh not in candidates:
                            candidates.append(neigh)
            declined = []
            for node in self.layers[lc]:
                if node not in visited:
                    declined.append(node)
            if len(declined) == 0:
                break
            for d in declined:
                neigh = self.get_nearest(d,visited,False)
                if d not in layer_neighs[neigh]:
                    layer_neighs[neigh].append(d)
                if neigh not in layer_neighs[d]:
                   layer_neighs[d].append(neigh)  
            if iter > 20:
                break
        return layer_neighs

    #Поиск в глубину на уровне
    def DFS_check(self, lc):
        while(True):
            visited = []    
            candidates = []
            n = self.enter_point
            candidates.append(n)
            while len(candidates) != 0:
                node = candidates.pop()
                if node not in visited:
                    visited.append(node)
                    for neigh in self.neighs[node][lc]:
                        if neigh not in visited and neigh not in candidates:
                            candidates.append(neigh)
            declined = []
            for node in self.layers[lc]:
                if node not in visited:
                    declined.append(node)
            if len(declined) == 0:
                break
            for d in declined:
                neigh = self.get_nearest(d,visited,False)
                if d not in self.neighs[neigh][lc]:
                    self.neighs[neigh][lc].append(d)
                if neigh not in self.neighs[d][lc]:
                    self.neighs[d][lc].append(neigh)  

    #Построение HNSW
    def build_HNSW_main(self,M,Mmax,efConstruction,select):
        for node in self.indexes:
            self.rewire_hnsw_new(node,M,Mmax,efConstruction, select)

    def manage_hyrec_pool(self,num_processes,M,Mmax,efConstruction):
        border = round(len(self.indexes)/num_processes)
        nodes = []
        first = 0
        end = border
        for i in range(num_processes):
            if i == num_processes-1:
                nodes.append(self.indexes[first:])
            else:
                nodes.append(self.indexes[first:end])
            first = first + border
            end = end + border
        args = []
        for i in range(num_processes):
            args.append([nodes[i],M,Mmax,efConstruction])
        pool = Pool(processes=num_processes)
        resul = pool.map(self.HyRec_pool, args)
        result_nns = []
        for r in resul:
            result_nns = result_nns + r
        for v in self.indexes:
            self.neighs[v] = result_nns[v]
            for lc in range(len(self.neighs[v])):
                self.set_in_connections(v,result_nns[v][lc],lc,Mmax)
    
    def HyRec_pool(self,args):
        nodes = args[0]
        M = args[1]
        Mmax = args[2]
        efConstruction = args[3]
        final_neighs = []
        for i in range(len(nodes)):
            final_neighs.append([])
        for i,q in enumerate(nodes):
            W = []
            l = len(self.neighs[q])-1
            for j in range(l+1):
                final_neighs[i].append([])
            for lc in range(l,-1,-1):
                neighs = self.neighs[q][lc]
                W = neighs
                for neighbor in neighs:
                    if lc<=(len(self.neighs[neighbor])-1):
                        W = W + self.neighs[neighbor][lc]
                k = M
                if len(self.layers[lc])>k*2:
                    W = W + list(np.random.choice(self.layers[lc],k,False))
                W = list(set(W))
                if q in W:
                    W.remove(q)
                if len(W)>efConstruction:
                    W = W[:efConstruction]
                neighbors = self.select_neighbors_simple(q,W,M,False)
                final_neighs[i][lc] = neighbors 
        return final_neighs
    
    #HyRec
    def HyRec(self,q,M,Mmax,efConstruction):
        W = []
        l = len(self.neighs[q])-1 #верхний уровень присутствия вершины
        for lc in range(l,-1,-1):
            neighs = self.neighs[q][lc]
            W = neighs
            for neighbor in neighs:
                if lc<=(len(self.neighs[neighbor])-1):
                    W = W + self.neighs[neighbor][lc]
            k = M
            if len(self.layers[lc])>k*2:
                W = W + list(np.random.choice(self.layers[lc],k,False))
            W = list(set(W))
            if q in W:
                W.remove(q)
            if len(W)>efConstruction:
                W = W[:efConstruction]
            neighbors = self.select_neighbors_simple(q,W,M,False)
            self.neighs[q][lc] = neighbors #связи от q к соседям
            self.set_in_connections(q,neighbors,lc,Mmax)
    
    #Установление входящих соединений
    def set_in_connections(self,q,neighbors,lc, Mmax):
        for node in neighbors:
            if q not in self.neighs[node][lc]:
                self.neighs[node][lc].append(q)
            if len(self.neighs[node][lc]) > Mmax:
                temp = self.select_neighbors_simple(node,self.neighs[node][lc],Mmax,False)
                self.neighs[node][lc] = temp
    
    #Простая функция выбора соседей
    def select_neighbors_simple(self,q,candidates,M,q_is_vector): #M - количество кандидатов для возврата
        control_set = []
        for node in candidates:
            heapq.heappush(control_set,(self.l2_distance(node,q,q_is_vector),node))
        control_set = heapq.nsmallest(M,control_set)
        final = [x[1] for x in control_set]
        return final
    
    #Поиск по слою (уровню)
    def search_on_layer(self,q,ep,ef,lc,q_is_vector,path_length =False):  #q - вектор запроса, ep - индекс входной вершины
        self.matr_dist_Q = []
        for i in range(len(self.indexes)):
            self.matr_dist_Q.append(-1)
        
        if type(ep) is list:
            visited = ep.copy()
            candidates = ep.copy()
            W = ep.copy()
        else:
            visited = [ep]
            candidates = [ep]
            W = [ep]
        while len(candidates)>0:
            c = self.get_nearest(q,candidates,q_is_vector)
            candidates.remove(c)
            f = self.get_furthest(q,W,q_is_vector)
            if self.l2_distance(c,q,q_is_vector)>self.l2_distance(f,q,q_is_vector):
                break  
            for e in self.neighs[c][lc]:
                if e not in visited:
                    visited.append(e)
                    f = self.get_furthest(q,W,q_is_vector)
                    if self.l2_distance(e,q,q_is_vector)<self.l2_distance(f,q,q_is_vector) or len(W)<ef:
                        if e not in candidates:
                            candidates.append(e)
                        if e not in W:
                            W.append(e)
                        if len(W)>ef:
                            temp = self.get_furthest(q,W,q_is_vector)
                            W.remove(temp)
        hops = len(visited)
        if path_length:
            #print('Hops for layer =',lc,' : ', len(visited))
            print(len(visited))
        return W, hops

    #Поиск K ближайших соседей 
    def kNN_search(self,q,K,ef, q_is_vector = True,select = 'simple', path_length =False):
        W = []
        ep = self.enter_point
        L = len(self.layers)-1
        hops = 0
        n_visited = []
        for lc in range(L,0,-1):
            W,hops = self.search_on_layer(q,ep,ef=1,lc=lc,q_is_vector=q_is_vector,path_length=path_length)
            ep = self.get_nearest(q,W, q_is_vector)
            n_visited.append(hops)
        W,hops = self.search_on_layer(q,ep,ef,0, q_is_vector,path_length)
        n_visited.append(hops)
        sum_visited = sum(n_visited)
        if select == 'simple':
            knns = self.select_neighbors_simple(q,W,K, q_is_vector)
        elif select == 'heuristic':
            knns = self.select_neighbors_heuristic(q,W,K,0,True,True,q_is_vector)
        else:
            print('There is no method with such name. Simple method will be used')
            knns = self.select_neighbors_simple(q,W,K, q_is_vector)
        final = []
        for nn in knns:
            final.append((self.l2_distance(nn,q, q_is_vector),nn))
        return final,sum_visited
    
    #Эвристика выбора соседей
    def select_neighbors_heuristic(self,q,C,M,lc,extendCandidates, keepPrunedConnections, q_is_vector):
        R = []
        W = C #кандидаты
        if extendCandidates:
            for e in C:
                for eadj in self.neighs[e][lc]:
                    if eadj not in W:
                        W.append(eadj)
        W = list(set(W))
        Wd = []
        while len(W)>0 and len(R)<M:
            e = self.get_nearest(q,W,q_is_vector)
            W.remove(e)
            e_closer_R = 0
            for r in R:
                if self.l2_distance(e,q,q_is_vector)< self.l2_distance(r,q,q_is_vector):
                    e_closer_R= e_closer_R+1
            if e_closer_R==len(R):
               R.append(e)
            else:
                Wd.append(e)
        R = list(set(R))
        Wd = list(set(Wd))
        if keepPrunedConnections:
            while len(Wd)>0 and len(R)<M:
                w = self.get_nearest(q,Wd,q_is_vector)
                Wd.remove(w)
                R.append(w)
        return R
    
    #Добавление вершины в граф
    def insert(self,q,M,Mmax,efConstruction,mL, select = 'simple'): #q - вектор запроса
        ind_q = self.indexes[-1]+1
        self.indexes.append(ind_q)
        self.data.append(q)
        self.neighs.append([])
        W = []
        hops = 0
        ep = self.enter_point
        L = len(self.layers)-1
        l = round((-1)*np.log10(np.random.uniform(0,1))*mL)
        for i in range (l+1):
            self.neighs[ind_q].append([])
        for lc in range(L,l,-1):  
            W, hops = self.search_on_layer(q,ep,ef=1,lc=lc,q_is_vector=True)
            ep = self.get_nearest(q,W,True)
        for lc in range(min(l,L),-1,-1):
            W, hops = self.search_on_layer(q,ep, ef= efConstruction, lc=lc,q_is_vector=True)
            if select == 'simple':
                neighbors = self.select_neighbors_simple(q,W,M,True)
            elif select =='heuristic':
                neighbors = self.select_neighbors_heuristic(q,W,M,lc,True,True,True)
            else:
                print('There is no method with such name. Simple method will be used')
                neighbors = self.select_neighbors_simple(q,W,M,True)
            self.neighs[ind_q][lc] = neighbors
            self.set_in_connections(ind_q,neighbors,lc,Mmax)
        if l >L:
            self.enter_point = ind_q 
            for i in range(l-L):
                self.layers.append([])
        for i in range(0,l+1):
            self.layers[i].append(ind_q)     


    def pool_arrange(self,num_processes,M,Mmax,efConstruction,select):
        border = round(len(self.indexes)/num_processes)
        first = 0
        end = border
        nodes = []
        for i in range(num_processes):
            if i == num_processes-1:
                nodes.append(self.indexes[first:])
            else:
                nodes.append(self.indexes[first:end])
            first = first + border
            end = end + border
        args = []
        for i in range (num_processes):
            args.append([nodes[i],M,Mmax,efConstruction,select])
        pool = Pool(processes=num_processes)
        resul = pool.map(self.arrange_conns_hnsw, args)
        result_nns = []
        for r in resul:
            result_nns = result_nns + r
        for v in self.indexes:
            self.neighs[v] = result_nns[v]
            for lc in range(len(self.neighs[v])):
                self.set_in_connections(v,result_nns[v][lc],lc,Mmax)

    def preHyRec(self,M,Mmax,efConstruction):
        for v in self.indexes:
            self.HyRec(v,M,Mmax,efConstruction)

    def arrange_conns_hnsw(self,args):
        nodes = args[0]
        M = args[1]
        Mmax = args[2]
        efConstruction = args[3]
        select = args[4]
        result_neighs = []
        for i in range(len(nodes)):
            result_neighs.append([])
        L = len(self.layers)-1
        for i,q in enumerate(nodes):
            W = []
            ep = self.enter_point
            l = len(self.neighs[q])-1
            for j in range(len(self.neighs[q])):
                result_neighs[i].append([])
            hops = 0
            for lc in range(L,l,-1):
                W,hops = self.search_on_layer(q,ep,ef=2,lc=lc,q_is_vector=False)
                if q in W:
                    W.remove(q)
                ep = self.get_nearest(q,W,False)
            for lc in range(min(L,l),-1,-1):
                W, hops = self.search_on_layer(q,ep,ef=efConstruction,lc=lc,q_is_vector=False)
                if q in W:
                    W.remove(q)
                if select == 'simple':
                    neighbors = self.select_neighbors_simple(q,W,M,False)
                elif select == 'heuristic':
                    neighbors = self.select_neighbors_heuristic(q,W,M,lc,True,True,False)
                else:
                    print('There is no method with such name. Simple method will be used')
                    neighbors = self.select_neighbors_simple(q,W,M,False)
                neighbors = list(set(neighbors))
                if q in neighbors:
                    neighbors.remove(q)
                result_neighs[i][lc] = neighbors
                #self.set_in_connections(q,neighbors,lc,Mmax)
                ep = W
        return result_neighs

    #Построение HNSW    
    def rewire_hnsw_new(self,q,M,Mmax,efConstruction, select = 'simple'): #q - индекс
        W = []
        ep =self.enter_point
        L = len(self.layers)-1
        l = len(self.neighs[q])-1
        hops = 0
        for lc in range(L,l,-1): 
            W, hops = self.search_on_layer(q,ep,ef=2,lc=lc,q_is_vector=False)
            if q in W:
                W.remove(q)
            ep = self.get_nearest(q,W,False)
        for lc in range(min(L,l),-1,-1):
            W, hops = self.search_on_layer(q,ep,ef=efConstruction,lc=lc,q_is_vector=False)
            if q in W:
                W.remove(q)
            if select == 'simple':
                neighbors = self.select_neighbors_simple(q,W,M,False)
            elif select == 'heuristic':
                neighbors = self.select_neighbors_heuristic(q,W,M,lc,True,True,False)
            else:
                print('There is no method with such name. Simple method will be used')
                neighbors = self.select_neighbors_simple(q,W,M,False)
            neighbors = list(set(neighbors))
            if q in neighbors:
                neighbors.remove(q)
            self.neighs[q][lc] = neighbors
            self.set_in_connections(q,neighbors,lc,Mmax)
            ep = W

    #Получение ближайшей к q вершины из списка
    def get_nearest(self,q,candidates,q_is_vector):
        control_set = []
        for node in candidates:
            heapq.heappush(control_set,(self.l2_distance(node,q,q_is_vector),node))
        control_set = heapq.nsmallest(1,control_set)
        return control_set[0][1]
    
    #Получение самой дальней от q вершины из списка
    def get_furthest(self,q,candidates,q_is_vector):
        control_set = []
        for node in candidates:
            heapq.heappush(control_set,(self.l2_distance(node,q,q_is_vector),node))
        control_set = heapq.nlargest(1,control_set)
        return control_set[0][1]
    
    #Вывод параметров HNSW
    def show_structure(self):
        print('HNSW:')
        print('indexes: ', self.indexes)
        for i, n in enumerate(self.neighs):
            print('i = ', i, ' neighs: ', n)
        print('layers: ', self.layers)
        print('enter point: ', self.enter_point)
    
    #Отображение слоев HNSW 
    def show_graph(self):
        for i,layer in enumerate(self.layers):
            edges = []
            nodes = layer
            for node_ind in layer:
                neighbors = self.neighs[node_ind][i]
                for neigh in neighbors:
                    if (node_ind,neigh) not in edges:
                        edges.append((node_ind,neigh))
            G = nx.DiGraph()  # создаём объект графа
            G.add_nodes_from(nodes) # добавляем информацию в объект графа
            G.add_edges_from(edges)
            nx.draw(G, with_labels=True, font_weight='bold') # рисуем граф и отображаем его
            plt.show()
    
    #Вывод памяти, занимаемой параметрами HNSW
    def memory_stat(self):
        print('indexes: ', asizeof.asizeof(self.indexes))
        print('data: ', asizeof.asizeof(self.data))
        print('neighs: ', asizeof.asizeof(self.neighs))
        print('layers: ', asizeof.asizeof(self.layers))
    
    #Функция расстояния
    def l2_distance1(self,a,q, q_is_vector=False):
        if q_is_vector:
            if self.matr_dist_Q[a] == -1:
                result =  np.linalg.norm(self.data[a] - q)
                self.matr_dist_Q[a] = result
            else:
                result = self.matr_dist_Q[a] 
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
    