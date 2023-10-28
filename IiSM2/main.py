import math
import statistics as st
from abc import abstractmethod
#import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
class AbstractGenerator:
    @abstractmethod
    def get_next(self):
        pass

    @abstractmethod
    def reset(self):
        pass
    def get_seq(self, n: int):
        seq = [0.] * n
        for i in range(n):
            seq[i] = self.get_next()
        return seq


class MKM(AbstractGenerator):
    def __init__(self, alpha, beta, M):
        self.alpha = alpha
        self.beta = beta
        self.M = M
        self.alpha_i = self.alpha
    def reset(self):
        self.alpha_i = self.alpha
    def get_next(self):
        next = (self.alpha_i*self.beta)%self.M
        self.alpha_i = next
        return self.alpha_i/self.M
class MaclarenMarsalji(AbstractGenerator):
    def __init__(self, b: AbstractGenerator, c: AbstractGenerator, K: int):
        self.b = b
        self.c = c
        self.K = K
        self.b.reset()
        self.c.reset()
        self.V = self.b.get_seq(self.K)
    def reset(self):
        self.b.reset()
        self.c.reset()
        self.V = self.b.get_seq(self.K)
    def get_next(self):
        s = int(self.c.get_next()*self.K)
        next = self.V[s]
        self.V[s] = self.b.get_next()
        return next
class DSV_Bi(AbstractGenerator):
    def __init__(self,m: int,p: float, a: AbstractGenerator):
        self.m = m
        self.p = p
        self.a = a
        self.a.reset()

    def reset(self):
        self.a.reset()
    def get_next(self):
        A = self.a.get_seq(self.m)
        x = 0
        for i in range(self.m):
            if(A[i]<self.p):
                x+=1
        return x
class DSV_Exp(AbstractGenerator):
    def __init__(self,l: float, a: AbstractGenerator):
        self.l = l
        self.a = a
        self.a.reset()

    def reset(self):
        self.a.reset()
    def get_next(self):
        A = self.a.get_next()
        k = 0
        x = math.e**(-self.l)
        Summ = x
        while(A>Summ):
            k=k+1
            x = x*self.l/k
            Summ+=x
        return k


class Test:
    def Kolmogorov_r(self, X, a, b):
        statistic, pvalue=stats.kstest(X, 'uniform', args=(a, b))
        if pvalue > 0.05:
            print("Критерий Колмогорова подтверждает гипотезу о равномерном распределении!")
        else:
            print("Критерий Колмогорова НЕ подтверждает гипотезу о равномерном распределении!")
    def Pearson_r(self, X, a, b):
        n = len(X)
        k = int((10*n)**(1/3))
        T = [0.]*k
        h = (b-a)/k
        X = sorted(X)
        a_i = a+h
        j = 0
        for i in range(n):
            if X[i]<a_i:
                T[j]+=1/n
            else:
                a_i+=h
                j+=1
                T[j] += 1 / n
        T_0 = [1/k]*k
        statistic, pvalue=stats.chisquare(f_obs=T,f_exp=T_0)
        if pvalue > 0.05:
            print("Критерий Пирсона подтверждает гипотезу о равномерном распределении!")
        else:
            print("Критерий Пирсона НЕ подтверждает гипотезу о равномерном распределении!")
    def Pearson_exp(self, X, l: float):
        n = len(X)
        X = sorted(X)
        K = 1
        for i in range(1,n):
            if X[i]!=X[i-1]:
                K+=1
        T_obs = [0.]*K
        k = 0
        T_obs[0] = 1
        for i in range(1,n):
            if X[i]!=X[i-1]:
                k+=1
            T_obs[k]+=1
        T_exp = [0.]*K
        T_exp[0] = math.exp(-l)
        Summ = T_exp[0]
        for i in range(1,K):
            T_exp[i] = T_exp[i-1]*l/i
            Summ+=T_exp[i]
        T_exp[K-1]+=1-Summ
        for i in range(K):
            T_exp[i]*=n
        #print(T_obs)
        #print(T_exp)



        statistic, pvalue=stats.chisquare(f_obs=T_obs,f_exp=T_exp)
        #print(pvalue)
        if pvalue > 0.05:
            return True
        else:
            return False

    def Pearson_bin(self, X, m: float, p: float):
        n = len(X)
        X = sorted(X)
        K = 1
        for i in range(1, n):
            if X[i] != X[i - 1]:
                K += 1
        T_obs = [0.] * K
        k = 0
        T_obs[0] = 1 / n
        for i in range(1, n):
            if X[i] != X[i - 1]:
                k += 1
            T_obs[k] += 1 / n
        T_exp = [0.] * K
        for i in range(0, K):
            T_exp[i] = math.comb(m, i) * (p ** i) * ((1 - p) ** (m - i))
        for i in range(K):
            T_exp[i]*=n
            T_obs[i]*=n
        #print(T_obs)
        #print(T_exp)

        statistic, pvalue = stats.chisquare(f_obs=T_obs, f_exp=T_exp)
        #print(pvalue)
        if pvalue > 0.05:
            return True
        else:
            return False




Mkm = MKM(50653, 50653,2**31)
Macmar = MaclarenMarsalji(MKM(78125, 78125,2**31),MKM(16387, 16387,2**31),64)
Bin = DSV_Bi(5,0.6,Macmar)
Exp = DSV_Exp(2, Macmar)
N = 1000
alpha_bin = 0
alpha_exp = 0
beta_bin = 0
beta_exp = 0
M_exp = 0
M_bin = 0
D_exp = 0
D_bin = 0
T = Test()
for i in range(N):
    bin_seq = Bin.get_seq(1000)
    M_bin+=st.mean(bin_seq)/N
    D_bin+=st.variance(bin_seq)/N
    if(T.Pearson_bin(bin_seq,5,0.6)==False):
        alpha_bin += 1
alpha_bin/=N
print("Ошибка первого рода для биномиального распределения: "+str(alpha_bin))
print("Матожидание для биномиального распределения: "+str(M_bin))
print("Дисперсия для биномиального распределения: "+str(D_bin))
for i in range(N):
    exp_seq = Exp.get_seq(1000)
    M_exp+=st.mean(exp_seq)/N
    D_exp+=st.variance(exp_seq)/N
    if(T.Pearson_exp(exp_seq,2)==False):
        alpha_exp += 1
alpha_exp/=N
print("Ошибка первого рода для распределения Пуассона: "+str(alpha_exp))
print("Матожидание для распределения Пуассона: "+str(M_exp))
print("Матожидание для распределения Пуассона: "+str(D_exp))

