import math
import random as rn
from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
class AbstractGenerator:
    @abstractmethod
    def get_next(self):
        pass

    @abstractmethod
    def reset(self):
        pass
    def get_seq(self, n: int):
        self.reset()
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
class Test:
    def Kolmogorov_r(self, X, a, b):
        statistic, pvalue=stats.kstest(seq_mkm, 'uniform', args=(a, b))
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


Mkm = MKM(50653, 50653,2**31)
seq_mkm = Mkm.get_seq(100)
#for i in range(100):
#    print(seq_mkm[i])

Macmar = MaclarenMarsalji(MKM(78125, 78125,2**31),MKM(16387, 16387,2**31),64)
seq_mm = Macmar.get_seq(1000)
#for i in range(1000):
#    print(seq_mm[i])
t = Test()
print("Проверка по критерию Колмогорова:")
t.Kolmogorov_r(seq_mkm,0,1)
t.Kolmogorov_r(seq_mm,0,1)
print("Проверка по критерию хи-квадрат:")
t.Pearson_r(seq_mkm,0,1)
t.Pearson_r(seq_mm,0,1)

