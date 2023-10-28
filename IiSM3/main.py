import math
import statistics as st
from abc import abstractmethod
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats

def Laplace_cdf(x, l):
    if x < 0:
        return 0.5*math.e**(l*x)
    else:
        return 1 - 0.5*math.e**(-l*x)
def Cauchy_cdf(x, m, c):
    return 0.5 + (1/math.pi)*math.atan((x-m)/c)


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


class N(AbstractGenerator):
    def __init__(self,m: float, d: float, a: AbstractGenerator, N = 12):
        self.m = m
        self.a = a
        self.d = d
        self.N = N
        self.a.reset()

    def reset(self):
        self.a.reset()
    def get_next(self):
        A = self.a.get_seq(self.N)
        x = 0
        for i in range(self.N):
            x += A[i]
        x -= self.N/2
        x *= (12/self.N)**(1/2)
        return (x*self.d +self.m)

class Cauchy(AbstractGenerator):
    def __init__(self,m: float, c: float, a: AbstractGenerator):
        self.m = m
        #self.a = N(0,1,a)
        self.a = a
        self.c = c
        self.a.reset()

    def reset(self):
        self.a.reset()
    def get_next(self):

        A = self.a.get_next()
        #B = self.a.get_next()
        x = self.m + self.c * math.tan(math.pi*(A - 1/2))
        return x

class Laplace(AbstractGenerator):
    def __init__(self,l: float, a: AbstractGenerator):
        self.l = l
        self.a = a
        self.a.reset()

    def reset(self):
        self.a.reset()
    def get_next(self):
        A = self.a.get_next()
        x = 0
        if(A < 0.5):
            x = (1/self.l)*math.log(2*A)
        else:
            x = -(1/self.l)*math.log(2*(1-A))
        return x
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

    def Pearson_N(self, X, m: float, d: float):
        n = len(X)
        K = int((10*n)**(1/3))
        X = sorted(X)
        h = (X[n-1] - X[0])/K
        T_obs = np.histogram(X, bins=K)[0]
        #print(T_obs[0])

        T_exp = [0.] * K
        H = X[0]+h
        T_exp[0] = (stats.norm.cdf(X[0]+h,m,d) - stats.norm.cdf(-math.inf,m,d))*n
        T_exp[K-1] = (stats.norm.cdf(math.inf, m, d) - stats.norm.cdf(X[n-1]-h, m, d)) * n
        for i in range(1,K-1):
            T_exp[i] = (stats.norm.cdf(H+h,m,d) - stats.norm.cdf(H,m,d))*n
            H+=h

        K1 = K
        for i in range(K):
            if (T_obs[i] < 2):
                K1 -= 1
        T_obs1 = [0.] * K1
        T_exp1 = [0.] * K1
        k = 0
        for i in range(K):
            T_obs1[k] += T_obs[i]
            T_exp1[k] += T_exp[i]
            if (T_obs[i] >= 2 and k != K1 - 1):
                k += 1
        statistic, pvalue = stats.chisquare(f_obs=T_obs1, f_exp=T_exp1)
        #print(pvalue)
        crit_value = stats.chi2.ppf(0.95, K1)
        if statistic < crit_value:
            return True
        else:
            return False
    def Pearson_Cauchy(self, X, m: float, c: float):
        n = len(X)
        K = 3*int((10 * n) ** (1 / 3))
        X = sorted(X)
        h = (X[n-1] - X[0])/K
        T_obs = np.histogram(X, bins = K)[0]

        T_exp = [0.] * K
        H = X[0]+h
        T_exp[0] = (Cauchy_cdf(X[0]+h,m,c) - Cauchy_cdf(-math.inf,m,c))*n
        T_exp[K-1] = (Cauchy_cdf(math.inf, m, c) - Cauchy_cdf(X[n-1]-h, m, c)) * n
        for i in range(1,K-1):
            T_exp[i] = (Cauchy_cdf(H+h,m,c) - Cauchy_cdf(H,m,c))*n
            H+=h
        K1 = K
        for i in range(K):
            if(T_obs[i] < 2):
                K1-=1
        T_obs1 = [0.]*K1
        T_exp1 = [0.]*K1
        k = 0
        for i in range(K):
            T_obs1[k]+=T_obs[i]
            T_exp1[k] += T_exp[i]
            if (T_obs[i] >= 2 and k!=K1-1):
                k += 1


        statistic, pvalue = stats.chisquare(f_obs=T_obs1, f_exp=T_exp1)
        #print(pvalue)
        crit_value = stats.chi2.ppf(0.95, K1)
        if statistic < crit_value:
            return True
        else:
            return False

    def Pearson_Laplace(self, X, l: float):
        n = len(X)
        K = int((10 * n) ** (1 / 3))
        X = sorted(X)
        h = (X[n - 1] - X[0]) / K
        T_obs = np.histogram(X, bins=K)
        #print(T_obs[0])

        T_exp = [0.] * K
        H = X[0] + h
        T_exp[0] = (Laplace_cdf(X[0] + h, l) - Laplace_cdf(-math.inf, l)) * n
        T_exp[K - 1] = (Laplace_cdf(math.inf, l) - Laplace_cdf(X[n - 1] - h, l)) * n
        for i in range(1, K - 1):
            T_exp[i] = (Laplace_cdf(H + h, l) - Laplace_cdf(H, l)) * n
            H += h
        #print(T_exp)




        statistic, pvalue = stats.chisquare(f_obs=T_obs[0], f_exp=T_exp)
        # print(pvalue)
        crit_value = stats.chi2.ppf(0.95, K+2)
        if statistic < crit_value:
            return True
        else:
            return False
    def Kolmogorov_N(self, X, m, d):
        statistic, pvalue=stats.kstest(X, 'norm', args=(m, d))
        if pvalue > 0.05:
            return True
        else:
            return False
    def Kolmogorov_Cauchy(self, X, m, c):
        statistic, pvalue=stats.kstest(X, 'cauchy', args=(m, c))
        if pvalue > 0.05:
            return True
        else:
            return False
    def Kolmogorov_Laplace(self, X, l):
        statistic, pvalue=stats.kstest(X, 'laplace', args=(0, 1/l))
        if pvalue > 0.05:
            return True
        else:
            return False
Norm = N(5,3,MaclarenMarsalji(MKM(78125, 78125,2**31),MKM(16387, 16387,2**31),64))
Cau = Cauchy(-1,3,MaclarenMarsalji(MKM(78125, 78125,2**31),MKM(16387, 16387,2**31),64))
Lap = Laplace(2, MaclarenMarsalji(MKM(78125, 78125,2**31),MKM(16387, 16387,2**31),64))
T = Test()
n = 1000
alpha_n = 0
alpha_n1 = 0
alpha_c = 0
alpha_c1 = 0
alpha_l = 0
alpha_l1 = 0
M_n = 0
M_c = 0
M_l = 0
D_n = 0
D_c = 0
D_l = 0
T = Test()
for i in range(n):
    n_seq = Norm.get_seq(100)
    M_n+=st.mean(n_seq)/n
    D_n+=st.variance(n_seq)/n
    if(T.Pearson_N(n_seq,5,3)==False):
        alpha_n += 1
    if (T.Kolmogorov_N(n_seq, 5, 3) == False):
        alpha_n1 += 1
    c_seq = Cau.get_seq(100)
    if (T.Pearson_Cauchy(c_seq, -1, 3) == False):
        alpha_c += 1
    if (T.Kolmogorov_Cauchy(c_seq, -1, 3) == False):
        alpha_c1 += 1
    l_seq = Lap.get_seq(100)
    M_l += st.mean(l_seq) / n
    D_l += st.variance(l_seq) / n
    if (T.Pearson_Laplace(l_seq, 2) == False):
        alpha_l += 1
    if (T.Kolmogorov_Laplace(l_seq, 2) == False):
        alpha_l1 += 1
alpha_n/=n
alpha_c/=n
alpha_l/=n
alpha_n1/=n
alpha_c1/=n
alpha_l1/=n
print("Ошибка первого рода для нормального распределения (хи-квадрат): "+str(alpha_n))
print("Ошибка первого рода для нормального распределения (Колмогоров): "+str(alpha_n1))

print("Матожидание для нормального распределения: "+str(M_n))
print("Дисперсия для нормального распределения: "+str(D_n))
print("Теоретическое матожидание для нормального распределения: "+str(5))
print("Теоретическая дисперсия для нормального распределения: "+str(3))

print("Ошибка первого рода для распределения Коши (хи-квадрат): "+str(alpha_c))
print("Ошибка первого рода для распределения Коши (Колмогоров): "+str(alpha_c1))

print("Ошибка первого рода для распределения Лапласа (хи-квадрат): "+str(alpha_l))
print("Ошибка первого рода для распределения Лапласа (Колмогоров): "+str(alpha_l1))
print("Матожидание для распределения Лапласа: "+str(M_l))
print("Дисперсия для распределения Лапласа: "+str(D_l))
print("Теоретическое матожидание для распределения Лапласа: "+str(0))
print("Теоретическая дисперсия для распределения Лапласа: "+str(2/2**2))



