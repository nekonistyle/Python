import random
import math #prod
import numpy as np #matrix
import fractions
from functools import reduce

# functions
ReLU = lambda x:max(0,x)
prod = math.prod

# select division operator ('Fraction' or '/')
divop = fractions.Fraction
# divop = lambda x,y: x / y

# constants
idim = 5 # the number of input neurons
a1 = 3 # the number of 1st hidden neurons
a2 = 3 # the number of 2nd hidden neurons

N = a1 * a2 # the number of data (do not change)


# random data
def randomidata(idim,idataRange):
    def mkdata(n,k):
        if n == 0:
            return []
        else:
            return [*mkdata(n-1,k // idataRange), k % idataRange]
    
    return np.array([mkdata(idim,x) for x in random.sample(range(idataRange ** idim), N)]).T

def randomodata(odataRange):
    return np.array(random.choices(range(odataRange), k = N))

# set data
idataRange = 100
odataRange = 100

idata = randomidata(idim,idataRange) # input data must be unique
odata = randomodata(odataRange)


# make parameters
def make_parameters(idata,odata):
    def make_w1(data):
        def dminmax(data):
            if len(data) == 0:
                return (1,0)
            else:
                s = sorted(data)
                min = reduce(lambda dm, x: (dm[0] if x == dm[1] else min([dm[0],x - dm[1]]),x),s[:1],(1,s[0]))
                max = s[-1] - s[0]
                return (min[0],max)

        def f(wS,s):
            mM = dminmax(s)
            K = 1 + divop(2 * wS[1],mM[0])
            return ([K,*wS[0]],(mM[0] + mM[1]) * K)

        return reduce(f,data,([],0))[0]

    w1 = np.array(make_w1(idata))
    flatdata = np.array(sorted(np.array([[np.dot(w1,x) for x in idata.T], odata]).T,key=lambda x:x[0])).T
    flatidata = list(flatdata[0])
    flatodata = flatdata[1]
    W1 = list([w1 for x in range(a1)])

    def make_b(idata):
        def make_b_rec(idata):
            if len(idata) < a2:
                return []
            elif len(idata) == a2:
                return [idata[a2-1] + 1]
            else:
                return [divop(idata[a2-1] + idata[a2],2), *make_b_rec(idata[a2:])]

        if idata == []:
            return []
        else:
            return [idata[0]-1,*make_b_rec(idata)]

    b = make_b(flatidata)

    def make_matrix(data,b,flag):
        if len(data) < a2:
            return data
        elif flag:
            s = data[:a2]
            return [[*s,divop(b[1] + s[-1],2),divop(s[0] + b[0],2)],*make_matrix(data[a2:],b[1:],not flag)]
        else:
            s = list(reversed(data[:a2]))
            return [[*s,divop(b[0] + s[-1],2),divop(s[0] + b[1],2)],*make_matrix(data[a2:],b[1:],not flag)]

    x = make_matrix(flatidata,b,True)
    fx = make_matrix(flatodata,b,True)

    C = min([s[0] for s in fx]) - 1

    def make_ckM(i,j):
        def make_Kj(i):
            Dij = fx[i][j] - C - sum([(-1) ** i * divop(x[i][j] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
            D2ij = fx[i][j+1] - C - sum([(-1) ** i * divop(x[i][j+1] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
            Eij = divop(max([x[i][j-1],x[i][j]]) - b[i],(-1) ** i * (x[i][j] - x[i][j-1])) * abs(Dij)
            E2ij = divop(max([x[i][j-1],x[i][j]]) - b[i],(-1) ** i * (x[i][j+1] - x[i][j])) * abs(D2ij)
            Kij = max([abs(Dij), Eij, E2ij]) + 1
            if i == 0:
                return [Kij]
            else:
                return [*make_Kj(i-1),Kij]

        if i == 0:
            if j == 0:
                ckM = ([],[],[])
            else:
                ckM = make_ckM(a1-1,j-1)
            k = ckM[1]
            M = ckM[2]
            Mij = 1
        else:
            ckM = make_ckM(i-1,j)
            k = ckM[1]
            M = ckM[2]
            Mij = divop(b[i] - k[i-1][j],k[i-1][j] - b[i-1]) * M[i-1][j]
        c = ckM[0]
        if i == 0:
            Kj = make_Kj(a1-1)
            cj = (-1) ** j * max([prod([divop(max([x[s][j-1],x[s][j]]) - b[s],b[s+1] - max([x[s][j-1],x[s][j]])) for s in range(i)]) * Kj[i] for i in range(a1)])
            c = [*c,cj]
        if j == 0:
            M = [*M,[Mij]]
            Dij = fx[i][j] - C - sum([(-1) ** i * divop(x[i][j] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
            kij = divop((-1) ** i * x[i][j] * M[i][j] * c[j] + b[i] * Dij,(-1) ** i * M[i][j] * c[j] + Dij)
            k = [*k,[kij]]
        else:
            M = [*M[:i],[*M[i],Mij],*M[i+1:]]
            Dij = fx[i][j] - C - sum([(-1) ** i * divop(x[i][j] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
            kij = divop((-1) ** i * x[i][j] * M[i][j] * c[j] + b[i] * Dij,(-1) ** i * M[i][j] * c[j] + Dij)
            k = [*k[:i],[*k[i],kij],*k[i+1:]]
        return (c,k,M)

    ckM = make_ckM(a1-1,a2-1)
    c = ckM[0]
    k = ckM[1]
    M = ckM[2]

    def make_w(i,j):
        if i == 0:
            wij = divop(c[j],k[0][j] - b[0])
            if j == 0:
                return [[wij]]
            else:
                w = make_w(i,j-1)
                return [*w[:-1],[*w[-1],wij]]
        else:
            wij = (-1) ** i * (k[i][j] - k[i-1][j]) * M[i-1][j] * divop(c[j],(k[i][j] - b[i]) * (k[i-1][j] - b[i-1]))
            if j == 0:
                return [*make_w(i-1,a2-1),[wij]]
            else:
                w = make_w(i,j-1)
                return [*w[:-1],[*w[-1],wij]]

    w = make_w(a1-1,a2-1)

    def oddinv(s):
        return [-s[i] if i % 2 == 1 else s[i] for i in range(len(s))]

    W2 = [oddinv(s) for s in w]
    c = oddinv(c)

    W3 = [(-1) ** i for i in range(a2)]
    d = C

    def Tnp(s):
        return np.array([[x] for x in s])

    return ((np.array([W3]),np.array([[d]])),(np.array(W2).T,-Tnp(c)),(np.array(W1),-Tnp(b[:-1])))


#neural network
def activation(X):
    return np.array([[ReLU(x) for x in s] for s in X])

def MP1(a,x):
    return np.dot(a[0],x) + a[1]

def MP(A,x):
    return MP1(A[0],activation(MP1(A[1],activation(MP1(A[2],x)))))


# print parameters
def print_parameters(idata,odata):
    print()
    print('(',idim,',',a1,',',a2,', 1 ) ReLU neural network')
    print()
    print('the number of data =',N)
    print()
    print(' input =')
    print(idata)
    print('output =')
    print(odata)
    print()
    print()
    print('parameters')
    A = make_parameters(idata,odata)
    print()
    print('1st layer')
    print('W1 =')
    print(A[2][0])
    print('b1 =')
    print(A[2][1])
    print()
    print('2nd layer')
    print('W2 =')
    print(A[1][0])
    print('b2 =')
    print(A[1][1])
    print()
    print('3rd layer')
    print('W3 =')
    print(A[0][0])
    print('b3 =')
    print(A[0][1])
    print()
    print()
    print('check')
    result = [MP(A,x)[0][0] for x in idata.T]
    print()
    print('MP(x) =', result)
    print('output=', np.array(odata))
    print()
    print('MP(x) - output =',list(map(float,result - np.array(odata))))
    return

print_parameters(idata,odata)
