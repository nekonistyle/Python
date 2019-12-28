import random
import math #prod
import numpy as np #matrix
import fractions
import matplotlib.pyplot as plt #graph

# constants
idataRange = 100
odataRange = 100

a1 = 3 # the number of 1st hidden neurons
a2 = 3 # the number of 2nd hidden neurons


# the number of data
N = a1 * a2
print('a1 =',a1)
print('a2 =',a2)
print('the number of data =',N)
print()

# random data
def randomidata():
    return sorted(random.sample(range(idataRange), N))
def randomodata():
    return random.choices(range(odataRange), k = N)


# set data
idata = randomidata() # input data must be sorted and unique
odata = randomodata()
print(' input =',idata)
print('output =',odata)
print()


# functions
ReLU = lambda x:max(0,x)

frac = fractions.Fraction
prod = math.prod


# make parameters
print('parameters')
print()

W1 = list([1 for x in range(a1)])

def make_b(idata):
    def make_b_rec(idata):
        if len(idata) < a2:
            return []
        elif len(idata) == a2:
            return [idata[a2-1] + 1]
        else:
            return [frac(idata[a2-1] + idata[a2],2), *make_b_rec(idata[a2:])]

    if idata == []:
        return []
    else:
        return [idata[0]-1,*make_b_rec(idata)]

b = make_b(idata)


def make_matrix(data,b,flag):
    if len(data) < a2:
        return data
    elif flag:
        s = data[:a2]
        return [[*s,frac(b[1] + s[-1],2),frac(s[0] + b[0],2)],*make_matrix(data[a2:],b[1:],not flag)]
    else:
        s = list(reversed(data[:a2]))
        return [[*s,frac(b[0] + s[-1],2),frac(s[0] + b[1],2)],*make_matrix(data[a2:],b[1:],not flag)]

x = make_matrix(idata,b,True)
fx = make_matrix(odata,b,True)


C = min([s[0] for s in fx]) - 1

def make_ckM(i,j):
    def make_Kj(i):
        Dij = fx[i][j] - C - sum([(-1) ** i * frac(x[i][j] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
        D2ij = fx[i][j+1] - C - sum([(-1) ** i * frac(x[i][j+1] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
        Eij = frac(max([x[i][j-1],x[i][j]]) - b[i],(-1) ** i * (x[i][j] - x[i][j-1])) * abs(Dij)
        E2ij = frac(max([x[i][j-1],x[i][j]]) - b[i],(-1) ** i * (x[i][j+1] - x[i][j])) * abs(D2ij)
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
        Mij = frac(b[i] - k[i-1][j],k[i-1][j] - b[i-1]) * M[i-1][j]
    c = ckM[0]
    if i == 0:
        Kj = make_Kj(a1-1)
        cj = (-1) ** j * max([prod([frac(max([x[s][j-1],x[s][j]]) - b[s],b[s+1] - max([x[s][j-1],x[s][j]])) for s in range(i)]) * Kj[i] for i in range(a1)])
        c = [*c,cj]
    if j == 0:
        M = [*M,[Mij]]
        Dij = fx[i][j] - C - sum([(-1) ** i * frac(x[i][j] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
        kij = frac((-1) ** i * x[i][j] * M[i][j] * c[j] + b[i] * Dij,(-1) ** i * M[i][j] * c[j] + Dij)
        k = [*k,[kij]]
    else:
        M = [*M[:i],[*M[i],Mij],*M[i+1:]]
        Dij = fx[i][j] - C - sum([(-1) ** i * frac(x[i][j] - k[i][t],k[i][t] - b[i]) * M[i][t] * c[t] for t in range(j)])
        kij = frac((-1) ** i * x[i][j] * M[i][j] * c[j] + b[i] * Dij,(-1) ** i * M[i][j] * c[j] + Dij)
        k = [*k[:i],[*k[i],kij],*k[i+1:]]
    return (c,k,M)

ckM = make_ckM(a1-1,a2-1)
c = ckM[0]
k = ckM[1]
M = ckM[2]

#print('c='+str(c))
#print('k='+str(k))

def make_w(i,j):
    if i == 0:
        wij = frac(c[j],k[0][j] - b[0])
        if j == 0:
            return [[wij]]
        else:
            w = make_w(i,j-1)
            return [*w[:-1],[*w[-1],wij]]
    else:
        wij = (-1) ** i * (k[i][j] - k[i-1][j]) * M[i-1][j] * frac(c[j],(k[i][j] - b[i]) * (k[i-1][j] - b[i-1]))
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


# print parameters
print('1st layer')

npW1 = Tnp(W1)
print('W1 =')
print(npW1)

npb1 = -Tnp(b[:-1])
print('b1 =')
print(npb1)

print()
print('2nd layer')

npW2 = np.array(W2).T
print('W2 =')
print(npW2)

npb2 = -Tnp(c)
print('b2 =')
print(npb2)

print()
print('3rd layer')

npW3 = np.array([W3])
print('W3 =')
print(npW3)

npb3 = np.array([[d]])
print('b3 =')
print(npb3)

print()


# check
def activation(X):
    return np.array([[ReLU(x) for x in s] for s in X])

def MP1(W,b,x):
    return np.dot(W,x) + b

def MP(x):
    return MP1(npW3,npb3,activation(MP1(npW2,npb2,activation(MP1(npW1,npb1,np.array([[x]]))))))

result = [MP(x)[0][0] for x in idata]
print('MP(x) =', result)
print('output=', np.array(odata))
print()
print('MP(x) - output =',list(map(float,result - np.array(odata))))


# graph
x = np.array([idata[0]-2,*sorted(sum(k,b[:-1])),idata[-1]+2])
y = np.array([MP(n)[0][0] for n in x])

plt.subplot(2,1,1)
plt.title('(1,'+str(a1)+','+str(a2)+',1) neural network expressing random '+str(a1)+'*'+str(a2)+' data')
plt.scatter(idata,odata)
plt.plot(x,y)

plt.subplot(2,1,2)
plt.ylim([-odataRange*0.05,odataRange*1.05])
plt.xlabel('Input')
plt.ylabel('Output')
plt.scatter(idata,odata)
plt.plot(x,y)

#plt.subplot(3,1,3)
#plt.scatter(idata,odata)
#x = np.array(range(idata[0]-2,idata[-1]+2))
#y = np.array([abs(MP(n)[0][0]) for n in x])
#plt.plot(x,y)
#plt.yscale("log")

plt.show()
