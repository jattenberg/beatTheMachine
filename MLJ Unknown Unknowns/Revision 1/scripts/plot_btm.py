import numpy as np
import pylab as P
import random
import sys
import math

weights = [0,0]
rate = 0.1
thresh = 0.5
stddev = 0.6
mux = [0.3, 4.6]
muy = [8.9, 9.3]

def point_prob(x,y, xmu, ymu):
    xp = gauss_pdf(x,xmu,0.01+random.gauss(0,0.02))
    yp = gauss_pdf(y,ymu,0.1+random.gauss(0,0.02))
    return xp*yp

def gauss_pdf(x,mu,sigma):
    return math.exp(-(x-mu)**2/(2*(sigma**2)))/math.sqrt(2*math.pi*(sigma**2))

def update(x,z):
    result = ip(x)
    error = z - result
    weights[0] = weights[0] + rate*error*x
    weights[1] = weights[1] + rate*error

def ip(x):
    return weights[0]*x + weights[1]

def lin(x):
    return .75*x + 1.2

def space(x,y):
    rand = random.gauss(0,stddev)
    f = lin(x)+rand
    if y < f:
        return 1
        #return 0.5 + abs(y - f)**2
    else:
        return 0
        #return 0.5 - 0.2*(abs(y + f)**2)
    

def main():
    x_close = np.arange(0.0, 5.5, .01)
    y_close = np.arange(0.0, 11.0, .02)

    z = []
    xct =  0
    for x in x_close:
        z.append([])
        for y in y_close:
            z[xct].append(0)
        xct = xct + 1
    xct = 0
    for x in x_close:
        yct = 0
        for y in y_close:
            z[yct][xct] = space(x,y) 
            yct = yct + 1
        xct = xct + 1;
    P.figure()
    P.subplot(2,1,1)
    d = P.pcolor(x_close, y_close ,z)
    q = P.winter()

    P.plot(x_close, lin(x_close), 'k')
    P.plot(x_close, [lin(x) + 2*stddev for x in x_close], 'k') 
    P.plot(x_close, [lin(x) - 2*stddev for x in x_close], 'k') 

    P.xlim([0,5])
    P.ylim([0,10])
    P.rc('text', usetex=True)
    P.rc('font', family='serif')
    P.title(r"typical prediction problem with an $\epsilon$-uncertainty radius")

    #add disjunctive regions to z
    
    P.subplot(2,1,2)
    for i in [0,1]:
        max = 0
        xct = 0
        for x in x_close:
            yct = 0
            for y in y_close:
                p = point_prob(x,y,mux[i], muy[i])
                max = p if p > max else max
                z[yct][xct] = 1 if p > 6.2 else z[yct][xct]
                yct = yct+1
            xct = xct + 1

    d = P.pcolor(x_close, y_close ,z)
    q = P.winter()

    P.plot(x_close, lin(x_close), 'k')
    P.plot(x_close, [lin(x) + 2*stddev for x in x_close], 'k') 
    P.plot(x_close, [lin(x) - 2*stddev for x in x_close], 'k') 

    P.xlim([0,5])
    P.ylim([0,10])
    P.rc('text', usetex=True)
    P.rc('font', family='serif')
    P.title(r"the same prediction problem with disjunctive ``unknowns''")


    P.savefig("../plots/example_function_2.png")
    P.show()

if __name__ == '__main__':
    main()
