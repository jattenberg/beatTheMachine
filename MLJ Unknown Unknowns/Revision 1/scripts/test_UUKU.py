import numpy as np
import pylab as P
import random
import sys
import math


step = 1
weights = [0,0]
rate = 0.1
thresh = 0.5
stddev = 0.6
mux = [0.3, 4.6]
muy = [8.9, 9.3]
denom = math.sqrt(.75*.75 + 1)


def dist(x,y):
    return (x[0]-y[0])**2 + (x[1] - y[1])**2

def point_prob(x,y, xmu, ymu):
    xp = gauss_pdf(x,xmu,0.01+random.gauss(0,0.02))
    yp = gauss_pdf(y,ymu,0.1+random.gauss(0,0.02))
    return xp*yp

def gauss_pdf(x,mu,sigma):
    return math.exp(-(x-mu)**2/(2*(sigma**2)))/math.sqrt(2*math.pi*(sigma**2))

def lin(x):
    return .75*x + 1.2

def lin_dist(x,y):
    num = abs(.75*float(x) - float(y) + 1.2)
    return num/denom

def lin_correct(x,y,z):
    fx = lin(x)
    y_hat = True if y < fx else False   
    return y_hat == z
    
def space(x,y):
    rand = random.gauss(0,stddev)
    f = lin(x)+rand
    if y < f:
        return True
    else:
        return False

def closest(training, item, n):
    dists = []
    for x in training:
        dists.append(dist(x,item))

    distances = np.array(dists)
    closest = distances.argmin()
    neighbors = (closest, )
  #set current closest to large number and find next
    for i in range(1, n):
        distances[closest] = sys.maxint
        closest = distances.argmin()
        neighbors += (closest, )
    return neighbors

def main():
    x = sys.argv
    k = int(x[1])
    trainingsize = int(x[2])

    x_close = np.arange(0.0, 5.5, .04)
    y_close = np.arange(0.0, 11.0, .08)

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

    #compute the number of mislabeled UUs 
    #and the number of examples in the KU region
    
    dif = True
    r = 0;
    unknowns_a = []
    knowns_a = []
    iter = 0
    tot = float(x_close.size*y_close.size)

    while dif:
        iter = iter + 1
        print "on iter: " + str(iter)
        correct = 0
        maxdif = 0

        numunknown = 0
        numknown = 0

        xct = 0
        
        for x in x_close:
            yct = 0
            for y in y_close:
                dist_to_point = lin_dist(x,y)
                if dist_to_point <= r:
                    numknown = numknown + 1
                maxdif = dist_to_point if dist_to_point > maxdif else maxdif
                
                if dist_to_point > r and not lin_correct(x, y, z[yct][xct]):
                    numunknown = numunknown+1

                if lin_correct(x, y, z[yct][xct]):
                    correct = correct + 1

                yct = yct + 1
            xct = xct + 1

        dif = True if r < maxdif else False
        print "correct: " + str(float(correct)/tot)
 
        unknowns_a.append(float(numunknown)/tot)
        knowns_a.append(float(numknown)/tot)
        r = r + step



    P.figure()
    P.plot(knowns_a, unknowns_a, 'k', linewidth=2.0)
    #draw points for knn

    print "print shuffling points"
    points = range(0,x_close.size*y_close.size)
    random.shuffle(points)
    print "done"
    indicies = dict()
    i = 0
    while i < trainingsize if trainingsize < tot else tot:
        indicies[points[i]] = 1
        i = i + 1
    
    ct = 0
    xct = 0
    training = []
    for x in x_close:
        yct = 0
        for y in y_close:
            if ct in indicies:
                training.append([xct,yct])
            ct = ct + 1
            yct = yct + 1
        xct = xct + 1

    Y = []
    xct =  0
    for x in x_close:
        Y.append([])
        for y in y_close:
            Y[xct].append(0)
        xct = xct + 1
    xct = 0

    dif = True
    r = 0;
    unknowns_a = []
    unknowns_c = []
    knowns_a = []
    knowns_c = []

    iter = 0
    tot = float(x_close.size*y_close.size)

    while dif:
        iter = iter + 1
        print "on iter: " + str(iter)
        correct = 0
        maxdif = 0

        numunknown = 0
        numknown = 0
        cunknown = 0
        cknown = 0
        #a really dumb knn search- no data structures used. 
        xct = 0
        accy = 0
        for x in x_close:
            yct = 0
            for y in y_close:
                knn = closest(training, [xct,yct], k)
                fct = tct = 0            
                for i in knn:
                    p = training[i]
                    label = z[p[1]][p[0]]
                    if label:
                        tct = tct + 1
                    else:
                        fct = fct + 1
                Y[yct][xct] = float(tct) / float(tct + fct)
                l = True if tct > fct else False

                correct = False
                if l == z[yct][xct]:
                    accy = accy + 1
                    correct = True
                ccorrect = correct or lin_correct(x, y, z[yct][xct])

                onenn = closest(training, [xct,yct], 1)
                onepoint = training[onenn[0]]
                onedist = math.sqrt(dist([xct, yct],onepoint))

                if onedist < r:
                    numknown = numknown + 1
                maxdif = onedist if onedist > maxdif else maxdif

                if onedist > r and not correct:
                    numunknown = numunknown+1
                dist_to_point = lin_dist(x,y)
                mindist = dist_to_point if dist_to_point > onedist else onedist
                if mindist < r:
                    cknown = cknown + 1
                if mindist > r and not ccorrect:
                    cunknown = cunknown+1

                yct = yct + 1
            xct = xct + 1
            print float(accy) / float(xct*yct)
        dif = True if r < maxdif else False

 
        unknowns_a.append(float(numunknown)/tot)
        knowns_a.append(float(numknown)/tot)
        unknowns_c.append(float(cunknown)/tot)
        knowns_c.append(float(cknown)/tot)
        r = r + step
 

    P.plot(knowns_a, unknowns_a, 'g', linewidth=2.0)
    P.plot(knowns_c, unknowns_c, 'r', linewidth=2.0)

    filename = "decreasing_UU-KU_" + str(trainingsize) + "_" + str(k)+"knn.png"

    P.xlabel("% of points deemed uncertain")
    P.ylabel("% of points that are unknown unknown")
    P.legend(["linear model", str(k)+"-NN", "hybrid"], loc=1)
    P.savefig("../plots/"+filename)
    P.show()



if __name__ == '__main__':
    main()
