import math

def correlationGenerator(x,y):
    n = len(x)

    sqrx = 0
    sqry = 0
    sumx = 0
    sumy = 0
    sumxy = 0

    for i in x:
        sumx += i
        sqrx += i*i
        sumxy += (x[i-1] * y[i-1])

    for j in y:
        sumy += j
        sqry += j*j

    sxx = sqrx - ((sumx*sumx)/n)
    syy = sqry - ((sumy*sumy)/n)
    sxy = sumxy - ((sumx * sumy)/n)

    r = sxy/(math.sqrt(sxx*syy))
    return(r)


