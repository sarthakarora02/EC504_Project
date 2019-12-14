import math

def boundaryPenalty(ip, iq):
    SIGMA = 30
    bp = 100 * math.exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp
# def boundaryPenalty(ip, iq, a,b):
#     bp = 100 * math.exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))/math.sqrt(pow((b[1]-a[1]),2)+pow((b[0]-a[0]),2))
#     return bp
