#! usr/bin/env python 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import groupby

#Define colours 
martaRed = "#%02x%02x%02x" %(194, 76, 81)
martaGreen = "#%02x%02x%02x" %(84, 166, 102)
martaBlue = "#%02x%02x%02x" %(76, 112, 176)
martaPurple = "#%02x%02x%02x" %(127, 112, 176) 
martaGold = "#%02x%02x%02x" %(204, 184, 115) 

#Colours
Icolour = '#DB2420' #Central line red
Ocolour = '#00A0E2' #Victoria line blue
O2colour = '#868F98' #Jubilee line grey
D2colour = '#F386A0' #Hammersmith line pink
D4colour = '#97015E' #Metropolitan line magenta
D6colour = '#B05F0F' #Bakerloo line brown
D5colour = '#00843D' #District line green
O3colour = '#021EA9' 

font = {'size'   : 20}
matplotlib.rc('font', **font)


def plot_bif(p, u, s):

    #Group according to stability
    plt.ion() 
    for g_s, group in groupby(zip(zip(p, u), s), lambda p: p[1]):
        g_p, g_u = [], []
        for i in group:
            g_p.append(i[0][0])
            g_u.append(i[0][1])
        #Plot as dashed or solid line
        if g_s:
            plt.plot(g_p, g_u, linestyle='--', color=martaRed, linewidth=2.0)
        else:
            plt.plot(g_p, g_u, linestyle='-', color=martaRed, linewidth=2.0)

    plt.ylabel('$||u||_{2}$', fontsize=24, rotation=0, labelpad = 26)
    plt.xlabel('$h$', fontsize=24)
    plt.tight_layout()
    plt.pause(0.001) 