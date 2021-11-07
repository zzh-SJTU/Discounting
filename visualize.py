#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:59:50 2021

@author: zzh
"""
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[ 1617.86,5655.36,118.01,7.47,1]
plt.plot(x,y,color="blue", linewidth=1.0)
plt.xlabel('gram')
plt.ylabel('Perplexity')
xticks(np.linspace(0,5,6,endpoint=True))
savefig("exercice2.png",dpi=500)

