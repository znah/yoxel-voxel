from numpy import *
import os

import pylab as pl
#from matplotlib import pyplot as pl

def make_count_chart(inName, outName):
    a = fromfile(inName, int32).reshape(768, 1024)
    
    fig = pl.figure()
    fig.subplots_adjust(left = 0, right=1)
    fig.set_figwidth(9)

    ax = fig.add_subplot(111)

    cax = ax.imshow(a, origin='bottom', interpolation='nearest', cmap=pl.cm.gray_r)

    fig.colorbar(cax)
    
    pl.savefig(outName, dpi=150)
    


make_count_chart('data/dumps/iter_count.dat', 'images/trace_iters.png') 
os.system('imconvert -quality 90 images/trace_iters.png images/trace_iters.jpg')




