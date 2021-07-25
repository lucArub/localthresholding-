# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import glob         
import time as t  
from matplotlib import pyplot as plt
import skimage.io   
import skimage.viewer
from skimage.filters import (threshold_niblack,
                             threshold_sauvola,
                             threshold_otsu)

from filter_mod import (threshold_singh,
                        threshold_niblack_slow,
                        threshold_sauvola_slow)

img_number = 0 # counter
path = 'yourpath/*.*'


r = range(91,92,2)  # dim(matrix) wxw that I want to test
MAX = 5

time_nib = []
time_nib_slow = []
time_sau = []
time_sau_slow = []
time_sin = []   
time_ots = []  # array in which I store the times



for file in glob.glob(path):   

    img_number+=1
   
    image1 = skimage.io.imread(fname=file,as_gray=True) # as_gray=True no RGB color
    image2 = skimage.io.imread(fname=file,as_gray=True)
    image3 = skimage.io.imread(fname=file,as_gray=True)
    image4 = skimage.io.imread(fname=file,as_gray=True)
    image5 = skimage.io.imread(fname=file,as_gray=True)
    image6 = skimage.io.imread(fname=file,as_gray=True)
    
                                    

    for i in range(0,MAX):    
        print('Iterazione '+str(i)+' di '+str(MAX))
        sub_nib = []
        sub_nib_slow = []
        sub_sau = []
        sub_sau_slow = []
        sub_sin = []
        sub_ots = []
      
        for w in r:
            # slow --> without sum integral image
            start = t.time() #start clock
            t_niblack = threshold_niblack(image1, w, k=0.2) #  T(x,y) niblack
            end = t.time()- start #stop

            sub_nib.append(end) #appending the time in the array

            start = t.time() #start clock
            t_niblack_slow = threshold_niblack_slow(image2, w, k=0.2) # T(x,y) niblack slow
            end = t.time()- start #stop

            sub_nib_slow.append(end) #appending the time in the array


            
            start = t.time()
            t_sauvola = threshold_sauvola(image3, w, k=0.2) #  T(x,y) sauvola
            end = t.time()- start

            sub_sau.append(end) #appending the time in the array

            start = t.time()
            t_sauvola_slow = threshold_sauvola_slow(image4, w, k=0.2) #  T(x,y) sauvola slow
            end = t.time()- start

            sub_sau_slow.append(end) #appending the time in the array

            
            
            start = t.time()
            t_singh = threshold_singh(image5, w, k=0.2) #  T(x,y) singh
            end = t.time()- start

            sub_sin.append(end) #appending the time in the array


            start = t.time()
            t_otsu = threshold_otsu(image6) #  T(x,y) singh
            end = t.time()- start

            sub_ots.append(end) #appending the time in the array
            
        time_nib.append(sub_nib)
        time_nib_slow.append(sub_nib_slow)
        time_sau.append(sub_sau)
        time_sau_slow.append(sub_sau_slow)
        time_sin.append(sub_sin)
        time_ots.append(sub_ots)

time_nib = (pd.DataFrame(time_nib)).transpose()
time_nib_slow = (pd.DataFrame(time_nib_slow)).transpose()
time_sau = (pd.DataFrame(time_sau)).transpose()
time_sau_slow = (pd.DataFrame(time_sau_slow)).transpose()
time_sin = (pd.DataFrame(time_sin)).transpose()
time_ots = (pd.DataFrame(time_ots)).transpose()

time_nib['mean'], time_nib['std'] = time_nib.mean(axis=1), time_nib.std(axis=1)
time_nib_slow['mean'], time_nib_slow['std'] = time_nib_slow.mean(axis=1), time_nib_slow.std(axis=1)
time_sau['mean'], time_sau['std'] = time_sau.mean(axis=1), time_sau.std(axis=1)
time_sau_slow['mean'], time_sau_slow['std'] = time_sau_slow.mean(axis=1), time_sau_slow.std(axis=1)
time_sin['mean'], time_sin['std'] = time_sin.mean(axis=1), time_sin.std(axis=1)
time_ots['mean'], time_ots['std'] = time_ots.mean(axis=1), time_ots.std(axis=1)

#%%

fig, ax = plt.subplots()

ax.errorbar(r, time_nib['mean'], xerr=None, yerr=time_nib['std'], 
            elinewidth=0.5, capsize=2, color='black', label='Niblack')
ax.errorbar(r, time_nib_slow['mean'], xerr=None, yerr=time_nib_slow['std'], 
            elinewidth=0.5, capsize=2, color='black', linestyle='dashed', label='Niblack slow')
ax.errorbar(r, time_sau['mean'], xerr=None, yerr=time_sau['std'], 
            elinewidth=0.5, capsize=2, color='red', label='Sauvola')
ax.errorbar(r, time_sau_slow['mean'], xerr=None, yerr=time_sau_slow['std'], 
            elinewidth=0.5, capsize=2, color='red', linestyle='dashed', label='Sauvola slow')
ax.errorbar(r, time_sin['mean'], xerr=None, yerr=time_sin['std'], 
            elinewidth=0.5, capsize=2, color='blue', label='Singh')
ax.errorbar(r, time_ots['mean'], xerr=None, yerr=time_sin['std'], 
            elinewidth=0.5, capsize=2, color='grey', label='Total (Otsu)')


ax.legend()
ax.set(title='N='+str(MAX)+', w='+str(min(r))+':'+str(max(r)))
ax.grid(True, color='gainsboro')
#ax.semilogy()
fig.suptitle( "Confronto tempi algoritmi di thresholding", fontsize=10)
ax.set_xlabel('w')
ax.set_ylabel('time (s)')
#ax.set_ylim(ymin=-0.1, ymax=1.3)


#title='time_comparison_(all linear)_N='+str(MAX)+'_w='+str(min(r))+'-'+str(max(r))+' (600x350px).png'
#fig.savefig(title, dpi=200, bbox_inches='tight')
