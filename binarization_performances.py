# -*- coding: utf-8 -*-


"""

BINARIZATION PERFORMANCE INDEXES:

True positive: pixel identified as ink in both the segmented image and
               the ground truth image.
True negative: pixel identified as background in both the segmented imag
               and the ground truth image.
False positive: pixel identified as a ink in the segmented image
                but is background in the ground truth image.
False negative: pixel identified as background in the segmented image
                but is ink in the ground truth image.

Using these percentages, the following metrics were calculated:

Precision: How many pixels detected are actually ink?           [TP/(TP+FP)]

Sensitivity/True Positive Rate: What fraction of the total 
                                ink pixels were detected?       [TP/(TP+FN)]
                                
Specificity/True Negative Rate: What fraction of the total background
                                pixels were detected?           [TN/(TN+FP)]
                                
Accuracy: percentage of pixels which are correctly classified   [(TN+TP)/(TN+TP+FP+FN)]

F1-score: harmonic mean between precision and sensitivity       [(2*p*s/(p+s))]

"""

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import glob
import numpy as np
import skimage.io
import skimage.viewer
from skimage.filters import (threshold_niblack,
                             threshold_sauvola,
                             threshold_singh,
                             threshold_otsu
                             )

im_path = 'yourpath/*.*'  #image path
GT_path = 'yourpath/*.*'  #ground_truth path

GT = glob.glob(GT_path)
im = glob.glob(im_path)
image = skimage.io.imread(fname=im[0],as_gray=True)
image_GT = skimage.img_as_bool(skimage.io.imread(fname=GT[0],as_gray=True))

size = np.size(image_GT)
        
Rw = range(3,70,6)
Rh = range(1,60,4)

fig_nib, ax_nib = plt.subplots()
fig_sau, ax_sau = plt.subplots()
fig_sin, ax_sin = plt.subplots()

accy_nib = []
accy_sau = []
accy_sin = []
MSE_nib = []
MSE_sau = []
MSE_sin = []
PSNR_nib = []
PSNR_sau = []
PSNR_sin = []
F1_nib = []
F1_sau = []
F1_sin = []
 
for w in Rw:
    print(w)
    sub_accy_nib = []
    sub_accy_sau = []
    sub_accy_sin = []
    sub_MSE_nib = []
    sub_MSE_sau = []
    sub_MSE_sin = []
    sub_PSNR_nib = []
    sub_PSNR_sau = []
    sub_PSNR_sin = []
    sub_F1_nib = []
    sub_F1_sau = []
    sub_F1_sin = []

    for h in Rh:
        bench = np.zeros(shape=(11,3)) # bench 0 = true positive                                          
                                       # bench 1 = true negative
                                       # bench 2 = false positive
                                       # bench 3 = false negative
                                       # bench 4 = precision
                                       # bench 5 = true positive rate (sensitivity)
                                       # bench 6 = true negative rate (specitivity)
                                       # bench 7 = accuracy
                                       # bench 8 = F1-score
                                       # bench 9 = MSE
                                       # bench 10 = PSNR

        #Computing thresholding
        image_nib = image
        t_nib = threshold_niblack(image_nib, window_size=w, k=h/100)
        image_nib = image_nib > t_nib
        bench[0][0] = np.sum(np.logical_and(np.logical_not(image_nib) , np.logical_not(image_GT)))
        bench[1][0] = np.sum(np.logical_and(image_nib , image_GT))
        bench[2][0] = np.sum(np.logical_and(np.logical_not(image_nib) , image_GT))
        bench[3][0] = np.sum(np.logical_and(image_nib , np.logical_not(image_GT)))
        bench[9][0] = skimage.metrics.mean_squared_error(image_nib, image_GT)        
        bench[10][0] = skimage.metrics.peak_signal_noise_ratio(image_nib, image_GT)        

        image_sau = image        
        t_sau = threshold_sauvola(image_sau, window_size=w, k=h/100)
        image_sau = image_sau > t_sau
        bench[0][1] = np.sum(np.logical_and(np.logical_not(image_sau) , np.logical_not(image_GT)))
        bench[1][1] = np.sum(np.logical_and(image_sau , image_GT))
        bench[2][1] = np.sum(np.logical_and(np.logical_not(image_sau) , image_GT))
        bench[3][1] = np.sum(np.logical_and(image_sau , np.logical_not(image_GT)))
        bench[9][1] = skimage.metrics.mean_squared_error(image_sau, image_GT)        
        bench[10][1] = skimage.metrics.peak_signal_noise_ratio(image_sau, image_GT)        
        
        image_sin = image
        t_sin = threshold_singh(image_sin, window_size=w, k=h/100)
        image_sin = image_sin > t_sin
        bench[0][2] = np.sum(np.logical_and(np.logical_not(image_sin) , np.logical_not(image_GT)))
        bench[1][2] = np.sum(np.logical_and(image_sin , image_GT))
        bench[2][2] = np.sum(np.logical_and(np.logical_not(image_sin) , image_GT))
        bench[3][2] = np.sum(np.logical_and(image_sin , np.logical_not(image_GT)))
        bench[9][2] = skimage.metrics.mean_squared_error(image_sin, image_GT)
        bench[10][2] = skimage.metrics.peak_signal_noise_ratio(image_sin, image_GT)        

        
        #Other indexes 
        bench[4]=bench[0]/(bench[0]+bench[2])
        bench[5]=bench[0]/(bench[0]+bench[3])
        bench[6]=bench[1]/(bench[1]+bench[2])
        bench[7]=(bench[0]+bench[1])/(bench[0]+bench[1]+bench[2]+bench[3])
        b=1
        bench[8]=((1+b*b)*bench[4]*bench[5])/(b*b*bench[4]+bench[5])
        
  
        sub_accy_nib.append(bench[7][0])
        sub_accy_sau.append(bench[7][1])
        sub_accy_sin.append(bench[7][2])
        sub_F1_nib.append(bench[8][0]) 
        sub_F1_sau.append(bench[8][1]) 
        sub_F1_sin.append(bench[8][2])
        sub_MSE_nib.append(bench[9][0])
        sub_MSE_sau.append(bench[9][1])
        sub_MSE_sin.append(bench[9][2])
        sub_PSNR_nib.append(bench[10][0])
        sub_PSNR_sau.append(bench[10][1])
        sub_PSNR_sin.append(bench[10][2])


    accy_nib.append(np.asarray(sub_accy_nib))
    accy_sau.append(np.asarray(sub_accy_sau))
    accy_sin.append(np.asarray(sub_accy_sin))
    MSE_nib.append(np.asarray(sub_MSE_nib))
    MSE_sau.append(np.asarray(sub_MSE_sau))
    MSE_sin.append(np.asarray(sub_MSE_sin))
    PSNR_nib.append(np.asarray(sub_PSNR_nib))
    PSNR_sau.append(np.asarray(sub_PSNR_sau))
    PSNR_sin.append(np.asarray(sub_PSNR_sin))
    F1_nib.append(np.asarray(sub_F1_nib))
    F1_sau.append(np.asarray(sub_F1_sau))
    F1_sin.append(np.asarray(sub_F1_sin))


#%%
Rh, Rw = np.meshgrid(Rh, Rw)
#EXAMPLE: PLOTTING PSNR SAUVOLA ALGORITHM
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(Rh, Rw, np.asarray(PSNR_sau), rstride=1, cstride=1, cmap=cm.Reds)

ax.set(title='PSNR maximization')
fig.suptitle( 'Sauvola algorithm', fontsize=10)
ax.set_xlabel('k(x10'+'\xb2'+')')
ax.set_ylabel('w')
title = 'PSNR_sauvola.png'
fig.savefig(title, dpi=300, bbox_inches='tight')
