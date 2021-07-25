#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import glob     
import sys    
import skimage.io  
import skimage.viewer
from skimage.filters import (threshold_niblack,
                             threshold_sauvola,
                             threshold_otsu
                             )
from filter_mod import threshold_singh

img_number = 0 # variable that counts the number of images in a directory
#path = 'yourpath/*.*' # path/*.* in order to slied all the images 
# bias

#w = 25
#bias = 0.2

script_name, path, alg, w, bias = sys.argv
path2 = path + "/*.*"

print(type(bias))


for file in glob.glob(path2):
  
  
  
  img_number+=1
   
   #  image1 --> threshold with niblack  image2--> threshold with sauvola 
   #  image3 --> threshold with singh  image4 --> threshold with otsu
  image1 = skimage.io.imread(fname=file,as_gray=True) # as_gray=True no RGB color
  image2 = skimage.io.imread(fname=file,as_gray=True)
  image3 = skimage.io.imread(fname=file,as_gray=True)
  image4 = skimage.io.imread(fname=file,as_gray=True)
            
  

  if alg == 'nib':

    print("ok") 
    t_niblack = threshold_niblack(image1, window_size=int(w), k=float(bias)) #  T(x,y) niblack
    image1[image1 < t_niblack]=0 # threshold  
    image1[image1 > t_niblack]=255       
    skimage.io.imsave(path+str(img_number)+'nib'+'.jpg' ,image1 )


  if alg == 'sau':

    t_sauvola = threshold_sauvola(image2, window_size=int(w), k=float(bias)) # T(x,y) sauvola
    image2[image2 < t_sauvola]=0 # threshold 
    image2[image2 > t_sauvola]=255       
    skimage.io.imsave(path+str(img_number)+'sau'+'.jpg' ,image2 )

  if alg == 'singh':

    t_singh = threshold_singh(image3, window_size=int(w), k=float(bias)) # T(x,y) singh
    image3[image3 < t_singh]=0 # threshold 
    image3[image3 > t_singh]=255
    skimage.io.imsave(path+str(img_number)+'sin'+'.jpg' ,image3 )


  if alg == 'otsu':
    
    t_otsu = threshold_otsu(image4) # T global otsu
    image4[image4 < t_otsu]=0
    image4[image4 > t_otsu]=255
    skimage.io.imsave(path+str(img_number)+'ots'+'.jpg' ,image4 )
  