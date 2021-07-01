***SKIMAGE MOD for BINARIZATION ALGORITHMS COMPARISON***

################################## WHAT'S NEW

This mod will add to your skimage library the definition of a new local adaptive
thresholding algorithm (Singh et al. [1]), two new (slower) versions of already
implemented local adaptive thresholding algorithms (Niblack "slow", Sauvola "slow").

They are also provide some scripts to process images with these algorithms and to compare
the results obtained with different techniques.

################################## HOW TO

In all the scripts you must select your path to the directory in which the image are stored.

binarization_algorithms.py = script to process images with different algorithms
binarization_performances = quality metrics (using ground truth images) 
timing_comparison.py = processing time of the algorithms

In order to use these scripts you will have to modify your skimage library:
------> substitute the files in C:/user/.../skimage/filter
	with those (homonyms) contained in "package mods".


DIBCO 2009 --> directory in which you find test images with the corresponding GT.
images --> directory with images output.

[1] "A New Local Adaptive Thresholding Technique in Binarization",
	T.Romen Singh , Sudipta Roy, O.Imocha Singh, Tejmani Sinam, Kh.Manglem Singh
