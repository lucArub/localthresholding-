
import numpy as np
from skimage.filters import thresholding


def _only_mean(image, w):
    """Return local mean of each pixel using a
    neighborhood defined by a rectangular window size ``w``.
    The algorithm uses integral images to speedup computation.

    Parameters
    ----------
    image : ndarray
        Input image.
    w : int, or iterable of int
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).

    Returns
    -------
    m : ndarray of float, same shape as ``image``
        Local mean of the image.
    """

    if not isinstance(w, thresholding.Iterable):
        w = (w,) * image.ndim
    thresholding._validate_window_size(w)

    pad_width = tuple((k // 2 + 1, k // 2) for k in w)
    padded = np.pad(image.astype('float'), pad_width,
                    mode='reflect')
    #padded_sq = padded * padded

    integral = thresholding.integral_image(padded)
    #integral_sq = integral_image(padded_sq)

    kern = np.zeros(tuple(k + 1 for k in w))
    for indices in thresholding.itertools.product(*([[0, -1]] * image.ndim)):
        kern[indices] = (-1) ** (image.ndim % 2 != np.sum(indices) % 2)

    total_window_size = np.prod(w)
    sum_full = thresholding.ndi.correlate(integral, kern, mode='constant')
    m = thresholding.crop(sum_full, pad_width) / total_window_size
    #sum_sq_full = ndi.correlate(integral_sq, kern, mode='constant')
    #g2 = crop(sum_sq_full, pad_width) / total_window_size
    # Note: we use np.clip because g2 is not guaranteed to be greater than
    # m*m when floating point error is considered
    #s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return m#, s



def _mean_std_slow(image, w):

    """Return local mean of each pixel using a
    neighborhood defined by a rectangular window size ``w``.
    This version does not use integral images. ("naive" method used for test)

    Parameters
    ----------
    image : ndarray
        Input image.
    w : int, or iterable of int
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).

    Returns
    -------
    m : ndarray of float, same shape as ``image``
        Local mean of the image.
    """

    if not isinstance(w, thresholding.Iterable):
        w = (w,) * image.ndim
    thresholding._validate_window_size(w)

    pad_width = tuple((k // 2 + 1, k // 2) for k in w)
    padded = np.pad(image.astype('float'), pad_width,
                    mode='reflect')
    padded_sq = padded * padded
    
    kern = np.ones(tuple(k + 1 for k in w))

    total_window_size = np.prod(w)
    sum_full = thresholding.ndi.correlate(padded, kern, mode='constant')
    m = thresholding.crop(sum_full, pad_width) / total_window_size
    sum_sq_full = thresholding.ndi.correlate(padded_sq, kern, mode='constant')
    g2 = thresholding.crop(sum_sq_full, pad_width) / total_window_size
    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return m, s


def threshold_singh(image, window_size=15, k=0.2, r=None):
    """Applies Singh local threshold to an array.

    The threshold T is calculated for every pixel in the image using
    the following formula:

        T = m(x,y) * (1 + k * ((d(x,y) / 1 - d(x,y)) - 1))

    where m(x,y) and d(x,y) are the local mean and the local deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size wxw
    centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.
   
    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of the positive parameter k.

    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    References
    ----------
    [1] T Romen Singh, Sudipta Roy, O Imocha Singh, Tejmani Sinam, Kh Manglem Singh.
        "A New Local Adaptive Thresholding Technique in Binarization." IJCSI
        International Journal of Computer Science Issues. 2011; 8(6-2): 271-276.

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> t_singh = threshold_singh(image, window_size=15, k=0.2)
    >>> binary_image = image > t_singh
    """
    
    m = _only_mean(image, window_size)
    d = image - m
    
    return m * (1 + k * ((d / 1-d) - 1))



def threshold_niblack_slow(image, window_size=15, k=0.2):
    """Applies Niblack local threshold to an array.

    A threshold T is calculated for every pixel in the image using the
    following formula::

        T = m(x,y) - k * s(x,y)

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.

    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of parameter k in threshold formula.

    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    The Bradley threshold is a particular case of the Niblack
    one, being equivalent to

    >>> from skimage import data
    >>> image = data.page()
    >>> q = 1
    >>> threshold_image = threshold_niblack(image, k=0) * q

    for some value ``q``. By default, Bradley and Roth use ``q=1``.


    References
    ----------
    .. [1] W. Niblack, An introduction to Digital Image Processing,
           Prentice-Hall, 1986.
    .. [2] D. Bradley and G. Roth, "Adaptive thresholding using Integral
           Image", Journal of Graphics Tools 12(2), pp. 13-21, 2007.
           :DOI:`10.1080/2151237X.2007.10129236`

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> threshold_image = threshold_niblack(image, window_size=7, k=0.1)
    """
    m, s = _mean_std_slow(image, window_size)
    return m - k * s


def threshold_sauvola_slow(image, window_size=15, k=0.2, r=None):
    """Applies Sauvola local threshold to an array. Sauvola is a
    modification of Niblack technique.

    In the original method a threshold T is calculated for every pixel
    in the image using the following formula::

        T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))

    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.
    R is the maximum standard deviation of a greyscale image.

    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of the positive parameter k.
    r : float, optional
        Value of R, the dynamic range of standard deviation.
        If None, set to the half of the image dtype range.

    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.

    Notes
    -----
    This algorithm is originally designed for text recognition.

    References
    ----------
    .. [1] J. Sauvola and M. Pietikainen, "Adaptive document image
           binarization," Pattern Recognition 33(2),
           pp. 225-236, 2000.
           :DOI:`10.1016/S0031-3203(99)00055-2`

    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> t_sauvola = threshold_sauvola(image, window_size=15, k=0.2)
    >>> binary_image = image > t_sauvola
    """
    if r is None:
        imin, imax = thresholding.dtype_limits(image, clip_negative=False)
        r = 0.5 * (imax - imin)
    m, s = _mean_std_slow(image, window_size)
    return m * (1 + k * ((s / r) - 1))