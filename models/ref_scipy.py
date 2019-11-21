# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:38:04 2019

@author: No Name
"""

def denoise_wavelet(image, sigma=None, wavelet='db1', mode='soft',
                    wavelet_levels=None, multichannel=False,
                    convert2ycbcr=False, method='BayesShrink',
                    rescale_sigma=None):
    """Perform wavelet denoising on an image.
    Parameters
    ----------
    image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    sigma : float or list, optional
        The noise standard deviation used when computing the wavelet detail
        coefficient threshold(s). When None (default), the noise standard
        deviation is estimated via the method in [2]_.
    wavelet : string, optional
        The type of wavelet to perform and can be any of the options
        ``pywt.wavelist`` outputs. The default is `'db1'`. For example,
        ``wavelet`` can be any of ``{'db2', 'haar', 'sym9'}`` and many more.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels.
    multichannel : bool, optional
        Apply wavelet denoising separately for each channel (where channels
        correspond to the final axis of the array).
    convert2ycbcr : bool, optional
        If True and multichannel True, do the wavelet denoising in the YCbCr
        colorspace instead of the RGB color space. This typically results in
        better performance for RGB images.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. Defaults to "BayesShrink".
    rescale_sigma : bool or None, optional
        If False, no rescaling of the user-provided ``sigma`` will be
        performed. The default of ``None`` rescales sigma appropriately if the
        image is rescaled internally. A ``DeprecationWarning`` is raised to
        warn the user about this new behaviour. This warning can be avoided
        by setting ``rescale_sigma=True``.
        .. versionadded:: 0.16
           ``rescale_sigma`` was introduced in 0.16
    Returns
    -------
    out : ndarray
        Denoised image.
    Notes
    -----
    The wavelet domain is a sparse representation of the image, and can be
    thought of similarly to the frequency domain of the Fourier transform.
    Sparse representations have most values zero or near-zero and truly random
    noise is (usually) represented by many small values in the wavelet domain.
    Setting all values below some threshold to 0 reduces the noise in the
    image, but larger thresholds also decrease the detail present in the image.
    If the input is 3D, this function performs wavelet denoising on each color
    plane separately.
    .. versionchanged:: 0.16
       For floating point inputs, the original input range is maintained and
       there is no clipping applied to the output. Other input types will be
       converted to a floating point value in the range [-1, 1] or [0, 1]
       depending on the input image range. Unless ``rescale_sigma = False``,
       any internal rescaling applied to the ``image`` will also be applied
       to ``sigma`` to maintain the same relative amplitude.
    Many wavelet coefficient thresholding approaches have been proposed. By
    default, ``denoise_wavelet`` applies BayesShrink, which is an adaptive
    thresholding method that computes separate thresholds for each wavelet
    sub-band as described in [1]_.
    If ``method == "VisuShrink"``, a single "universal threshold" is applied to
    all wavelet detail coefficients as described in [2]_. This threshold
    is designed to remove all Gaussian noise at a given ``sigma`` with high
    probability, but tends to produce images that appear overly smooth.
    Although any of the wavelets from ``PyWavelets`` can be selected, the
    thresholding methods assume an orthogonal wavelet transform and may not
    choose the threshold appropriately for biorthogonal wavelets. Orthogonal
    wavelets are desirable because white noise in the input remains white noise
    in the subbands. Biorthogonal wavelets lead to colored noise in the
    subbands. Additionally, the orthogonal wavelets in PyWavelets are
    orthonormal so that noise variance in the subbands remains identical to the
    noise variance of the input. Example orthogonal wavelets are the Daubechies
    (e.g. 'db2') or symmlet (e.g. 'sym2') families.
    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           :DOI:`10.1109/83.862633`
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           :DOI:`10.1093/biomet/81.3.425`
    Examples
    --------
    >>> from skimage import color, data
    >>> img = img_as_float(data.astronaut())
    >>> img = color.rgb2gray(img)
    >>> img += 0.1 * np.random.randn(*img.shape)
    >>> img = np.clip(img, 0, 1)
    >>> denoised_img = denoise_wavelet(img, sigma=0.1, rescale_sigma=True)
    """
    if method not in ["BayesShrink", "VisuShrink"]:
        raise ValueError(
            ('Invalid method: {}. The currently supported methods are '
             '"BayesShrink" and "VisuShrink"').format(method))

    # floating-point inputs are not rescaled, so don't clip their output.
    clip_output = image.dtype.kind != 'f'

    if convert2ycbcr and not multichannel:
        raise ValueError("convert2ycbcr requires multichannel == True")

    if rescale_sigma is None:
        msg = (
            "As of scikit-image 0.16, automated rescaling of sigma to match "
            "any internal rescaling of the image is performed. Setting "
            "rescale_sigma to False, will disable this new behaviour. To "
            "avoid this warning the user should explicitly set rescale_sigma "
            "to True or False."
        )
        warn(msg, DeprecationWarning)
        rescale_sigma = True
    image, sigma = _scale_sigma_and_image_consistently(image,
                                                       sigma,
                                                       multichannel,
                                                       rescale_sigma)
    if multichannel:
        if convert2ycbcr:
            out = color.rgb2ycbcr(image)
            # convert user-supplied sigmas to the new colorspace as well
            if rescale_sigma:
                sigma = _rescale_sigma_rgb2ycbcr(sigma)
            for i in range(3):
                # renormalizing this color channel to live in [0, 1]
                _min, _max = out[..., i].min(), out[..., i].max()
                scale_factor = _max - _min
                if scale_factor == 0:
                    # skip any channel containing only zeros!
                    continue
                channel = out[..., i] - _min
                channel /= scale_factor
                sigma_channel = sigma[i]
                if sigma_channel is not None:
                    sigma_channel /= scale_factor
                out[..., i] = denoise_wavelet(channel,
                                              wavelet=wavelet,
                                              method=method,
                                              sigma=sigma_channel,
                                              mode=mode,
                                              wavelet_levels=wavelet_levels,
                                              rescale_sigma=rescale_sigma)
                out[..., i] = out[..., i] * scale_factor
                out[..., i] += _min
            out = color.ycbcr2rgb(out)
        else:
            out = np.empty_like(image)
            for c in range(image.shape[-1]):
                out[..., c] = _wavelet_threshold(image[..., c],
                                                 wavelet=wavelet,
                                                 method=method,
                                                 sigma=sigma[c], mode=mode,
                                                 wavelet_levels=wavelet_levels)
    else:
        out = _wavelet_threshold(image, wavelet=wavelet, method=method,
                                 sigma=sigma, mode=mode,
                                 wavelet_levels=wavelet_levels)

    if clip_output:
        clip_range = (-1, 1) if image.min() < 0 else (0, 1)
        out = np.clip(out, *clip_range, out=out)
    return out