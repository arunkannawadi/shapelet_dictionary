import galsim

def get_gaussian_weight_image(img_array):
    """
    Given an arbitrary galaxy image, obtain the image of the best-fit elliptical Gaussian.
    This is aimed to obtain the weight for the pixels in the galaxy postage stamp.

    @params img_array       Input image, either as a numpy array or as a galsim.Image instance

    @returns                Image of the best-fit elliptical Gaussian, 
                            in the same format as the input.
    """
    
    flag = 1

    ## Interpret the array as a GalSim Image instance
    if not isinstance(img_array, galsim.Image):
        img = galsim.Image(img_array)

    ## Calculate the best-fit elliptical Gaussian parameters
    try:
        mom = img.FindAdaptiveMom()
    except RuntimeError as error:
        print("RuntimeError:{0}".format(error))
        flag = -1
        return [0], flag

    x0, y0 = mom.moments_centroid.x, mom.moments_centroid.y
    sigma = mom.moments_sigma
    shape = mom.observed_shape

    ## Create the Gaussian profile
    gauss = galsim.Gaussian(sigma=sigma).shear(shape)

    ## Create the weight image, with same dimensions as the given Image
    weight = galsim.Image(bounds=img.bounds)
    offset = galsim.PositionD(x0,y0)-weight.trueCenter()
    weight = gauss.drawImage(image=weight, offset=offset, scale=1., method='no_pixel')

    ## Return the weight image in the same format as the input
    if isinstance(img_array, galsim.Image):
        return weight,flag
    else:
        return weight.array,flag



