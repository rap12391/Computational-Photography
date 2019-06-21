""" Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import cv2


def getImageCorners(image):
    """Return the x, y coordinates for the four corners bounding the input
    image and return them in the shape expected by the cv2.perspectiveTransform
    function. (The boundaries should completely encompass the image.)

    Parameters
    ----------
    image : numpy.ndarray
        Input can be a grayscale or color image

    Returns
    -------
    numpy.ndarray(dtype=np.float32)
        Array of shape (4, 1, 2).  The precision of the output is required
        for compatibility with the cv2.warpPerspective function.

    Notes
    -----
        (1) Review the documentation for cv2.perspectiveTransform (which will
        be used on the output of this function) to see the reason for the
        unintuitive shape of the output array.

        (2) When storing your corners, they must be in (X, Y) order -- keep
        this in mind and make SURE you get it right.
    """
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE

    dims = image.shape
    corners[0,0,:] = [0,0]
    corners[1,0,:] = [dims[1], 0]
    corners[2,0,:] = [dims[1], dims[0]]
    corners[3,0,:] = [0, dims[0]]

    return corners


def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    Notes
    -----
        (1) You will not be graded for this function.
    """
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # WRITE YOUR CODE HERE.

    for i in range(len(matches)):
        image_1_points[i] = np.float32(image_1_kp[matches[i].queryIdx].pt)
        image_2_points[i] = np.float32(image_2_kp[matches[i].trainIdx].pt)

    h, m = cv2.findHomography(image_1_points,image_2_points,cv2.RANSAC, 5.0)

    return h


def getBoundingCorners(corners_1, corners_2, homography):
    """Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Follow these steps:

        1. Use the homography to transform the perspective of the corners from
           image 1 (but NOT image 2) to get the location of the warped
           image corners.

        2. Get the boundaries in each dimension of the enclosing rectangle by
           finding the minimum x, maximum x, minimum y, and maximum y.

    Parameters
    ----------
    corners_1 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 1

    corners_2 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 2

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_min, y_min) -- the coordinates of the
        top left corner of the bounding rectangle of a canvas large enough to
        fit both images (leave them as floats)

    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_max, y_max) -- the coordinates of the
        bottom right corner of the bounding rectangle of a canvas large enough
        to fit both images (leave them as floats)

    Notes
    -----
        (1) The inputs may be either color or grayscale, but they will never
        be mixed; both images will either be color, or both will be grayscale.

        (2) Python functions can return multiple values by listing them
        separated by commas.

        Ex.
            def foo():
                return [], [], []
    """
    # WRITE YOUR CODE HERE

    corners_transformed = cv2.perspectiveTransform(corners_1, homography)

    x_min = min(min(corners_transformed[:,0,0]), min(corners_2[:,0,0]))
    x_max = max(max(corners_transformed[:,0,0]), max(corners_2[:,0,0]))

    y_min = min(min(corners_transformed[:,0,1]), min(corners_2[:,0,1]))
    y_max = max(max(corners_transformed[:,0,1]), max(corners_2[:,0,1]))


    return np.array([x_min,y_min], dtype=np.float64), np.array([x_max,y_max], dtype=np.float64)


def warpCanvas(image, homography, min_xy, max_xy):
    """Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Follow these steps:

        1. Create a translation matrix (numpy.ndarray) that will shift
           the image by x_min and y_min. This looks like this:

            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]]

        2. Compute the dot product of your translation matrix and the
           homography in order to obtain the homography matrix with a
           translation.

        NOTE: Matrix multiplication (dot product) is not the same thing
              as the * operator (which performs element-wise multiplication).
              See Numpy documentation for details.

        3. Call cv2.warpPerspective() and pass in image 1, the combined
           translation/homography transform matrix, and a vector describing
           the dimensions of a canvas that will fit both images.

        NOTE: cv2.warpPerspective() is touchy about the type of the output
              shape argument, which should be an integer.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between two sequential
        images in a panorama sequence

    min_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the top left corner of a
        canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    max_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the bottom right corner of
        a canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    Returns
    -------
    numpy.ndarray(dtype=image.dtype)
        An array containing the warped input image embedded in a canvas
        large enough to join with the next image in the panorama; the output
        type should match the input type (following the convention of
        cv2.warpPerspective)

    Notes
    -----
        (1) You must explain the reason for multiplying x_min and y_min
        by negative 1 in your writeup.
    """
    # canvas_size properly encodes the size parameter for cv2.warpPerspective,
    # which requires a tuple of ints to specify size, or else it may throw
    # a warning/error, or fail silently
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))
    # WRITE YOUR CODE HERE

    t_matrix = np.array([[1, 0, -min_xy[0]],
                         [0, 1, -min_xy[1]],
                         [0, 0, 1]])

    h_matrix = np.dot(t_matrix,homography)

    warped_img = cv2.warpPerspective(image, h_matrix, canvas_size)

    return warped_img


def blendImagePair(image_1, image_2, num_matches):
    """This function takes two images as input and fits them onto a single
    canvas by performing a homography warp on image_1 so that the keypoints
    in image_1 aligns with the matched keypoints in image_2.

    **************************************************************************

        You MUST replace the basic insertion blend provided here to earn
                         credit for this function.

       The most common implementation is to use alpha blending to take the
       average between the images for the pixels that overlap, but you are
                    encouraged to use other approaches.

           Be creative -- good blending is the primary way to earn
                  Above & Beyond credit on this assignment.

    **************************************************************************

    Parameters
    ----------
    image_1 : numpy.ndarray
        A grayscale or color image

    image_2 : numpy.ndarray
        A grayscale or color image

    num_matches : int
        The number of keypoint matches to find between the input images

    Returns:
    ----------
    numpy.ndarray
        An array containing both input images on a single canvas

    Notes
    -----
        (1) This function is not graded by the autograder. It will be scored
        manually by the TAs.

        (2) The inputs may be either color or grayscale, but they will never be
        mixed; both images will either be color, or both will be grayscale.

        (3) You can modify this function however you see fit -- e.g., change
        input parameters, return values, etc. -- to develop your blending
        process.
    """
    kp1, kp2, matches = findMatchesBetweenImages(
        image_1, image_2, num_matches)
    homography = findHomography(kp1, kp2, matches)
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
    output_image = warpCanvas(image_1, homography, min_xy, max_xy)
    # WRITE YOUR CODE HERE - REPLACE THIS WITH YOUR BLENDING CODE

    temp = np.array(output_image)
    min_xy = min_xy.astype(np.int)
    output_image[-min_xy[1]:-min_xy[1] + image_2.shape[0],
                 -min_xy[0]:-min_xy[0] + image_2.shape[1]] = image_2

    yy = -min_xy[1] + image_2.shape[0]
    xx = -min_xy[0] + image_1.shape[1] + image_2.shape[1] - output_image.shape[1]

    i, j = 0, 0
    for r in range(-min_xy[1], yy):
        for c in range(-min_xy[0], xx):
            if temp[r,c].any() != 0:
                alpha = 0.45
                output_image[r, c] =  (alpha)*temp[r, c] +  (1 - alpha)*image_2[j,i]
            i += 1
        i = 0
        j += 1

    return output_image
    # END OF FUNCTION



# ########################################################################################
# def generatingKernel(a):
#     """Return a 5x5 generating kernel based on an input parameter (i.e., a
#     square "5-tap" filter.)
#
#     Parameters
#     ----------
#     a : float
#         The kernel generating parameter in the range [0, 1] used to generate a
#         5-tap filter kernel.
#
#     Returns
#     -------
#     output : numpy.ndarray
#         A 5x5 array containing the generated kernel
#     """
#     # DO NOT CHANGE THE CODE IN THIS FUNCTION
#     kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
#     return np.outer(kernel, kernel)
#
#
# def reduce_layer(image, kernel=generatingKernel(0.4)):
#     """Convolve the input image with a generating kernel and then reduce its
#     width and height each by a factor of two.
#
#     For grading purposes, it is important that you use a reflected border
#     (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid
#     region (i.e., the convolution operation should return an image of the same
#     shape as the input) for the convolution. Subsampling must include the first
#     row and column, skip the second, etc.
#
#     Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):
#
#                           fefghg
#         abcd     Pad      babcdc   Convolve   ZYXW   Subsample   ZX
#         efgh   ------->   fefghg   -------->  VUTS   -------->   RP
#         ijkl    BORDER    jijklk     keep     RQPO               JH
#         mnop   REFLECT    nmnopo     valid    NMLK
#         qrst              rqrsts              JIHG
#                           nmnopo
#
#     A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
#     Please consult the lectures for a more in-depth discussion of how to
#     tackle the reduce function.
#
#     Parameters
#     ----------
#     image : numpy.ndarray
#         A grayscale image of shape (r, c). The array may have any data type
#         (e.g., np.uint8, np.float64, etc.)
#
#     kernel : numpy.ndarray (Optional)
#         A kernel of shape (N, N). The array may have any data type (e.g.,
#         np.uint8, np.float64, etc.)
#
#     Returns
#     -------
#     numpy.ndarray(dtype=np.float64)
#         An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
#         5x7, the output will be 3x4.
#     """
#
#     # WRITE YOUR CODE HERE.
#     image = image.astype(np.float64)
#
#     conv = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_REFLECT101)
#
#     sub = conv[::2,::2]
#
#     return sub
#
#
# def expand_layer(image, kernel=generatingKernel(0.4)):
#     """Upsample the image to double the row and column dimensions, and then
#     convolve it with a generating kernel.
#
#     Upsampling the image means that every other row and every other column will
#     have a value of zero (which is why we apply the convolution after). For
#     grading purposes, it is important that you use a reflected border (i.e.,
#     padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid region
#     (i.e., the convolution operation should return an image of the same
#     shape as the input) for the convolution.
#
#     Finally, multiply your output image by a factor of 4 in order to scale it
#     back up. If you do not do this (and you should try it out without that)
#     you will see that your images darken as you apply the convolution.
#     You must explain why this happens in your submission PDF.
#
#     Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):
#
#                                           000000
#              Upsample   A0B0     Pad      0A0B0B   Convolve   zyxw
#         AB   ------->   0000   ------->   000000   ------->   vuts
#         CD              C0D0    BORDER    0C0D0D     keep     rqpo
#         EF              0000   REFLECT    000000    valid     nmlk
#                         E0F0              0E0F0F              jihg
#                         0000              000000              fedc
#                                           0E0F00
#
#                 NOTE: Remember to multiply the output by 4.
#
#     A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
#     Please consult the lectures for a more in-depth discussion of how to
#     tackle the expand function.
#
#     Parameters
#     ----------
#     image : numpy.ndarray
#         A grayscale image of shape (r, c). The array may have any data
#         type (e.g., np.uint8, np.float64, etc.)
#
#     kernel : numpy.ndarray (Optional)
#         A kernel of shape (N, N). The array may have any data
#         type (e.g., np.uint8, np.float64, etc.)
#
#     Returns
#     -------
#     numpy.ndarray(dtype=np.float64)
#         An image of shape (2*r, 2*c). For instance, if the input is 3x4, then
#         the output will be 6x8.
#     """
#
#     # WRITE YOUR CODE HERE.
#     size = image.shape
#
#     #for 2D image
#     if len(size)==2:
#
#         up_sample = np.zeros((image.shape[0]*2, image.shape[1]*2))
#     #for 3D image
#     elif len(size)==3:
#
#         up_sample = np.zeros((image.shape[0] * 2, image.shape[1] * 2, image.shape[2]))
#
#     up_sample[::2, ::2] = image
#
#     conv = cv2.filter2D(up_sample, -1, kernel, borderType=cv2.BORDER_REFLECT_101) * 4
#
#     return conv
#
#
# def gaussPyramid(image, levels):
#     """Construct a pyramid from the image by reducing it by the number of
#     levels specified by the input.
#
#     You must use your reduce_layer() function to generate the pyramid.
#
#     Parameters
#     ----------
#     image : numpy.ndarray
#         An image of dimension (r, c).
#
#     levels : int
#         A positive integer that specifies the number of reductions to perform.
#         For example, levels=0 should return a list containing just the input
#         image; levels = 1 should perform one reduction and return a list with
#         two images. In general, len(output) = levels + 1.
#
#     Returns
#     -------
#     list<numpy.ndarray(dtype=np.float)>
#         A list of arrays of dtype np.float. The first element of the list
#         (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
#         layer 1 of the pyramid (image reduced once), etc.
#     """
#
#     # WRITE YOUR CODE HERE.
#     #intializin the pyramid - first level in the image itself
#     image = image.astype(np.float64)
#
#     g_pyramid = [image]
#
#     #Using reduce operation to blur the previous level
#
#     for i in range(levels):
#
#         g_pyramid.append(reduce_layer(g_pyramid[-1]))
#
#     return g_pyramid
#
# def laplPyramid(gaussPyr):
#     """Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
#     pyramid will have the same number of levels as the input.
#
#     You must use your expand_layer() function to generate the pyramid. The
#     Gaussian Pyramid that is passed in is the output of your gaussPyramid
#     function.
#
#     Parameters
#     ----------
#     gaussPyr : list<numpy.ndarray(dtype=np.float)>
#         A Gaussian Pyramid (as returned by your gaussPyramid function), which
#         is a list of numpy.ndarray items.
#
#     Returns
#     -------
#     list<numpy.ndarray(dtype=np.float)>
#         A laplacian pyramid of the same size as gaussPyr. This pyramid should
#         be represented in the same way as guassPyr, as a list of arrays. Every
#         element of the list now corresponds to a layer of the laplacian
#         pyramid, containing the difference between two layers of the gaussian
#         pyramid.
#
#         NOTE: The last element of output should be identical to the last layer
#               of the input pyramid since it cannot be subtracted anymore.
#
#     Notes
#     -----
#         (1) Sometimes the size of the expanded image will be larger than the
#         given layer. You should crop the expanded image to match in shape with
#         the given layer. If you do not do this, you will get a 'ValueError:
#         operands could not be broadcast together' because you can't subtract
#         differently sized matrices.
#
#         For example, if my layer is of size 5x7, reducing and expanding will
#         result in an image of size 6x8. In this case, crop the expanded layer
#         to 5x7.
#     """
#
#     # WRITE YOUR CODE HERE.
#     def crop(image, size):
#         dims = image.shape
#         h = dims[0]
#         w=dims[1]
#         if h != size[0]:
#             image = image[:size[0]-h, :]
#         if w != size[1]:
#             image = image[:, :size[1]-w]
#         return image
#
#
#     gaussPyr = gaussPyr[::-1]
#
#     lapPyr = [gaussPyr[0]]
#
#     for i, layer in enumerate(gaussPyr[:-1]):
#
#         layer = expand_layer(layer)
#
#         layer = crop(layer, gaussPyr[i+1].shape)
#
#         lapPyr.append(gaussPyr[i+1] - layer)
#
#     return lapPyr[::-1]
#
# def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
#     """Blend two laplacian pyramids by weighting them with a gaussian mask.
#
#     You should return a laplacian pyramid that is of the same dimensions as the
#     input pyramids. Every layer should be an alpha blend of the corresponding
#     layers of the input pyramids, weighted by the gaussian mask.
#
#     Therefore, pixels where current_mask == 1 should be taken completely from
#     the white image, and pixels where current_mask == 0 should be taken
#     completely from the black image.
#
#     (The variables `current_mask`, `white_image`, and `black_image` refer to
#     the images from each layer of the pyramids. This computation must be
#     performed for every layer of the pyramid.)
#
#     Parameters
#     ----------
#     laplPyrWhite : list<numpy.ndarray(dtype=np.float)>
#         A laplacian pyramid of an image constructed by your laplPyramid
#         function.
#
#     laplPyrBlack : list<numpy.ndarray(dtype=np.float)>
#         A laplacian pyramid of another image constructed by your laplPyramid
#         function.
#
#     gaussPyrMask : list<numpy.ndarray(dtype=np.float)>
#         A gaussian pyramid of the mask. Each value should be in the range
#         [0, 1].
#
#     Returns
#     -------
#     list<numpy.ndarray(dtype=np.float)>
#         A list containing the blended layers of the two laplacian pyramids
#
#     Notes
#     -----
#         (1) The input pyramids will always have the same number of levels.
#         Furthermore, each layer is guaranteed to have the same shape as
#         previous levels.
#     """
#
#     # WRITE YOUR CODE HERE.
#     blended = np.array(laplPyrWhite)*np.array(gaussPyrMask) + np.array(laplPyrBlack) * (1-np.array(gaussPyrMask))
#     return blended
#
#
# def collapse(pyramid):
#     """Collapse an input pyramid.
#
#     Approach this problem as follows: start at the smallest layer of the
#     pyramid (at the end of the pyramid list). Expand the smallest layer and
#     add it to the second to smallest layer. Then, expand the second to
#     smallest layer, and continue the process until you are at the largest
#     image. This is your result.
#
#     Parameters
#     ----------
#     pyramid : list<numpy.ndarray(dtype=np.float)>
#         A list of numpy.ndarray images. You can assume the input is taken
#         from blend() or laplPyramid().
#
#     Returns
#     -------
#     numpy.ndarray(dtype=np.float)
#         An image of the same shape as the base layer of the pyramid.
#
#     Notes
#     -----
#         (1) Sometimes expand will return an image that is larger than the next
#         layer. In this case, you should crop the expanded image down to the
#         size of the next layer. Look into numpy slicing to do this easily.
#
#         For example, expanding a layer of size 3x4 will result in an image of
#         size 6x8. If the next layer is of size 5x7, crop the expanded image
#         to size 5x7.
#     """
#
#     # WRITE YOUR CODE HERE.
#     def crop(image, size):
#         dims = image.shape
#         h = dims[0]
#         w=dims[1]
#         if h != size[0]:
#             image = image[:size[0]-h, :]
#         if w != size[1]:
#             image = image[:, :size[1]-w]
#         return image
#
#     added = pyramid[-1]
#
#     for i in range(2,len(pyramid)+1):
#
#         expanded = expand_layer(added)
#
#         added = crop(expanded, pyramid[-i].shape) + pyramid[-i]
#
#     return added

# if __name__ == '__main__':
#

