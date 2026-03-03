# EE 5271 Robot Vision Homework 2
# Erik Sarkinen
# Student ID: 3854563
# 2/23/26

import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
from scipy.interpolate import interpn
import sympy
import math



def find_match(img1, img2):
    # To do
    # Create a SIFT (Scale-Invariant Feature Transform) object
    sift = cv2.SIFT_create()   
    '''
    sift.detectAndCompute: locates keypoints and calculates feature descriptors, which are
    128-dimensional vector
    
    Finds keypoints in both images and calculates 
    a mathematical description ("descriptor") for the area around each point.
    Keypoints: The (x, y) coordinates of corners, blobs, or distinct features.
    Descriptors: 128-dimensional vectors that are a histogram of image gradients.
    '''
    keypts1, descript1 = sift.detectAndCompute(img1, None)
    keypts2, descript2 = sift.detectAndCompute(img2, None)

    '''
    find the 2 closest matches in img2 for every feature in img1.
    neighbors.fit(descriptors2): Prepares the descriptors from the second image for searching.
    neighbors.kneighbors(descriptors1): For every descriptor in img1, it finds the two most 
    similar descriptors in img2 based on Euclidean distance.
    distance is a 2D array with dimensions (Number of Keypoints in Image 1, 2).
    '''
    neighbors = NearestNeighbors(n_neighbors=2).fit(descript2)
    distance, x_y = neighbors.kneighbors(descript1)

    '''
    The following code compares the distance of the best match (distance[i][0]) to the 
    second-best match (distance[i][1]). If the best match is much closer 
    (less than 0.7*distance) than the second-best, it is a match. If the distances are 
    similar (ratio >= 0.75), it means the feature in img1 is similar to multiple features 
    in img2, so there is no match.
    Euclidean distance refers to the similarity between the feature descriptors
    distance[i][0] is the Euclidean distance to the closest match in Image 2.
    distance[i][1] is the Euclidean distance to the second closest match in Image 2.
    These values represent how "similar" the features are. A smaller distance means a 
    better match.
    '''
    x1 = []
    x2 = []
    for i in range(len(distance)):
        if distance[i][0] < 0.75 * distance[i][1]:
            x1.append(keypts1[i].pt)    # match found, extracts just the location data   
            x2.append(keypts2[x_y[i][0]].pt)
    x_array1 = np.array(x1)
    x_array2 = np.array(x2)
    return x_array1, x_array2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    '''
    RANSAC algorithm for Affine Transformation fitting. 
    '''
    inliers = []
    distance = []
    num_inliers = 0
    best_inliers_mask = None
    max_inlier_count = 0
    num_samples = 3 
    '''
    The goal is to find the best 2D transformation (rotation, scale, translation, shear) 
    that aligns the points in x1 with the points in x2, even if many of the initial matches 
    are incorrect (outliers).
    '''
    for i in range(ransac_iter):
        # Select random samples
        index = np.random.choice(len(x1), num_samples, replace=False)
        points1 = x1[index]
        points2 = x2[index]
        
        '''
        Fit Affine Model to 3 points
        We want AFFINE_TEMP such that points2 = AFFINE_TEMP * points1 (in homogeneous coordinates)
        X_ARRAY * AFFINE_TRANSPOSE = Y_ARRAY  =>  AFFINE_TRANSPOSE = inv(X_ARRAY) * Y_ARRAY
        It calculates a temporary 2x3 affine matrix AFFINE_TEMP that perfectly maps the 3 
        randomly chosen source points (pts1) to their corresponding target points (pts2). This 
        is done by solving the linear system Y_ARRAY = AFFINE_TEMP * X_ARRAY.
        '''
        X_ARRAY = np.hstack((points1, np.ones((3, 1))))     # Inserts a column of 1's (x, y, 1), homogenous coordinates
        Y_ARRAY = points2
        try:
            if np.linalg.matrix_rank(X_ARRAY) < 3:    # Makes sure that all 3 points aren't on the same line
                continue
            AFFINE_TRANSPOSE = np.linalg.solve(X_ARRAY, Y_ARRAY)
            AFFINE_TEMP = AFFINE_TRANSPOSE.T # 2x3 matrix
        except np.linalg.LinAlgError:
            continue
        
        '''
        Transform all x1 points using M
        Once a candidate transformation (AFFINE_TEMP) is calculated from the 3 random points, 
        the code tests how well it works for all the other points. X_all becomes an (N, 3) 
        matrix where every row is (x, y, 1).
        X_all: The source points in homogeneous coordinates Nx3.
        AFFINE_TEMP: The 2x3 affine matrix calculated from the random sample.
        x2_predicted is an Nx2 matrix containing the predicted coordinates in the second image.
        '''
        X_all = np.hstack((x1, np.ones((len(x1), 1))))
        x2_predicted = X_all @ AFFINE_TEMP.T
        
        '''
        x2: where the points are. x2_predicted: where the points should be.
        If diff is small, count as an inlier.
        '''
        diff = x2 - x2_predicted
        distance = np.linalg.norm(diff, axis=1)
        
        '''
        Points with an error smaller than the threshold are marked as inliers. These are the "good" 
        matches that agree with the current model.
        distance: This is a 1D NumPy array containing the error for each point. The error is the 
        pixel distance between where a point from the first image is predicted to be in the second 
        image (using the temporary model AFFINE_TEMP) and where it actually is. 
        inliers is a mask with 
        '''
        inliers = distance < ransac_thr     # inliers is a boolean mask, True if distance < ransac_thr
        num_inliers = np.sum(inliers)

        '''
        It keeps track of the model that has the highest number of inliers. This is assumed to be 
        the correct model. num_inliers and best_inliers are boolean masks with True at the index
        where distance is < ransac_thr.
        '''  
        if num_inliers > max_inlier_count:
            max_inlier_count = num_inliers
            best_inliers = inliers
    
    # Re-fit with all inliers
    if best_inliers is not None and max_inlier_count >= 3:
        '''
        It extracts the actual coordinates of all the "good" points (inlier_x1, inlier_x2) using 
        the saved mask best_inliers.
        '''
        inlier_x1 = x1[best_inliers]
        inlier_x2 = x2[best_inliers]
        
        X_ARRAY = np.hstack((inlier_x1, np.ones((len(inlier_x1), 1))))
        Y_ARRAY = inlier_x2
        
        '''
        It takes all the inliers found (which could be hundreds of points) and performs a 
        Least Squares fit. Unlike solve (which hits 3 points exactly), lstsq finds the 
        transformation that minimizes the average error across all valid points. This produces a 
        much more accurate and stable matrix.
        '''
        res = np.linalg.lstsq(X_ARRAY, Y_ARRAY, rcond=None)
        AFFINE_TRANSPOSE = res[0]
        A_AFFINE = AFFINE_TRANSPOSE.T
        '''
        It formats the result into a standard 3x3 homogeneous affine matrix 
        (adding [0, 0, 1] at the bottom) so it can be used for image warping later.
        '''
        A = np.vstack((A_AFFINE, [0, 0, 1]))
    else:
        print("RANSAC failed.")
        A = np.eye(3)       # Return identity matrix.
        
    print(f"Number of inliers found: {max_inlier_count}")
    print("Affine Transformation Matrix A:\n", A)
    return A



def get_differential_filter():
    # To do
    # Flip the Sobel kernels along the x-and-y axes
    filter_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    filter_x_flipped = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
    filter_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]             
    filter_y_flipped = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    # Unpack the shape into height and width
    height, width = im.shape
    im_filtered = np.zeros((height, width))
    # Perform a convolution with the Sobel kernel to get the derivative. Iterate through
    # the 3x3 kernel using indices k and l, multiply each element of the filter by the
    # corresponding element on the image, sum the result and set equal to im_filtered[j , i]
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            sum_val = 0
            for k in range(3):
                for l in range(3):
                    sum_val += im[j + k - 1, i + l - 1]*filter[k][l]
            im_filtered[j, i] = sum_val
    return im_filtered

def warp_image(img, A, output_size):
    # To do
    '''
    This function performs Inverse (Backward) Warping. It transforms the input image img into a new
    image of size output_size based on the affine transformation matrix A.
    '''
    h_dest, w_dest = output_size
    h_src, w_src = img.shape

    # Define the points/coordinates of the original image grid
    # interpn expects 1D arrays for each dimension's coordinates
    points = (np.arange(h_src), np.arange(w_src))

    # Create a grid of coordinates for the destination image
    dest_coords_rows, dest_coords_cols = np.indices((h_dest, w_dest), dtype=float)
    
    # Flatten and create homogeneous coordinates (x, y, 1)
    dest_coords_flat = np.vstack((dest_coords_cols.flatten(), dest_coords_rows.flatten(), np.ones(h_dest * w_dest)))

    # Apply the affine transformation to find corresponding source coordinates
    src_coords_flat = A @ dest_coords_flat

    # Combine source coordinates into the format interpn expects: (npoints, ndims) -> (y, x)
    xi = np.vstack((src_coords_flat[1], src_coords_flat[0])).T

    # Interpolate values
    warped_values_flat = interpn(points, img, xi, method='linear', bounds_error=False, fill_value=0)

    img_warped = warped_values_flat.reshape((h_dest, w_dest))
    return img_warped

def align_image(template, target, A):
    # To do
    '''
    A is the affine transformation matrix.
    Image alignment consists of moving/deforming a template to minimize the
    difference between the template and an image.
    
    The following code converts the standard 3x3 affine transformation matrix A into a 6-element 
    parameter vector p. This conversion is crucial for the iterative Inverse Compositional 
    alignment algorithm.
    1. Two Representations of an Affine Warp
    There are different ways to represent the same 2D affine transformation.
    The Standard Affine Matrix A calculates a standard 3x3 affine transformation matrix A in 
    homogeneous coordinates. It looks like this:     
    
        [ a b tx ]
    A = [ c d ty ]
        [ 0 0 1  ]

    Where a, b, c, d represent scaling, rotation, and shear, and tx, ty 
    represent translation.
    b) The Parameterized Warp W(x; p):
    The Inverse Compositional algorithm works by iteratively updating a set 
    of warp parameters, p. For an affine transformation, this is typically a 
    6-element vector p = [p1, p2, p3, p4, p5, p6]. These parameters define a 
    warp matrix W that is centered around the identity transformation 
    (i.e., when p=0, W is the identity matrix). This specific parameterization 
    is defined as:

              [ 1+p1   p3   p5 ]
    W(x; p) = [  p2   1+p4  p6 ]
              [  0     0    1  ]

    2. Mapping A to p
    The following line of code initializes the parameter vector p for 
    the iterative algorithm using the initial guess A that was found using 
    feature matching. To do this, it equates the two matrix forms and solves 
    for the p parameters:

    A = W(x; p)

    [ A[0,0]  A[0,1]  A[0,2] ]   [ 1+p1   p3   p5 ]
    [ A[1,0]  A[1,1]  A[1,2] ] = [  p2   1+p4  p6 ]
    [   0       0       1    ]   [  0     0    1  ]

    By comparing the elements of these two matrices, we can derive the values for each 
    of the 6 parameter in p:

    A[0,0] = 1 + p1 => p1 = A[0,0] - 1
    A[1,0] = p2 => p2 = A[1,0]
    A[0,1] = p3 => p3 = A[0,1]
    A[1,1] = 1 + p4 => p4 = A[1,1] - 1
    A[0,2] = p5 => p5 = A[0,2] (translation in x)
    A[1,2] = p6 => p6 = A[1,2] (translation in y)
    3. The Code Implementation
    The Python code
    p = np.array([A[0, 0] - 1, A[1, 0], A[0, 1], A[1, 1] - 1, A[0, 2], A[1, 2]]) 
    directly implements these equations to create the initial parameter vector 
    p0 from the input matrix A. The algorithm will then start from this p0 
    and iteratively find small updates Δp to refine the alignment.
    '''

    # 1. Coordinate grid
    h, w = template.shape
    y, x = np.mgrid[0:h, 0:w]
    p = np.array([A[0, 0] - 1, A[1, 0], A[0, 1], A[1, 1] - 1, A[0, 2], A[1, 2]])
    '''
    Analytical Jacobian of the affine warp W(x;p) w.r.t p
    x' = (1 + p0)x + p2y + p4
    y' = p1x + (1 + p3)y + p5
    Shape: (h, w, 2, 6)
    J = [[x, 0, y, 0, 1, 0],
        [0, x, 0, y, 0, 1]]
    '''
    jacobian = np.zeros((h, w, 2, 6))
    jacobian[:, :, 0, 0] = (1 + p[0])*x
    jacobian[:, :, 0, 1] = 0
    jacobian[:, :, 0, 2] = p[2]*y
    jacobian[:, :, 0, 3] = 0
    jacobian[:, :, 0, 4] = p[4]
    jacobian[:, :, 0, 5] = 0
    jacobian[:, :, 1, 0] = 0
    jacobian[:, :, 1, 1] = p[1]*x
    jacobian[:, :, 1, 2] = 0
    jacobian[:, :, 1, 3] = (1 + p[3])*y
    jacobian[:, :, 1, 4] = 0
    jacobian[:, :, 1, 5] = p[5]

    # Compute Steepest Descent Images: VT * dW/dp
    filterx, filtery = get_differential_filter()
    # Sobel filter, derivate in the x-direction
    filtered_imagex = filter_image(template, filterx)
    # Sobel filter, derivate in the y-direction
    filtered_imagey = filter_image(template, filtery)

    # (df/dx, df/dy)
    grad_dx_dy = np.stack((filtered_imagex, filtered_imagey), axis=2)
    
    # Compute steepest descent iterating over all of the pixels
    steepest_grad = np.zeros((h, w, 6))
    for j in range(h):
        for i in range(w):
            steepest_grad[j, i] = np.matmul(grad_dx_dy[j, i], jacobian[j, i, :, :])

    # Compute Hessian
    # Flatten SDI to (N, 6) where N is total pixels. This allows standard matrix math.
    sdi_flat = steepest_grad.reshape(-1, 6)
    # Compute H = Sum(SDI.T * SDI) -> (6, N) @ (N, 6) -> (6, 6)
    H = sdi_flat.T @ sdi_flat
    H_inv = np.linalg.inv(H)

    # 5. Optimization Loop
    A_refined = A.copy()
    errors = []
    max_iter = 75

    for i in range(max_iter):
        warped_target = warp_image(target, A_refined, (h, w))
        error_img = warped_target.astype(np.float32) - template.astype(np.float32)
        errors.append(np.mean(error_img**2))
        '''
        This vector represents how much the error would change if you tweaked each of the
        6 affine parameters. It is subsequently multiplied by the Inverse Hessian (H_inv)
        to determine the actual step size (delta_p) for the current iteration.
        '''
        # Compute steepest descent update without einsum
        # Flatten error image to (N,) and dot product with flattened SDI
        sd_update = sdi_flat.T @ error_img.reshape(-1)
        delta_p = H_inv @ sd_update

        # Construct Delta_M from delta_p
        delta_M = np.array([
            [1 + delta_p[0], delta_p[2], delta_p[4]],
            [delta_p[1], 1 + delta_p[3], delta_p[5]],
            [0, 0, 1]
        ])
        
        A_refined = A_refined @ np.linalg.inv(delta_M)

        if np.linalg.norm(delta_p) < 9e-2:
            print(f"Delta_p = {delta_p}.")
            break

    return A_refined, np.array(errors)


def track_multi_frames(template, img_list):
    ransac_thr = 5.0
    ransac_iter = 1000    
    A_list = []
    template2 = template.copy()
    i = 0
    for img in img_list:
        # 1. Initialize with the first frame
        x1, x2 = find_match(template, img)
        current_A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
        # 2. Track: Use previous A as guess for current frame
        # align_image returns (A, errors), so we must unpack it
        refined_A, errors = align_image(template2, img, current_A)
        A_list.append(refined_A)
        # Update guess for next frame
        current_A = refined_A
        # Update template
        template2 = warp_image(template, refined_A, template2.shape)
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 5.0
    ransac_iter = 1000
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, target_list[0].shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
# 1. Ensure images are the same size and data type
    # Resizing the warped image to match the template dimensions if they differ
    if template.shape != img_warped.shape:
        img_warped = cv2.resize(img_warped, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
    # 3. Compute the absolute difference
    # Ensure images are the same data type (uint8) for cv2.absdiff
    img_warped_uint8 = img_warped.astype(np.uint8)
    # Compare the template with the warped image (aligned to template)
  #  error_map = cv2.absdiff(template, img_warped_uint8)

   # A_refined, errors = align_image(template, target_list[0], A)
  #  visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)