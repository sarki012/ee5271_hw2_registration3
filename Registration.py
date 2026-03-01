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
    # The variables keypoints hold the x-and-y coordinates of key points
    # Descriptors are 128-dimensional feature descriptor vectors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    neighbors = NearestNeighbors(n_neighbors=2).fit(descriptors2)
    distances, indices = neighbors.kneighbors(descriptors1)

    x1, x2 = [], []
    for i in range(len(distances)):
        if distances[i][0] < 0.7 * distances[i][1]:
            x1.append(keypoints1[i].pt)
            x2.append(keypoints2[indices[i][0]].pt)
    return np.array(x1), np.array(x2)

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    """
    RANSAC algorithm for Affine Transformation fitting.
    """
    best_inliers_mask = None
    max_inlier_count = 0
    N_samples = 3 

    for i in range(ransac_iter):
        # 1. Select random samples
        idx = np.random.choice(len(x1), N_samples, replace=False)
        pts1 = x1[idx]
        pts2 = x2[idx]
        
        # 2. Fit Affine Model to 3 points
        # We want M such that pts2 = M * pts1 (in homogeneous coordinates)
        # X * A_T = Y  =>  A_T = inv(X) * Y
        X = np.hstack((pts1, np.ones((3, 1))))
        Y = pts2
        
        try:
            if np.linalg.matrix_rank(X) < 3:
                continue
            A_T = np.linalg.solve(X, Y)
            M = A_T.T # 2x3 matrix
        except np.linalg.LinAlgError:
            continue
        
        # 3. Count inliers
        # Transform all x1 points using M
        X_all = np.hstack((x1, np.ones((len(x1), 1))))
        x2_pred = X_all @ M.T
        
        diff = x2 - x2_pred
        distances = np.linalg.norm(diff, axis=1)
        
        inliers_mask = distances < ransac_thr
        num_inliers = np.sum(inliers_mask)
        
        if num_inliers > max_inlier_count:
            max_inlier_count = num_inliers
            best_inliers_mask = inliers_mask
    
    # Re-fit with all inliers
    if best_inliers_mask is not None and max_inlier_count >= 3:
        inlier_x1 = x1[best_inliers_mask]
        inlier_x2 = x2[best_inliers_mask]
        
        X = np.hstack((inlier_x1, np.ones((len(inlier_x1), 1))))
        Y = inlier_x2
        
        # Least squares
        res = np.linalg.lstsq(X, Y, rcond=None)
        A_T = res[0]
        A_affine = A_T.T
        
        A = np.vstack((A_affine, [0, 0, 1]))
    else:
        print("RANSAC failed.")
        A = np.eye(3)
        
    print(f"Number of inliers found: {max_inlier_count}")
    print("Affine Transformation Matrix (A):\n", A)
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

def get_gradient(im_dx, im_dy):
    # To do
    # Calculate the gradient magnitude and angle
    grad_mag = np.zeros(im_dx.shape)
    grad_angle = np.zeros(im_dx.shape)
    for j in range(im_dx.shape[0]):
        for i in range(im_dx.shape[1]):
            # The magnitude of the gradient = sqrt(im_dx^2 + im_dy^2)
            grad_mag[j, i] = math.sqrt(im_dx[j, i]**2 + im_dy[j, i]**2)
            # The angle of the gradient = the inverse tan(im_dy/im_dx)
            grad_angle[j, i] = math.atan2(im_dy[j, i], im_dx[j, i])
            if grad_angle[j, i] < 0:
                grad_angle[j, i] += math.pi    # Make the angle positive
    return grad_mag, grad_angle
def warp_image(img, A, output_size):
    # To do
    '''
    This function performs Inverse (Backward) Warping. It transforms the input image img into a new
    image of size output_size based on the affine transformation matrix A.
    '''
    # Example: Create a sample original image (e.g., a gradient)
    # In a real scenario, this would be your loaded image data (grayscale for simplicity)
    # Shape of image: (rows, cols)
    rows, cols = img.shape
  #  original_image = np.indices((rows, cols), dtype=float)[0] + np.indices((rows, cols), dtype=float)[1]

    # Define the points/coordinates of the original image grid
    # interpn expects 1D arrays for each dimension's coordinates
    '''
    Defines the grid axes (y and x coordinates) of the input image. These are used by the
    interpolator to know where the pixel data sits.
    '''
    points = (np.arange(rows), np.arange(cols))

    # Assuming the warped image has the same dimensions as the original for simplicity
    dest_rows, dest_cols = output_size
    # Create a grid of destination coordinates
    dest_coords_rows, dest_coords_cols = np.indices((dest_rows, dest_cols), dtype=float)
    # Flatten destination coordinates and represent as homogeneous coordinates (x, y, 1)
    dest_coords_flat = np.vstack((dest_coords_cols.flatten(), dest_coords_rows.flatten(), np.ones(dest_rows * dest_cols)))

    # Apply inverse transformation
    # The result will be source coordinates (x', y', w')
    src_coords_flat_h = A @ dest_coords_flat

    # Convert back from homogeneous coordinates if necessary (divide x', y' by w')
    src_coords_cols_flat = src_coords_flat_h[0] / src_coords_flat_h[2]
    src_coords_rows_flat = src_coords_flat_h[1] / src_coords_flat_h[2]

    # Combine source coordinates into the format interpn expects: (npoints, ndims)
    xi = np.vstack((src_coords_rows_flat, src_coords_cols_flat)).T
    # Interpolate values
    warped_values_flat = interpn(points, img, xi, method='linear', bounds_error=False, fill_value=0) #

    # Reshape the result back into the warped image's shape
    img_warped = warped_values_flat.reshape((dest_rows, dest_cols))
    return img_warped

def align_image(template, target, A):
    # To do
    '''
    A is the affine transformation matrix.
    Image alignment consists of moving, and possibly deforming, a template to minimize the
    difference between the template and an image.
    '''
    '''
    The following line of code is responsible for converting the standard
    3x3 affine transformation matrix A into a 6-element parameter vector p. 
    This conversion is crucial for the iterative Inverse Compositional 
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

              [ 1+p1   p2   p3 ]
    W(x; p) = [  p4   1+p5  p6 ]
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

    filterx, filtery = get_differential_filter()
    # Sobel filter, derivate in the x-direction
    filtered_imagex = filter_image(template, filterx)
    # Sobel filter, derivate in the y-direction
    filtered_imagey = filter_image(template, filtery)
    grad = get_gradient(filtered_imagex, filtered_imagey)

    # 1. Coordinate grid
    h, w = template.shape
    y, x = np.mgrid[0:h, 0:w]

    # 2. Analytical Jacobian of the affine warp W(x;p) w.r.t p
    # Shape: (h, w, 2, 6)
    # J = [[x, 0, y, 0, 1, 0],
    #      [0, x, 0, y, 0, 1]]
    jacobian = np.zeros((h, w, 2, 6))
    jacobian[:, :, 0, 0] = x
    jacobian[:, :, 0, 2] = y
    jacobian[:, :, 0, 4] = 1
    jacobian[:, :, 1, 1] = x
    jacobian[:, :, 1, 3] = y
    jacobian[:, :, 1, 5] = 1

    # 3. Compute Steepest Descent Images: VT * dW/dp
    # Gradient stack: (h, w, 2)
    grad_stack = np.stack((filtered_imagex, filtered_imagey), axis=2)
    # Einsum: multiply (h,w,2) by (h,w,2,6) summing over the coordinate dim (2) -> (h,w,6)
    steepest_descent_images = np.einsum('ijk,ijkl->ijl', grad_stack, jacobian)

    # 4. Compute Hessian
    H = np.einsum('ijk,ijl->kl', steepest_descent_images, steepest_descent_images)
    H_inv = np.linalg.inv(H)

    # 5. Optimization Loop
    A_refined = A.copy()
    errors = []
    max_iter = 75

    for i in range(max_iter):
        warped_target = warp_image(target, A_refined, (h, w))
        error_img = warped_target.astype(np.uint8) - template.astype(np.uint8)
        errors.append(np.mean(error_img**2))
        '''
        This vector represents how much the error would change if you tweaked each of the
        6 affine parameters. It is subsequently multiplied by the Inverse Hessian (H_inv)
        to determine the actual step size (delta_p) for the current iteration.
        '''
        sd_update = np.einsum('ijk,ij->k', steepest_descent_images, error_img)
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
    # 1. Initialize with the first frame
    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    
    A_list = []
    current_A = A

    for img in img_list:
        # 2. Track: Use previous A as guess for current frame
        # align_image returns (A, errors), so we must unpack it
        refined_A, errors = align_image(template, img, current_A)
        A_list.append(refined_A)
        # Update guess for next frame
        current_A = refined_A
        
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
    error_map = cv2.absdiff(template, img_warped_uint8)

    # Apply a colormap (JET) to visualize errors in color
    error_map_color = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)

    # Display the result
    cv2.namedWindow("Error Map", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Error Map", error_map_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   # A_refined, errors = align_image(template, target_list[0], A)
  #  visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)