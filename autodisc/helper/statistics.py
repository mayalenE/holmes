import collections
import copy
import math
import warnings
from cmath import rect, phase
from math import radians, degrees

import numpy as np

import autodisc as ad

Moments = collections.namedtuple('Moments', ['y_avg', 'x_avg',
                                             'm00', 'm10', 'm01', 'm11', 'm20', 'm02', 'm21', 'm12', 'm22', 'm30', 'm31', 'm13', 'm03', 'm40', 'm04',
                                             'mu11', 'mu20', 'mu02', 'mu30', 'mu03', 'mu21', 'mu12', 'mu22', 'mu31', 'mu13', 'mu40', 'mu04',
                                             'eta11', 'eta20', 'eta02', 'eta30', 'eta03', 'eta21', 'eta12', 'eta22', 'eta31', 'eta13', 'eta40', 'eta04',
                                             'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7', 'hu8',
                                             'flusser9', 'flusser10', 'flusser11', 'flusser12', 'flusser13'])


def calc_image_moments(image):
    '''
    Calculates the image moments for an image.

    For more information see:
     - https://en.wikipedia.org/wiki/Image_moment
     - http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/SHUTLER3/node1.html

    The code is based on the Javascript implementation of Lenia by Bert Chan.
    The code had to be adapted, because for the original code the coordinates are (x,y), whereas here they are (y,x).

    :param image: 2d gray scale image in form of a numpy array.
    :return: Namedtupel with the different moments.
    '''

    eps = 0.00001

    size_y = image.shape[0]
    size_x = image.shape[1]

    x_grid, y_grid = np.meshgrid(range(size_x), range(size_y))

    y_power1_image = y_grid * image
    y_power2_image = y_grid * y_power1_image
    y_power3_image = y_grid * y_power2_image

    x_power1_image = x_grid * image
    x_power2_image = x_grid * x_power1_image
    x_power3_image = x_grid * x_power2_image

    # raw moments: m_qp
    m00 = np.sum(image)
    m10 = np.sum(y_power1_image)
    m01 = np.sum(x_power1_image)
    m11 = np.sum(y_grid * x_grid * image)
    m20 = np.sum(y_power2_image)
    m02 = np.sum(x_power2_image)
    m21 = np.sum(y_power2_image * x_grid)
    m12 = np.sum(x_power2_image * y_grid)
    m22 = np.sum(x_power2_image * y_grid * y_grid)
    m30 = np.sum(y_power3_image)
    m31 = np.sum(y_power3_image * x_grid)
    m13 = np.sum(y_grid * x_power3_image)
    m03 = np.sum(x_power3_image)
    m40 = np.sum(y_power3_image * y_grid)
    m04 = np.sum(x_power3_image * x_grid)

    # mY and mX describe the position of the centroid of the image
    # if there is no activation, then use the center position
    if m00 == 0:
        mY = (image.shape[0]-1) / 2
        mX = (image.shape[1]-1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00

    # in the case of very small activation (m00 ~ 0) the position becomes infinity, also then use the center position
    if mY == float('inf'):
        mY = (image.shape[0]-1) / 2
    if mX == float('inf'):
        mX = (image.shape[1]-1) / 2

    # calculate the central moments
    X2 = mX*mX
    X3 = X2*mX
    Y2 = mY*mY
    Y3 = Y2*mY
    XY = mX*mY

    mu11 = m11 - mY * m01
    mu20 = m20 - mY * m10
    mu02 = m02 - mX * m01
    mu30 = m30 - 3 * mY * m20 + 2 * Y2 * m10
    mu03 = m03 - 3 * mX * m02 + 2 * X2 * m01
    mu21 = m21 - 2 * mY * m11 - mX * m20 + 2 * Y2 * m01
    mu12 = m12 - 2 * mX * m11 - mY * m02 + 2 * X2 * m10
    mu22 = m22 - 2 * mX * m21 + X2 * m20 - 2 * mY * m12 + 4 * XY * m11 - 2 * mY * X2 * m10 + Y2 * m02 - 2 * Y2 * mX * m01 + Y2 * X2 * m00
    mu31 = m31 - mX * m30 + 3 * mY * (mX * m20 - m21) + 3 * Y2 * (m11 - mX * m10) + Y3 * (mX * m00 - m01)
    mu13 = m13 - mY * m03 + 3 * mX * (mY * m02 - m12) + 3 * X2 * (m11 - mY * m01) + X3 * (mY * m00 - m10)
    mu40 = m40 - 4 * mY * m30 + 6 * Y2 * m20 - 4 * Y3 * m10 + Y2 * Y2 * m00
    mu04 = m04 - 4 * mX * m03 + 6 * X2 * m02 - 4 * X3 * m01 + X2 * X2 * m00

    # Moment invariants: scale invariant
    if m00 < eps:
        eta11 = 0
        eta20 = 0
        eta02 = 0
        eta30 = 0
        eta03 = 0
        eta21 = 0
        eta12 = 0
        eta22 = 0
        eta31 = 0
        eta13 = 0
        eta40 = 0
        eta04 = 0
    else:
        m2 = m00 * m00
        mA = m00 * m00 * math.sqrt(m00)
        m3 = m00 * m00 * m00
        eta11 = mu11 / m2
        eta20 = mu20 / m2
        eta02 = mu02 / m2
        eta30 = mu30 / mA
        eta03 = mu03 / mA
        eta21 = mu21 / mA
        eta12 = mu12 / mA
        eta22 = mu22 / m3
        eta31 = mu31 / m3
        eta13 = mu13 / m3
        eta40 = mu40 / m3
        eta04 = mu04 / m3

    # Moment invariants: rotation invariants
    Z = 2 * eta11
    A = eta20 + eta02
    B = eta20 - eta02
    C = eta30 + eta12
    D = eta30 - eta12
    E = eta03 + eta21
    F = eta03 - eta21
    G = eta30 - 3*eta12
    H = 3*eta21 - eta03
    Y = 2*eta22
    I = eta40 + eta04
    J = eta40 - eta04
    K = eta31 + eta13
    L = eta31 - eta13
    CC = C*C
    EE = E*E
    CC_EE = CC - EE
    CC_EE3 = CC - 3*EE
    CC3_EE = 3*CC - EE
    CE = C*E
    DF = D*F
    M = I - 3 * Y
    t1 = CC_EE * CC_EE - 4 * CE * CE
    t2 = 4 * CE * CC_EE

    # invariants by Hu
    hu1 = A
    hu2 = B*B + Z*Z
    hu3 = G*G + H*H
    hu4 = CC + EE
    hu5 = G*C*CC_EE3 + H*E*CC3_EE
    hu6 = B*CC_EE + 2*Z*CE
    hu7 = H * C * CC_EE3 - G * E * CC3_EE
    hu8 = Z*CC_EE/2 - B*CE

    # extra invariants by Flusser
    flusser9 = I + Y
    flusser10 = J*CC_EE + 4*L*DF
    flusser11 = -2*K*CC_EE - 2*J*DF
    flusser12 = 4*L*t2 + M*t1
    flusser13 = -4*L*t1 + M*t2

    result = Moments(y_avg=mY, x_avg=mX,
                     m00=m00, m10=m10, m01=m01, m11=m11, m20=m20, m02=m02, m21=m21, m12=m12, m22=m22, m30=m30, m31=m31, m13=m13, m03=m03, m40=m40, m04=m04,
                     mu11=mu11, mu20=mu20, mu02=mu02, mu30=mu30, mu03=mu03, mu21=mu21, mu12=mu12, mu22=mu22, mu31=mu31, mu13=mu13, mu40=mu40, mu04=mu04,
                     eta11=eta11, eta20=eta20, eta02=eta02, eta30=eta30, eta03=eta03, eta21=eta21, eta12=eta12,  eta22=eta22, eta31=eta31, eta13=eta13, eta40=eta40, eta04=eta04,
                     hu1=hu1, hu2=hu2, hu3=hu3, hu4=hu4, hu5=hu5, hu6=hu6, hu7=hu7, hu8=hu8,
                     flusser9=flusser9, flusser10=flusser10, flusser11=flusser11, flusser12=flusser12, flusser13=flusser13)

    return result


def mean_over_angles_degrees(angles):
    '''Calculates the mean over angles that are given in degrees.'''
    if len(angles) == 0:
        return np.nan
    else:
        return degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles)))


def nan_mean_over_angles_degrees(angles):
    '''Calculates the mean over angles that are given in degrees. Ignores nan values.'''
    np_angles = np.array(angles)
    return mean_over_angles_degrees(np_angles[~np.isnan(np_angles)])


def std_over_angles_degrees(angles):
    '''Calculates the mean over angles that are given in degrees.'''
    # TODO: calc std over angles
    raise NotImplementedError('Computing the standard deviation over angles is not implemented!')


# def is_finite_object(img, tol=0.2, is_continuous_image=True, non_continuous_segmented_img=None):
#     '''
#     Checks if the activated segments do not touch the sides and represent therefore a finite object.
#     If the image has a border with non-activated cells around.
#     '''
#
#     if not is_continuous_image:
#         # only check if there is a non-alive border for non-continuous images
#
#         if np.all(img[0, :] < tol) and \
#            np.all(img[:, 0] < tol) and \
#            np.all(img[-1, :] < tol) and \
#            np.all(img[:, -1] < tol):
#
#             is_finite = True
#         else:
#             is_finite = False
#
#     else:
#         # need segmented images based, both continuous and non-continuous
#         if non_continuous_segmented_img is None:
#             non_continuous_segmented_img, _, _ = calc_binary_segments(img, tol=tol, is_eight_neighbors=False, is_continuous_image=False)
#
#         # check (north, south) and (east, west) borders for infinite objects
#
#         is_finite = True
#
#         for j in range(img.shape[1]):
#
#             if img[0, j] >= tol and img[-1, j] >= tol:
#                 # only check for infinite active area if both borders are active (thus they are connected)
#
#                 # check if they are also connected not using the borders, if so then it is a infinite object
#                 if non_continuous_segmented_img[0,j] == non_continuous_segmented_img[-1,j]:
#                     is_finite = False
#                     break
#
#         if is_finite:
#
#             for i in range(img.shape[0]):
#
#                 if img[i, 0] >= tol and img[i, -1] >= tol:
#                     # only check for infinite active area if both borders are active (thus they are connected)
#
#                     # check if they are also connected not using the borders, if so then it is a infinite object
#                     if non_continuous_segmented_img[i, 0] == non_continuous_segmented_img[i, -1]:
#                         is_finite = False
#                         break
#
#     return is_finite


def calc_binary_segments(img, tol=0.2, is_eight_neighbors=False, is_continuous_image=True):
    '''
    Identifies the different segments in a black and white image.
    Differentiates segments binary based on the tol - paramter (default = 0.2).
    '''

    # helper function
    def compare_current_seg_to_pos(compare_to_pos, cur_seg, seg_count, num_of_elements):
        '''
        Compares the current position to the compare_to_pos.
        If they are different segments, but have the same type (alive or dead), then the current segment is merged with the one it is compared to.
        '''

        compare_to_seg = seg_img[compare_to_pos]
        compare_to_type = img_binary[compare_to_pos]

        if compare_to_seg != cur_seg and compare_to_type == cur_seg_type:

            # select which segment is merged with which, and which one is deleted
            if cur_seg < 0:
                # if the current segment is new (-1) then merge it with the existing
                old_seg = cur_seg
                new_seg = compare_to_seg
            else:
                # otherwise, merge segment with the higher number with one that has the lower number, and delete the higher segment
                old_seg = max(cur_seg, compare_to_seg)
                new_seg = min(cur_seg, compare_to_seg)

            # set all other segment_elements of the current type also to this segment
            seg_img[seg_img == old_seg] = new_seg

            # erase the existing type, so that segment numbers have no jumps
            if old_seg >= 0:
                # if not the maximum segment yet, then reduce the segment number of all segments afterwards, and

                num_of_elements[new_seg] = num_of_elements[new_seg] + num_of_elements[old_seg]
                del (num_of_elements[old_seg])

                del (segment_types[old_seg])
                if old_seg < seg_count - 1:
                    for higher_seg in range(old_seg + 1, seg_count):
                        seg_img[seg_img == higher_seg] = higher_seg - 1


                seg_count = seg_count - 1

            cur_seg = new_seg

        return cur_seg, seg_count, seg_img, num_of_elements

    # segments
    seg_img = np.ones(img.shape, dtype=int) * -2

    seg_count = 0
    cur_seg = None
    cur_seg_type = None
    # saves tuples for each segment with (number of elements, type) where the type is binary for alive, not alive
    segment_types = []
    num_of_elements = []

    img_binary = img >= tol

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            is_alive = img_binary[i,j]

            if i == 0 and j == 0:
                cur_seg = -1
            else:
                # check if segment changed

                # collect all points to which the current point should be compared
                compare_to_positions = []

                if is_alive is not cur_seg_type or j == 0:
                    # change of segment
                    cur_seg_type = is_alive
                    cur_seg = -1 # either a new one or part of a neighbor

                # check if the current segment is part of a previous segment above, to the side
                if i > 0:
                    # check north

                    # if same type but different segment_number, then set current segment number to previous one

                    if is_eight_neighbors:

                        if j > 0:
                            compare_to_positions.append((i - 1, j - 1))
                        elif is_continuous_image:
                            compare_to_positions.append((i - 1, img.shape[1] - 1))

                        if j < img.shape[1]-1:
                            compare_to_positions.append((i - 1, j + 1))
                        elif is_continuous_image:
                            compare_to_positions.append((i - 1, 0))

                    compare_to_positions.append((i-1, j))

                if is_continuous_image:

                    # border case 1: east border --> check west
                    if j >= img.shape[1]-1:

                        if is_eight_neighbors:
                            if i > 0:
                                compare_to_positions.append((i-1, 0))

                        compare_to_positions.append((i, 0))

                    # border case 2: south border --> check north
                    if i >= img.shape[0]-1:

                        if is_eight_neighbors:
                            if j == 0:
                                compare_to_positions.append((0, img.shape[1] - 1))
                            else:
                                compare_to_positions.append((0, j - 1))

                            if j >= img.shape[1] - 1:
                                compare_to_positions.append((0, 0))
                            else:
                                compare_to_positions.append((0, j + 1))

                        if i > 0:
                            compare_to_positions.append((i - 1, 0))

                        compare_to_positions.append((0, j))

                for pos in compare_to_positions:
                    (cur_seg, seg_count, seg_img, num_of_elements) = compare_current_seg_to_pos(pos, cur_seg, seg_count, num_of_elements)

            cur_seg_type = is_alive

            if cur_seg == -1:
                # new segment that has to be added
                cur_seg = seg_count
                seg_count = seg_count + 1
                segment_types.append(cur_seg_type)

                num_of_elements.append(0)

            num_of_elements[cur_seg] = num_of_elements[cur_seg] + 1

            seg_img[i,j] = cur_seg

    return seg_img, segment_types, num_of_elements


def calc_space_distribution_bins(vectors, bin_config, ignore_out_of_range_values=True):

    # create section borders
    bins_per_dim = []
    for dim_config in bin_config:
         bins_per_dim.append(np.linspace(dim_config[0], dim_config[1], num=dim_config[2]+1))

    # identify point for every vector
    count_per_section = collections.defaultdict(int)

    for vector in vectors:

        section = []

        # check each dimension
        for dim_idx in range(len(vector)):

            # identify at which section in de fined grid the value falls
            idxs = np.where(bins_per_dim[dim_idx] > vector[dim_idx])[0]

            if len(idxs) == 0:
                # value is larger than upper grid border
                #warnings.warn('A Vector with value {} is outside the defined grid for dimension {}.'.format(vector[dim_idx], dim_idx))

                if ignore_out_of_range_values:
                    section = None
                    break
                else:
                    section_idx = len(bins_per_dim[dim_idx])

            elif idxs[0] == 0:
                # value is smaller than lower grid border
                #warnings.warn('A Vector with value {} is outside the defined grid for dimension {}.'.format(vector[dim_idx], dim_idx))

                if ignore_out_of_range_values:
                    section = None
                    break
                else:
                    section_idx = -1
            else:
                section_idx = idxs[0]-1

            section.append(section_idx)

        if section is not None:
            section = tuple(section)
            count_per_section[section] += 1

    # sort by number of exvaluations and generate description
    count_per_section_values = np.array(list(count_per_section.values()))

    sorted_idxs = np.flipud(np.argsort(count_per_section_values))

    n_points = count_per_section_values[sorted_idxs]
    section_idxs = np.array(list(count_per_section.keys()))[sorted_idxs]

    lower_borders = []
    upper_borders = []

    for section_idx in section_idxs:
        cur_lower_borders = []
        cur_upper_borders = []

        for dim_idx, bin_idx in enumerate(section_idx):

            if bin_idx <= -1:
                cur_lower_borders.append(np.nan)
                cur_upper_borders.append(bins_per_dim[dim_idx][0])
            elif bin_idx >= len(bins_per_dim[dim_idx]):
                cur_lower_borders.append(bins_per_dim[dim_idx][-1])
                cur_upper_borders.append(np.nan)
            else:
                cur_lower_borders.append(bins_per_dim[dim_idx][bin_idx])
                cur_upper_borders.append(bins_per_dim[dim_idx][bin_idx + 1])

        lower_borders.append(cur_lower_borders)
        upper_borders.append(cur_upper_borders)

    bins_descr = dict()
    bins_descr['n_points'] = n_points
    bins_descr['lower_borders'] = lower_borders
    bins_descr['upper_borders'] = upper_borders

    return bins_descr


def calc_space_distribution_histogram(vectors, bin_config, ignore_out_of_range_values=True):

    bins_descr = calc_space_distribution_bins(vectors, bin_config, ignore_out_of_range_values)


    H = np.array(bins_descr['n_points'])

    # define bins for the counts
    count_histogram_bins = list(range(0, int(np.max(H)+1)+1))

    # count how many sections had a certain count
    count_hist, count_hist_edges = np.histogram(H, bins=count_histogram_bins)
    count_hist = count_hist.tolist() # make to list because for bin goalspaces can numpy not encode the number correctly, but pure python can

    # have to set zero
    if ignore_out_of_range_values:
        n_sections = [dim_config[2] for dim_config in bin_config]
    else:
        n_sections = [dim_config[2] + 2 for dim_config in bin_config]

    total_num_of_sections = 1
    for n_dim_sections in n_sections:
        total_num_of_sections = total_num_of_sections * n_dim_sections

    num_zero_sections = total_num_of_sections - len(H)

    count_hist[0] = num_zero_sections

    return count_hist, count_histogram_bins


def calc_active_binary_segments(img, tol=0.2, r=1, is_continuous_image=True):

    r = int(r)

    # prepare the distances that have to be taken into account to say an active point is part of another active point
    max_dist_per_col = np.zeros(int(r)*2 + 1, dtype=int)
    dist_per_row = np.zeros(int(r) * 2 + 1, dtype=int)
    idx1 = 0
    idx2 = len(dist_per_row) - 1
    for x_d in range(r, -1, -1):
        dist_per_row[idx1] = -x_d
        dist_per_row[idx2] = x_d
        for y_d in range(r, -1, -1):
            if np.linalg.norm([x_d, y_d]) <= r:
                max_dist_per_col[idx1] = y_d
                max_dist_per_col[idx2] = y_d
                break
        idx1 += 1
        idx2 -= 1

    # create a binary image
    binary_img = np.array(img) >= tol

    # segments
    segmented_img = binary_img.copy() * 0

    segment_ids = {0}

    lowest_active_point_per_col = np.zeros(binary_img.shape[1], dtype=int) * np.nan

    height = binary_img.shape[0]
    width = binary_img.shape[1]

    if is_continuous_image:

        # identify lowest point per col for first r rows:
        for x in range(binary_img.shape[1]):
            for y in range(r - 1, -1, -1): # search from lowest to highest
                if binary_img[y,x]:
                    lowest_active_point_per_col[x] = y

                    next_segment_idx = max(segment_ids) + 1
                    segment_ids.add(next_segment_idx)

                    segmented_img[y,x] = next_segment_idx

                    break # found the lowest -> go to next column

        # go over all active points
        # start with points after height of r and later between 0 and r
        for y in list(range(r, height)) + list(range(0, r)):
            for x in range(width):
                if binary_img[y, x]:

                    # check if there are connected segments
                    col_idxs = np.mod(x + dist_per_row, width)

                    dist_lowest_point_per_col = np.mod(y - lowest_active_point_per_col[col_idxs], height)

                    cols_with_connected_segments = col_idxs[dist_lowest_point_per_col <= max_dist_per_col]

                    if len(cols_with_connected_segments) > 0:

                        segments = segmented_img[lowest_active_point_per_col[cols_with_connected_segments].astype(int), cols_with_connected_segments]

                        # if current point was already segmented,  if it was in the first r cols, then also keep its original segment number in consideration
                        if segmented_img[y, x] > 0:
                            segments = np.hstack((segments, segmented_img[y, x]))

                        cur_segment_id = segments[0]
                        segments_unique = set()
                        for segment_id in segments:
                            if segment_id not in segments_unique:
                                segments_unique.add(segment_id)
                                cur_segment_id = min(cur_segment_id, segment_id)

                        if len(segments_unique) > 1:
                            for segment_id in segments_unique:
                                if segment_id != cur_segment_id:
                                    segmented_img[segmented_img == segment_id] = cur_segment_id
                                    segment_ids.remove(segment_id)

                    else:

                        if segmented_img[y, x] == 0:
                            # give it new segment number if no other segment nearby
                            cur_segment_id = max(segment_ids) + 1
                            segment_ids.add(cur_segment_id)
                        else:
                            # if there it was already segmented because it was in the first r columns, then keep its segment number
                            cur_segment_id = segmented_img[y, x]

                    segmented_img[y, x] = cur_segment_id

                    lowest_active_point_per_col[x] = y

    else:
        # not continuopus image, but one with borders

        # go over all active points
        # start with points after height of r and later between 0 and r
        for y in range(height):
            for x in range(width):
                if binary_img[y, x]:

                    # check if there are connected segments
                    col_idxs = x + dist_per_row
                    col_feature_idxs = (col_idxs >= 0) & (col_idxs < width)
                    col_idxs = col_idxs[col_feature_idxs]

                    dist_lowest_point_per_col = y - lowest_active_point_per_col[col_idxs]

                    cols_with_connected_segments = col_idxs[dist_lowest_point_per_col <= max_dist_per_col[col_feature_idxs]]

                    if len(cols_with_connected_segments) > 0:

                        segments = segmented_img[lowest_active_point_per_col[cols_with_connected_segments].astype(int), cols_with_connected_segments]

                        # if current point was already segmented,  if it was in the first r cols, then also keep its original segment number in consideration
                        if segmented_img[y, x] > 0:
                            segments = np.hstack((segments, segmented_img[y, x]))


                        cur_segment_id = segments[0]
                        segments_unique = set()
                        for segment_id in segments:
                            if segment_id not in segments_unique:
                                segments_unique.add(segment_id)
                                cur_segment_id = min(cur_segment_id, segment_id)

                        if len(segments_unique) > 1:
                            for segment_id in segments_unique:
                                if segment_id != cur_segment_id:
                                    segmented_img[segmented_img == segment_id] = cur_segment_id
                                    segment_ids.remove(segment_id)

                    else:
                        # give it new segment number if no other segment nearby
                        cur_segment_id = max(segment_ids) + 1
                        segment_ids.add(cur_segment_id)

                    segmented_img[y, x] = cur_segment_id

                    lowest_active_point_per_col[x] = y

    sorted_segments = sorted(segment_ids)

    # relabel the segment ids to have no jumps
    for segment_idx in range(1, len(sorted_segments)):
        if segment_idx != sorted_segments[segment_idx]:
            segmented_img[segmented_img == sorted_segments[segment_idx]] = segment_idx

    return segmented_img, list(range(1, len(sorted_segments)))


# class BinaryImageSegmenter:
#
#     @staticmethod
#     def default_config():
#         config = ad.config.Config()
#         config.r = 1
#         config.tol = 0.1
#
#         return config
#
#
#     def __init__(self, config=None, **kwargs):
#         self.config = ad.config.set_default_config(kwargs, config, self.__class__.default_config())
#
#         # holds filter templates for different image sizes if needed
#         self.filter_templates = dict()
#
#
#     def create_filter_template(self, image_shape):
#
#         filter_template = np.zeros(image_shape)
#
#         mid_y = int(image_shape[0] / 2)
#         mid_x = int(image_shape[1] / 2)
#
#         for y in range(mid_y - self.config.r, mid_y + self.config.r + 1):
#             for x in range(mid_x - self.config.r, mid_x + self.config.r + 1):
#                 if np.linalg.norm(np.array([y, x]) - np.array([mid_y, mid_x])) <= self.config.r:
#                     filter_template[y, x] = 1
#
#         mid_point = np.array([mid_y, mid_x])
#
#         return filter_template, mid_point
#
#
#     def get_filter(self, pos, image_shape):
#
#         if image_shape in self.filter_templates:
#             [filter_template, mid_point] = self.filter_templates[image_shape]
#         else:
#             [filter_template, mid_point] = self.create_filter_template(image_shape)
#             self.filter_templates[image_shape] = (filter_template, mid_point)
#
#         shift = np.array(pos) - mid_point
#         return np.roll(filter_template, shift, axis=(0, 1))
#
#
#     def calc(self, img):
#
#         # create a binary image
#         binary_img = np.array(img) >= self.config.tol
#
#         # segments
#         segmented_img = binary_img.copy() * np.nan
#
#         segment_ids = [0]
#
#         # go over all active points
#         for y in range(binary_img.shape[0]):
#             for x in range(binary_img.shape[1]):
#                 if binary_img[y,x]:
#
#                     cur_filter = self.get_filter((y, x), binary_img.shape)
#
#                     # point wise multiplication of filter for current pos and image to identify all the active points that influence the current point
#                     connected_points = np.multiply(binary_img, cur_filter)
#                     connected_segments = np.multiply(segmented_img, cur_filter)
#
#                     # ignore outside segments
#                     connected_segments[connected_segments == 0] = np.nan
#
#                     # get sorted existing segments
#                     min_connected_segment_id = np.nanmin(connected_segments)
#
#                     if np.isnan(min_connected_segment_id):
#                         # if the first is nan, then all others are also nan, because of the sorting of unique
#
#                         # add new segment number because there are no other unseen segments nearby
#                         new_segment_id = segment_ids[-1] + 1
#                         segment_ids.append(new_segment_id)
#
#                         new_points_segment_id = new_segment_id
#
#                     else:
#                         # use for all found segments the minimal existing segment number
#
#                         cur_segment_ids = connected_segments[connected_segments > 0]
#
#                         removed_segment_ids = []
#                         for cur_segment_id in cur_segment_ids:
#                             if cur_segment_id != min_connected_segment_id and cur_segment_id not in removed_segment_ids:
#
#                                 segmented_img[segmented_img == cur_segment_id] = min_connected_segment_id
#
#                                 # remove segment id from list of segments
#                                 segment_ids.remove(cur_segment_id)
#                                 removed_segment_ids.append(cur_segment_id)
#
#                         new_points_segment_id = min_connected_segment_id
#
#                     segmented_img[connected_points.astype(bool)] = new_points_segment_id
#
#
#         # relabel the segment ids to have no jumps
#         segmented_img[np.isnan(segmented_img)] = 0
#
#         for segment_idx in range(1, len(segment_ids)):
#             if segment_idx != segment_ids[segment_idx]:
#                 segmented_img[segmented_img == segment_ids[segment_idx]] = segment_idx
#
#         return segmented_img


def calc_is_segments_finite(image=None, continuous_segmented_image=None, non_continuous_segmented_image=None, tol=0.2, r=1):
    '''This version has a bug. It detects infinite segments as finite if they are connected via different borders: N->E->W->S->N or N->W->E->S->N.'''

    warnings.warn(DeprecationWarning('The function calc_is_segments_finite contains a bug which classifies certain infinite segements as finite. Please use calc_is_segments_finite_v2 instead.'))

    r = int(r)

    if continuous_segmented_image is None:
        continuous_segmented_image, continuous_segments = ad.helper.statistics.calc_active_binary_segments(image, tol=tol, r=r, is_continuous_image=True)
    else:
        (continuous_segmented_image, continuous_segments) = continuous_segmented_image

    if non_continuous_segmented_image is None:
        non_continuous_segmented_image, non_continuous_segments = ad.helper.statistics.calc_active_binary_segments(image, tol=tol, r=r, is_continuous_image=False)
    else:
        (non_continuous_segmented_image, non_continuous_segments) = non_continuous_segmented_image

        # prepare the distances that have to be taken into account to say an active point is part of another active point
    max_dist_per_col = np.zeros(int(r)*2 + 1, dtype=int)
    dist_per_row = np.zeros(int(r) * 2 + 1, dtype=int)
    idx1 = 0
    idx2 = len(dist_per_row) - 1
    for x_d in range(r, -1, -1):
        dist_per_row[idx1] = -x_d
        dist_per_row[idx2] = x_d
        for y_d in range(r, -1, -1):
            if np.linalg.norm([x_d, y_d]) <= r:
                max_dist_per_col[idx1] = y_d
                max_dist_per_col[idx2] = y_d
                break
        idx1 += 1
        idx2 -= 1

    height = continuous_segmented_image.shape[0]
    width = continuous_segmented_image.shape[1]

    # get relevant segments, i.e. upper and lower segments of each side
    # identify lowest point per col for first r rows:

    default_location_array = np.zeros(width) * np.nan


    ###########################################
    # check upper and lower border

    upper_locations_per_segment = dict()
    lower_locations_per_segment = dict()
    for x in range(width):

        # upper area of image
        for y in range(r):  # search from upper to lower
            if continuous_segmented_image[y, x] > 0:

                segment_id = continuous_segmented_image[y, x]

                if segment_id not in upper_locations_per_segment:
                    upper_locations_per_segment[segment_id] = default_location_array.copy()

                upper_locations_per_segment[segment_id][x] = y

                break  # found the upper -> go to next column

        # lower area of image
        for y in range(height-1, height-r-1, -1):  # search from lower to upper
            if continuous_segmented_image[y, x] > 0:

                segment_id = continuous_segmented_image[y, x]

                if segment_id not in lower_locations_per_segment:
                    lower_locations_per_segment[segment_id] = default_location_array.copy()

                lower_locations_per_segment[segment_id][x] = y

                break  # found the lower -> go to next column

    # go over the segments in the border regions and check if they are connected in the continuous and non-continuous segmentation
    # if they are connected in both then they must be infinite

    # only check segments that are in both
    check_segment_ids = set(upper_locations_per_segment.keys()).intersection(lower_locations_per_segment.keys())

    infinite_segments_ids = []

    # check if one of them is infinite
    for cur_segment_id in check_segment_ids:

        stop = False

        for x, y in enumerate(upper_locations_per_segment[cur_segment_id]):
            if not np.isnan(y):
                y = int(y)

                # get all points that are connected in the lower area and chek if they are also connected only with a direct link via inside the image not via image borders

                # check if there are connected segments
                col_idxs = np.mod(x + dist_per_row, width)

                dist_lowest_point_per_col = y - (lower_locations_per_segment[cur_segment_id] - height)

                cols_with_connected_segments = col_idxs[dist_lowest_point_per_col[col_idxs] <= max_dist_per_col]
                rows_with_connected_segments = lower_locations_per_segment[cur_segment_id][cols_with_connected_segments].astype(int)

                # check each connected point to see if it is also connected in non continuous segmentation
                for idx in range(len(cols_with_connected_segments)):
                    if non_continuous_segmented_image[rows_with_connected_segments[idx], cols_with_connected_segments[idx]] == non_continuous_segmented_image[y, x]:
                        infinite_segments_ids.append(cur_segment_id)
                        stop = True
                        break

                if stop:
                    break

    ###########################################
    # check left and right border

    leftest_locations_per_segment = dict()
    rightest_locations_per_segment = dict()
    for y in range(height):

        # upper area of image
        for x in range(r):  # search from left to right
            if continuous_segmented_image[y, x] > 0:

                segment_id = continuous_segmented_image[y, x]

                if segment_id not in leftest_locations_per_segment:
                    leftest_locations_per_segment[segment_id] = default_location_array.copy()

                leftest_locations_per_segment[segment_id][y] = x

                break  # found the upper -> go to next column

        # lower area of image
        for x in range(width - 1, width - r - 1, -1):  # search from lower to upper
            if continuous_segmented_image[y, x] > 0:

                segment_id = continuous_segmented_image[y, x]

                if segment_id not in rightest_locations_per_segment:
                    rightest_locations_per_segment[segment_id] = default_location_array.copy()

                rightest_locations_per_segment[segment_id][y] = x

                break  # found the lower -> go to next column

    # go over the segments in the border regions and check if they are connected in the continuous and non-continuous segmentation
    # if they are connected in both then they must be infinite

    # only check segments that are in both
    check_segment_ids = set(leftest_locations_per_segment.keys()).intersection(rightest_locations_per_segment.keys())

    # do not check segments that are already found to be ininite
    check_segment_ids = check_segment_ids.difference(infinite_segments_ids)

    # check if one of them is infinite
    for cur_segment_id in check_segment_ids:

        stop = False

        for y, x in enumerate(leftest_locations_per_segment[cur_segment_id]):
            if not np.isnan(x):
                x = int(x)

                # get all points that are connected in the lower area and chek if they are also connected only with a direct link via inside the image not via image borders

                # check if there are connected segments
                row_idxs = np.mod(y + dist_per_row, height)

                dist_lowest_point_per_col = x - (rightest_locations_per_segment[cur_segment_id] - width)

                rows_with_connected_segments = row_idxs[dist_lowest_point_per_col[row_idxs] <= max_dist_per_col]
                cols_with_connected_segments = rightest_locations_per_segment[cur_segment_id][rows_with_connected_segments].astype(int)

                # check each connected point to see if it is also connected in non continuous segmentation
                for idx in range(len(cols_with_connected_segments)):
                    if non_continuous_segmented_image[rows_with_connected_segments[idx], cols_with_connected_segments[idx]] == non_continuous_segmented_image[y, x]:
                        infinite_segments_ids.append(cur_segment_id)
                        stop = True
                        break

                if stop:
                    break

    finite_segments = list(set(continuous_segments).difference(infinite_segments_ids))

    return finite_segments, (continuous_segmented_image, continuous_segments), (non_continuous_segmented_image, non_continuous_segments)


def calc_is_segments_finite_v2(image=None, continuous_segmented_image=None, tol=0.2, r=1):
    '''Identifies if finite elements are in the image'''

    r = int(r)

    if continuous_segmented_image is None:
        continuous_segmented_image, continuous_segments = ad.helper.statistics.calc_active_binary_segments(image, tol=tol, r=r, is_continuous_image=True)
    else:
        (continuous_segmented_image, continuous_segments) = continuous_segmented_image

    non_finite_segments = continuous_segments.copy()
    finite_segments = []

    # create filter used to multiply with the image to detect if certain segments are finite
    r_half = int(np.ceil(r / 2))

    filter_mat = np.full(continuous_segmented_image.shape, 0)
    filter_mat[0:r_half, :] = 1
    filter_mat[-r_half:, :] = 1
    filter_mat[:, 0:r_half] = 1
    filter_mat[:, -r_half:] = 1

    # centered on each pixel value, multiply the filter with the suroundings
    # start from the normal picture

    y_len = continuous_segmented_image.shape[0]
    x_len = continuous_segmented_image.shape[1]

    stop = False

    for y_shift in range(y_len):
        for x_shift in range(x_len):


            # Buffer positions in original image
            # ---------
            # | 1 | 2 |
            # |-------|
            # | 3 | 4 |
            # ---------

            buf1 = continuous_segmented_image[0:y_shift, 0:x_shift]           * filter_mat[(y_len-y_shift):y_len, (x_len-x_shift):x_len]
            buf2 = continuous_segmented_image[0:y_shift, x_shift:x_len]       * filter_mat[(y_len-y_shift):y_len, 0:(x_len-x_shift)]
            buf3 = continuous_segmented_image[y_shift:y_len, 0:x_shift]       * filter_mat[0:(y_len-y_shift), (x_len-x_shift):x_len]
            buf4 = continuous_segmented_image[y_shift:y_len, x_shift:x_len]   * filter_mat[0:(y_len-y_shift), 0:(x_len-x_shift)]

            # check if some of the segments are not in the
            remove_segments = []
            for segment_id in non_finite_segments:
                if not np.any(buf1 == segment_id) and not np.any(buf2 == segment_id) and not np.any(buf3 == segment_id) and not np.any(buf4 == segment_id):
                    remove_segments.append(segment_id)

                    finite_segments.append(segment_id)

                    # stop if all segments are finite
                    if not non_finite_segments:
                        stop = True

            for segment_id in remove_segments:
                non_finite_segments.remove(segment_id)

            if stop:
                break


    return sorted(finite_segments), (continuous_segmented_image, continuous_segments)




# def calc_is_segments_finite_v2(image=None, continuous_segmented_image=None, tol=0.2, r=1):
#     '''Identifies if finite elements are in the image'''
#
#     r = int(r)
#
#     if continuous_segmented_image is None:
#         continuous_segmented_image, continuous_segments = ad.helper.statistics.calc_active_binary_segments(image, tol=tol, r=r, is_continuous_image=True)
#     else:
#         (continuous_segmented_image, continuous_segments) = continuous_segmented_image
#
#     non_finite_segments = continuous_segments.copy()
#     finite_segments = []
#
#     # create filter used to multiply with the image to detect if certain segments are finite
#     r_half = int(np.ceil(r / 2))
#
#     filter_mat = np.full(continuous_segmented_image.shape, 0)
#     filter_mat[0:r_half, :] = 1
#     filter_mat[-r_half:, :] = 1
#     filter_mat[:, 0:r_half] = 1
#     filter_mat[:, -r_half:] = 1
#
#     # centered on each pixel value, multiply the filter with the suroundings
#     # start from the normal picture
#
#     #y_shifts, x_shifts = [0, 1, 2, ... ,-1, -2, ...]
#     y_shifts = [0] + list(range(1, int(np.ceil(continuous_segmented_image.shape[0] / 2)) + 1)) + list(range(-1, int(-np.ceil(continuous_segmented_image.shape[0] / 2)), -1))
#     x_shifts = [0] + list(range(1, int(np.ceil(continuous_segmented_image.shape[1] / 2)) + 1)) + list(range(-1, int(-np.ceil(continuous_segmented_image.shape[1] / 2)), -1))
#
#     stop = False
#
#     for y_shift in y_shifts:
#         for x_shift in x_shifts:
#
#             shifted_image = np.roll(continuous_segmented_image, (y_shift, x_shift), (0, 1))
#
#             buf = shifted_image * filter_mat
#
#             # check if some of the segments are not in the
#             remove_segments = []
#             for segment_id in non_finite_segments:
#                 if not np.any(buf == segment_id):
#                     remove_segments.append(segment_id)
#
#                     finite_segments.append(segment_id)
#
#                     # stop if all segments are finite
#                     if not non_finite_segments:
#                         stop = True
#
#             for segment_id in remove_segments:
#                 non_finite_segments.remove(segment_id)
#
#             if stop:
#                 pass
#                 #break
#
#     return sorted(finite_segments), (continuous_segmented_image, continuous_segments)




class NBinDiversityNBinPerDim():
    '''
    Calculates diversity of a set of points by counting in how many sections of the space in which the points exisits falls at elast one point.
    The space gets divided by having an equal distribution of bins per space dimension.
    '''

    @staticmethod
    def calc_for_points(points, n_bins_per_dimension=5, fixed_borders=None, ignore_out_of_range_values=False):
        div = NBinDiversityNBinPerDim(points=points, n_bins_per_dimension=n_bins_per_dimension, fixed_borders=fixed_borders, ignore_out_of_range_values=ignore_out_of_range_values)
        return div.diversity, div.diversity_per_added_point


    def __init__(self, points=None, n_bins_per_dimension=5, fixed_borders=None, ignore_out_of_range_values=False):

        self.n_bins_per_dimension = n_bins_per_dimension
        self.ignore_out_of_range_values = ignore_out_of_range_values

        if points is None:
            self.points = []
        elif isinstance(points, np.ndarray):
            # save points as a list and not as numpy array
            if len(points.shape) == 1:
                self.points = [points]
            else:
                self.points = list(points)
        else:
            # assume it is a list
            self.points = points

        if fixed_borders is None:
            self.is_fixed_borders = False

            if not self.points:
                self._min_values = np.array([])
                self._max_values = np.array([])
            else:
                self._min_values = np.nanmin(self.points, 0)  # min per dimension
                self._max_values = np.nanmax(self.points, 0)  # max per dimension
        else:
            self.is_fixed_borders = True

            self._min_values = np.array([borders[0] for borders in fixed_borders])
            self._max_values = np.array([borders[1] for borders in fixed_borders])

        self._bin_definition_per_dim = []
        for dim_idx in range(len(self._min_values)):
            self._bin_definition_per_dim.append(np.linspace(self._min_values[dim_idx], self._max_values[dim_idx], num=self.n_bins_per_dimension + 1))

        self._count_per_section = []

        self.diversity_per_added_point = [] # diversity dependent on the number of points in the list
        self.diversity = []

        self._recalculate_diversity_over_all_points()


    def _update_border_values(self, point):

        is_updated = False

        if not self.is_fixed_borders:

            is_changed_borders = False

            if self._min_values.size == 0  and self._min_values.size == 0:
                # values where never set --> first point
                self._min_values = np.array([val for val in point])
                self._max_values = np.array([val for val in point])
                is_changed_borders = True

            else:

                point = np.array(point)

                # check if new borders must be set
                smaller_inds = point < self._min_values
                if np.any(smaller_inds):
                    self._min_values[smaller_inds] = point[smaller_inds]
                    is_changed_borders = True

                larger_inds = point > self._max_values
                if np.any(larger_inds):
                    self._max_values[larger_inds] = point[larger_inds]
                    is_changed_borders = True

            if is_changed_borders:
                self._bin_definition_per_dim = []
                # create new section borders
                for dim_idx in range(len(self._min_values)):
                    self._bin_definition_per_dim.append(np.linspace(self._min_values[dim_idx], self._max_values[dim_idx], num=self.n_bins_per_dimension + 1))
                is_updated = True

        return is_updated


    def add_point(self, point):

        if not self.is_fixed_borders:

            # it is possible that the min and max value change
            # therefore new points need to be saved to recalculate the diversity in case the min and max values change
            self.points.append(point)

            is_updated = self._update_border_values(point)

            # if the ranges have changed, then the diversity for each step needs to be recomputed.
            if is_updated:
                self._recalculate_diversity_over_all_points()
            else:
                self._set_diversity_for_new_point(point)

        else:
            self._set_diversity_for_new_point(point)

        return self.diversity


    def _recalculate_diversity_over_all_points(self):

        self._count_per_section = collections.defaultdict(int)
        self.diversity_per_added_point = []
        self.diversity = []

        if self.points:
            for point in self.points:
                self._set_diversity_for_new_point(point)

        return self.diversity


    def _set_diversity_for_new_point(self, point):

        section = []

        # check each dimension
        for dim_idx in range(len(self._bin_definition_per_dim)):

            if np.isnan(point[dim_idx]):
                section_idx = -2 # extra section for nan numbers
            else:

                # identify at which section in de fined grid the value falls
                idxs = np.where(self._bin_definition_per_dim[dim_idx] > point[dim_idx])[0]

                if len(idxs) == 0:

                    if self._bin_definition_per_dim[dim_idx][-1] == point[dim_idx]:
                        section_idx = len(self._bin_definition_per_dim[dim_idx])-2
                    else:

                        # value is larger than upper grid border
                        if self.ignore_out_of_range_values:
                            section = None
                            break
                        else:
                            section_idx = len(self._bin_definition_per_dim[dim_idx])

                elif idxs[0] == 0:
                    # value is smaller than lower grid border

                    if self.ignore_out_of_range_values:
                        section = None
                        break
                    else:
                        section_idx = -1
                else:
                    section_idx = idxs[0] - 1

            section.append(section_idx)

        if section is not None:
            section = tuple(section)
            self._count_per_section[section] += 1

        new_diversity = len(self._count_per_section)
        self.diversity_per_added_point.append(new_diversity)
        self.diversity = new_diversity

        return new_diversity




class NBinDiversityNBins():
    '''
    Calculates diversity of a set of points by counting in how many sections of the space in which the points exisits falls at elast one point.
    The space gets divided by having a certain number of bins.
    '''

    @staticmethod
    def calc_for_points(points, n_bins=5, fixed_borders=None):
        div = NBinDiversityNBins(points=points, n_bins=n_bins, fixed_borders=fixed_borders)
        return div.diversity, div.diversity_per_added_point


    def __init__(self, points=None, n_bins=100, fixed_borders=None):

        self.n_bins = n_bins

        if points is None:
            self.points = []
        elif isinstance(points, np.ndarray):
            # save points as a list and not as numpy array
            if len(points.shape) == 1:
                self.points = [points]
            else:
                self.points = list(points)
        else:
            # assume it is a list
            self.points = points

        if fixed_borders is None:
            self.is_fixed_borders = False

            if not self.points:
                self._min_values = np.array([])
                self._max_values = np.array([])
            else:
                self._min_values = np.nanmin(self.points, 0)  # min per dimension
                self._max_values = np.nanmax(self.points, 0)  # max per dimension
        else:
            self.is_fixed_borders = True

            self._min_values = np.array([borders[0] for borders in fixed_borders])
            self._max_values = np.array([borders[1] for borders in fixed_borders])

        n_dims = None
        if self.is_fixed_borders:
            n_dims = len(self._min_values)
        elif len(self.points) > 0:
            n_dims = len(points[0])

        self._bin_start_definitions_template = np.array([])
        self._bin_end_definitions_template = np.array([])

        self._bin_start_definitions = np.array([])
        self._bin_end_definitions = np.array([])


        if n_dims is not None:
            self._bin_start_definitions_template, self._bin_end_definitions_template = self.get_section_definitions(n_bins, n_dims)
            self._update_bin_definitions()

        self._count_per_section = []

        self.diversity_per_added_point = [] # diversity dependent on the number of points in the list
        self.diversity = []

        self._recalculate_diversity_over_all_points()


    def _update_bin_definitions(self):

        if self._min_values.size > 0 and self._bin_start_definitions_template.size > 0:

            self._bin_start_definitions = np.zeros(self._bin_start_definitions_template.shape)
            self._bin_end_definitions = np.zeros(self._bin_end_definitions_template.shape)

            range_per_dim = self._max_values - self._min_values

            for dim_idx in range(self._min_values.shape[0]):

                self._bin_start_definitions[:, dim_idx] = self._min_values[dim_idx] + self._bin_start_definitions_template[:, dim_idx] * range_per_dim[dim_idx]
                self._bin_end_definitions[:, dim_idx] = self._min_values[dim_idx] + self._bin_end_definitions_template[:, dim_idx] * range_per_dim[dim_idx]


    @staticmethod
    def get_section_definitions(n_sections, n_dims):

        # initial bins span whole space

        # each bin is defined by its start and end point in the space for each dimension

        bin_start_definitions = [[0] * n_dims]
        bin_end_definitions = [[1] * n_dims]

        # detect the number of borders per dimension

        # the number of sections is given by
        # 2**(n_dim*n_cuts) depending on the number of dimensions and how many cuts are made
        # this can be generated for an eauall amount of cuts per dimension for x cuts as long as
        # 2**(n_dim*n_cuts) < n_sections
        # there rest of the sections can be introduced by the following algorithm

        # go over dimensions and introduce new cuts as long as more sections are needed

        cur_n_sections = 1

        dim_idx = 0
        bin_idx = 0

        while cur_n_sections < n_sections:

            # introduce a new section
            section_length = bin_end_definitions[bin_idx][dim_idx] - bin_start_definitions[bin_idx][dim_idx]

            new_border_point = bin_start_definitions[bin_idx][dim_idx] + 0.5 * section_length

            new_bin_start_definition = copy.copy(bin_start_definitions[bin_idx])
            new_bin_end_definition = copy.copy(bin_end_definitions[bin_idx])

            new_bin_start_definition[dim_idx] = new_border_point
            bin_end_definitions[bin_idx][dim_idx] = new_border_point

            bin_start_definitions.insert(bin_idx + 1, new_bin_start_definition)
            bin_end_definitions.insert(bin_idx + 1, new_bin_end_definition)

            cur_n_sections += 1

            bin_idx = bin_idx + 2

            if bin_idx >= len(bin_start_definitions):
                bin_idx = 0
                dim_idx += 1

            if dim_idx >= n_dims:
                dim_idx = 0

        return np.array(bin_start_definitions), np.array(bin_end_definitions)


    def _update_border_values(self, point):

        is_updated = False

        if not self.is_fixed_borders:

            is_changed_borders = False

            if self._min_values.size == 0  and self._min_values.size == 0:
                # values where never set --> first point
                self._min_values = np.array([val for val in point])
                self._max_values = np.array([val for val in point])
                is_changed_borders = True

            else:

                point = np.array(point)

                # check if new borders must be set
                smaller_inds = (point < self._min_values) | np.isnan(self._min_values)
                if np.any(smaller_inds):
                    self._min_values[smaller_inds] = point[smaller_inds]
                    is_changed_borders = True

                larger_inds = (point > self._max_values) | np.isnan(self._max_values)
                if np.any(larger_inds):
                    self._max_values[larger_inds] = point[larger_inds]
                    is_changed_borders = True

            if is_changed_borders:
                self._update_bin_definitions()
                is_updated = True

        return is_updated


    def add_point(self, point):

        # if there are no template yet, because the number of dimenisons was not known, then make them now
        if len(self._bin_start_definitions_template) == 0:
            self._bin_start_definitions_template, self._bin_end_definitions_template = self.get_section_definitions(self.n_bins, len(point))

        if not self.is_fixed_borders:

            # it is possible that the min and max value change
            # therefore new points need to be saved to recalculate the diversity in case the min and max values change
            self.points.append(point)

            is_updated = self._update_border_values(point)

            # if the ranges have changed, then the diversity for each step needs to be recomputed.
            if is_updated:
                self._recalculate_diversity_over_all_points()
            else:
                self._set_diversity_for_new_point(point)

        else:
            self._set_diversity_for_new_point(point)

        return self.diversity


    def _recalculate_diversity_over_all_points(self):

        self._count_per_section = collections.defaultdict(int)
        self.diversity_per_added_point = []
        self.diversity = []

        if self.points:
            for point in self.points:
                self._set_diversity_for_new_point(point)

        return self.diversity


    def _set_diversity_for_new_point(self, point):

        if np.any(np.isnan(point)):
            self._count_per_section[tuple(np.isnan(point))] += 1
        else:
            section_idx = None

            larger_than_start_inds = np.sum(point >= self._bin_start_definitions, axis=1) == len(point)
            smaller_than_end_inds = np.sum(point <= self._bin_end_definitions, axis=1) == len(point)

            valid_sections_inds = larger_than_start_inds & smaller_than_end_inds
            valid_sections_idxs = np.where(valid_sections_inds)[0]

            if valid_sections_idxs.size > 0:
                section_idx = valid_sections_idxs[0]

            if section_idx is not None:
                self._count_per_section[section_idx] += 1
            else:
                warnings.warn("Point outside of defined section borders.")

        new_diversity = len(self._count_per_section)
        self.diversity_per_added_point.append(new_diversity)
        self.diversity = new_diversity

        return new_diversity