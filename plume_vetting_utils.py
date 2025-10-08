import numpy as np
from plume_vetting.sfun import find_uniform_indices
import pylab as plt

def compute_masks(mf, mf_threshold=30, clouds_and_surface_water_mask=None, debug=False):
    ######### calculate bad pixel mask ################
    uni_row, uni_col = find_uniform_indices(mf)  # find rows and columns with two or fewer unique MF values
    rowcol_mask = np.full_like(clouds_and_surface_water_mask, False, dtype=bool)
    rowcol_mask[uni_row, :] = True
    rowcol_mask[:, uni_col] = True
    rowcol_mask[mf <= -9999] = True

    # This mask is used for target pixels. Combine the rowcol_mask with the clouds_and_surface_water_mask
    combined_mask = clouds_and_surface_water_mask | rowcol_mask

    # This mask is used for background pixels
    background_mask = combined_mask.copy()
    background_mask[(mf < -mf_threshold) | (mf > mf_threshold)] = True

    if debug:
        plt.imshow(combined_mask)
        plt.title("combined_mask")
        plt.show()
        plt.imshow(background_mask)
        plt.title("background_mask")
        plt.show()

    return combined_mask, background_mask

def coords_inside_plume_from_binary(thresholded_instance, debug=False):
    points_inside_plume = []
    minx, miny, maxx, maxy = 0, 0, thresholded_instance.shape[0], thresholded_instance.shape[1]
    for y in range(int(miny), int(maxy)):
        for x in range(int(minx), int(maxx)):
            if thresholded_instance[x,y]:
                points_inside_plume.append((y, x))

    if debug:
        debug_viz_data = np.zeros_like(thresholded_instance)
        for coords in points_inside_plume:
            y, x = coords
            debug_viz_data[x, y] = 1.
        plt.imshow(debug_viz_data)
        plt.show()

    return points_inside_plume