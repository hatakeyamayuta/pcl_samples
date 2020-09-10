from PIL import Image
import numpy as np
import open3d as o3d
import sys
# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def birds_eye_point_cloud(points, colors,
                          side_range=(-5, 5),
                          fwd_range=(0,1.4),
                          res=0.01,
                          min_height = -1.0,
                          max_height = 5.0,
                          saveto="sample.png"):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 2]
    y_lidar = -points[:, 0]
    z_lidar = -points[:, 1]
    
    
    r_lidar = colors[:, 0]
    g_lidar = colors[:, 1]
    b_lidar = colors[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()
    
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later
    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))
    min_height = np.amin(z_lidar[indices])
    max_height = np.amax(z_lidar[indices])
    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8)
    im2 = np.zeros([y_max, x_max,3], dtype=np.uint8)
    im[-y_img, x_img,0] = r_lidar[indices]*255 # -y because images start from top left
    im[-y_img, x_img,1] = g_lidar[indices]*255 # -y because images start from top left
    im[-y_img, x_img,2] = b_lidar[indices]*255 # -y because images start from top left


    im2[-y_img, x_img,0] = 255*(np.sin(1.5*np.pi*pixel_values/255+np.pi+np.pi/4)+1)/2 # -y because images start from top left
    im2[-y_img, x_img,1] = 255*(np.sin(1.5*np.pi*pixel_values/255+np.pi+np.pi/4+np.pi/2) +1)/2 # -y because images start from top left
    im2[-y_img, x_img,2] = 255*(np.sin(1.5*np.pi*pixel_values/255+np.pi+np.pi/4+np.pi)+1)/2 # -y because images start from top left
    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)
    im2 = Image.fromarray(im2)

    # SAVE THE IMAGE
    if saveto is not None:
        im.save(saveto)
        im2.save("heart_map.png")
    else:
        im.show()

if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(sys.argv[1])
    pt = np.array(pcd.points)
    print(pt[0])
    colors = np.array(pcd.colors)
    birds_eye_point_cloud(pt,colors)
