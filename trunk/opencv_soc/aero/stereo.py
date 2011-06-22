import numpy as np
import cv2


def calc_disparity(img1, img2, min_disp, max_disp):
    window_size = 3

    min_disp = int(np.floor(min_disp))
    num_disp = int(np.ceil(max_disp - min_disp))
    num_disp = ((num_disp+15) % 16) * 16

    stereo = cv2.StereoSGBM(minDisparity = min_disp, numDisparities = num_disp, SADWindowSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = True
    )

    disp = np.float32(stereo.compute(img1, img2) / 16.0)
    return disp


if __name__ == '__main__':
    from time import clock
    import pylab as pl

    img1 = cv2.imread('00_l.bmp', 1)
    img2 = cv2.imread('00_r.bmp', 1)

    t = clock()
    disp = calc_disparity(img1, img2, -40, 100)
    print clock() - t

    pl.imshow(disp)
    pl.colorbar()
    pl.show()
