from skimage.measure import compare_ssim as _compare_ssim
import numpy as np
import cv2
import math

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(grayA,grayB):
    (score, diff) = _compare_ssim(grayA, grayB, full=True)
    return score

def compare_psnr(original, superres, interpolated):
    interpolated_psnr = psnr(original, interpolated)
    superres_psnr = psnr(original, superres)
    return interpolated_psnr, superres_psnr

def compare_ssim(original, superres, interpolated):
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(superres, cv2.COLOR_BGR2GRAY)
    grayC = cv2.cvtColor(interpolated, cv2.COLOR_BGR2GRAY)
    test = np.zeros(grayC.shape)
    interpolated_ssim = ssim(grayA,grayC)
    superres_ssim = ssim(grayA,grayB)
    return interpolated_ssim, superres_ssim
