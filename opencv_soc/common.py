from numpy import *
import cv

def to_list(a):
    return [tuple(p) for p in a]

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return sqrt( anorm2(a) )


def gray2bgr(gray):
    w, h = cv.GetSize(gray)
    bgr = cv.CreateMat(h, w, cv.CV_8UC3)
    cv.CvtColor(gray, bgr, cv.CV_GRAY2BGR)
    return bgr
