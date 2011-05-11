import numpy as np
from numpy import random
import cv, cv2


cluster_n = 5

points = []
ref_means = []
ref_covs = []
for i in xrange(cluster_n):
    mean = random.rand(2) * 512
    a = (random.rand(2, 2)-0.5)*100
    cov = np.dot(a.T, a)
    n = random.randint(1000)
    pts = random.multivariate_normal(mean, cov, n)
    points.append( pts )
    ref_means.append( mean )
    ref_covs.append( cov )
points = np.float32( np.vstack(points) )

em = cv2.EM(points, params = dict( nclusters = cluster_n, cov_mat_type = cv2.EM_COV_MAT_GENERIC) )
print em.getMeans(**{})

covs = np.zeros((cluster_n, 2, 2), np.float32)
covs = em.getCovs(covs)


img = np.zeros((512, 512, 3), np.uint8)
cv.Circle(img, (100, 100), 50, (0, 255, 0))
cv2.imshow('em_test', img)
cv2.waitKey(0)



#x, y = points.T
#pl.plot(x, y, '.')
#pl.show()



