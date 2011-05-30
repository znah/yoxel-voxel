import argparse

import numpy as np
import cv2

def load_base(fn):
    a = np.loadtxt(fn, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })
    samples, responses = a[:,1:], a[:,0]
    return samples, responses


CV_ROW_SAMPLE = 1
CV_VAR_NUMERICAL   = 0
CV_VAR_ORDERED     = 0
CV_VAR_CATEGORICAL = 1


def test_rtrees(base):
    samples, responses = base
    sample_n, var_n = samples.shape

    print 'training...'
    forest = cv2.RTrees()
    var_types = np.array([CV_VAR_NUMERICAL] * var_n + [CV_VAR_CATEGORICAL], np.int8)

    #CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER));
    params = dict(max_depth=10 )

    forest.train(samples[:10000], CV_ROW_SAMPLE, responses[:10000], varType = var_types, params = params)

    print 'testing...'
    predicted = [forest.predict(s) for s in samples]

    print 'recognition rate: ', np.mean(predicted==responses)*100

def test_knearest(base):
    samples, responses = base
    sample_n, var_n = samples.shape

    print 'training...'
    kn = cv2.KNearest(samples)
   

if __name__ == '__main__':
    base = load_base('letter-recognition.data')
    test_rtrees(base)
