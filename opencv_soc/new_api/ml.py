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
    kn = cv2.KNearest(samples[:10000], responses[:10000])

    print 'testing...'
    retval, results, neigh_resp, dists = kn.find_nearest(samples, k = 10)
    results = np.ravel(results)
    print 'recognition rate: ', np.mean(results == responses)*100

def test_boosting(base):
    samples, responses = base
    sample_n, var_n = samples.shape
    class_n = 26

    # unroll
    new_samples = np.zeros((sample_n*class_n, var_n+1), np.float32)
    new_samples[:,:-1] = np.repeat(samples, class_n, axis=0)
    new_samples[:,-1] = np.tile(np.arange(class_n), sample_n)
    
    new_responses = np.zeros(sample_n*class_n, np.int32)
    resp_idx = np.int32( responses + np.arange(sample_n)*class_n )
    new_responses[resp_idx] = 1

    print 'training...'
    var_types = np.array([CV_VAR_NUMERICAL] * var_n + [CV_VAR_CATEGORICAL, CV_VAR_CATEGORICAL], np.int8)
    #CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 )
    params = dict(max_depth=5)
    boost = cv2.Boost(new_samples[:10000*class_n], CV_ROW_SAMPLE, new_responses[:10000*class_n], varType = var_types, params=params)
    
    
    print 'testing...'

    resp =  np.array( [boost.predict(s) for s in new_samples] )
    print (resp == new_responses).reshape(-1, class_n).all(axis=1).mean()



    # boost.train( new_data, CV_ROW_SAMPLE, new_responses, 0, 0, var_type, 0,
    # CvBoostParams(CvBoost::REAL, 100, 0.95, 5, false, 0 ));


   

if __name__ == '__main__':
    base = load_base('letter-recognition.data')
    test_boosting(base)
