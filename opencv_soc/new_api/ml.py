import numpy as np
import cv2

def load_base(fn):
    return np.loadtxt(fn, np.float32, delimiter=',', converters={ 0 : lambda ch : ord(ch)-ord('A') })

CV_ROW_SAMPLE = 1
CV_VAR_NUMERICAL   = 0
CV_VAR_ORDERED     = 0
CV_VAR_CATEGORICAL = 1

a = load_base('letter-recognition.data')
samples = a[:,1:]
responses = a[:,0]
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


