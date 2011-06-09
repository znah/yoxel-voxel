from glob import glob
import re

for fn in glob('*.py'):
    print ' --- ', fn
    code = open(fn).read()
    names = sorted(set(re.findall('cv2?\.\w+', code)))
    for s in names:
        print '  ', s

