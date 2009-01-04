from numpy import *
from ore.ore import *
import gc

bld = DynamicSVO()


LevelNum = 8
BaseRadius = 256


lev_gens = [] 
for lev in range(LevelNum):
    lev_gens.append( MakeSphereSource(BaseRadius / (2**lev), (128, 128, lev*255/LevelNum), False) )

def Rec(lev, pos, x, y, z):
    if lev > 4:
        bld.BuildRange(11, p3i(pos), BuildMode.GROW, lev_gens[lev])
        print bld.nodecount

    if lev < LevelNum-1:
        (x1, y1, z1) = [ v/2 for v in (x, y, z)]
        Rec(lev+1, pos+x, y1, z1, x1)
        Rec(lev+1, pos-x, y1, z1,-x1)

        Rec(lev+1, pos+y, x1, z1, y1)
        Rec(lev+1, pos-y, x1, z1,-y1)

        Rec(lev+1, pos+z, x1, y1, z1)

    
Rec(0, array([1024]*3), array([BaseRadius*1.5, 0, 0]), array([0, BaseRadius*1.5, 0]), array([0, 0, BaseRadius*1.5]))

print "saveing..."
bld.Save("data/spheres.vox")    

