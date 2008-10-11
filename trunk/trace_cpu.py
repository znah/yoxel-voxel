from numpy import *
from scipy import weave
from scipy.weave import converters
from ore.ore import *


class TreeRenderer:
    def __init__(self, scene, res=512):
        self.res = res
        self.img = zeros((res, res, 3), uint8)
        self.scene = scene

    def render(self, viewDir, first=False):
        res = self.res
        img = self.img
        (root, flags, parents, children) = (self.scene.root, self.scene.flags, self.scene.parents, self.scene.children)
        code = '''
            #line 2000 "tree_gen.py"
            VoxTree tree;
            tree.root = root;
            tree.flags = (VoxNodeInfo*)flags;
            tree.parents = (VoxNodeId*)parents;
            tree.children = (VoxChild*)children;

            TreeTracer tr(tree);

            for (OrthoRayGen rg(viewDir, res); !rg.done(); rg.next())
            {
              //uchar4 col = tr.traceRay(rg.p0(), rg.dir());
              uchar4 col = tr.stacklessTrace(rg.p0(), rg.dir());
              int ofs = rg.flat()*3;
              img[ofs] = col.x;
              img[ofs+1] = col.y;
              img[ofs+2] = col.z;
            }
        '''
        weave.inline(code, ["root", "flags", "parents", "children", "res", "viewDir", "img"], 
           headers=['"vox_trace.h"', '"ray_gen.h"'], include_dirs=['cpp', 'c:/cuda/include'], force=first)

    def getImage(self):
        return self.img

if __name__ == '__main__':
    scene = VoxTree()
    scene.load("data/hmap.vox")

    renderer = TreeRenderer(scene, 640)
    from render import renderTurn
    renderTurn(renderer)
