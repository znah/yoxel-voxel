#include <stdio.h>
#include <stdlib.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include "../data.h"
#include "vector.h"

int childMask = 0;
volatile spu_context ctx __attribute__ ((aligned (16)));
int result[800] __attribute__ ((aligned (16)));
int tag_id;

int trace(Node * nodeAdress, Vector t1, Vector t2) {
    Node node __attribute__ ((aligned (16)));
    //printf("Node addres %d\n", ((int)nodeAdress) % 64);
    spu_mfcdma32((void *)(&node), (unsigned int)nodeAdress, sizeof(Node), tag_id, MFC_GET_CMD);
    (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);

    Vector tm = {.v[0] = 0, .v[1] = 0, .v[2] = 0};
    add(&tm, &t1);
    add(&tm, &t2);
    div2(&tm, 2);
    int ch = 0;
    double tE = max(&t1);
    int coord = 0;
    for (coord = 0; coord < 3; coord++) {
        if (tE > tm.v[coord]) {
            ch |= 1 << coord;
        }
    }

    for (coord = 0; coord < 3; coord++) {
        if ((ch & (1 << coord)) != 0) {
            t1.v[coord] = tm.v[coord];
        } else {
            t2.v[coord] = tm.v[coord];
        }
    }
    while (1) {
        if (min(&t2) > 0) {
                int child = ch ^ childMask;
                if (node.type[child] == LEAF) {
                        return node.children[childMask];
                }
                if (node.type[child] == BRANCHING) {
                        int result = trace((Node *)(node.children[child]), t1, t2);
	                if (result != 0) {
                        	return result;
	                }
                }
        }

        int exitPlane = argMin(&t2);

        if ((ch & (1 << exitPlane)) != 0) {
                return 0;
        }

        ch |= (1 << exitPlane);
        double dt = t2.v[exitPlane] - t1.v[exitPlane];
        t1.v[exitPlane] = t2.v[exitPlane];
        t2.v[exitPlane] = t2.v[exitPlane] + dt;
    }
}

int render(Node * node, int i, int j, int r) {
    Vector position = {.v[0] = 0.5, .v[1] = 0.5, .v[2] = -0.2};
    Vector direction = {.v[0] = i, .v[1] = j, .v[2] = r};
    Vector t1 = {
        .v[0] = -position.x / direction.x,
        .v[1] = -position.y / direction.y,
        .v[2] = -position.z / direction.z};
    Vector t2 = {
        .v[0] = (1 - position.x) / direction.x,
        .v[1] = (1 - position.y) / direction.y,
        .v[2] = (1 - position.z) / direction.z};
    int coord = 0;

    for (coord = 0; coord < 3; coord++) {
        if (abs(direction.v[coord]) < 0.00001) {
                direction.v[coord] = 0.00001;
        }
        if (direction.v[coord] > 0) {
            t1.v[coord] = -position.v[coord] / direction.v[coord];
            t2.v[coord] = (1 - position.v[coord]) / direction.v[coord];
        } else {
            t2.v[coord] = -position.v[coord] / direction.v[coord];
            t1.v[coord] = (1 - position.v[coord]) / direction.v[coord];
        }
    }

    int result;
    if (max(&t1) < min(&t2)) {
        childMask = 0;
        for (coord = 0; coord < 3; coord++) {
            if (direction.v[coord] < 0) {
                childMask |= 1 << coord;
            }
        }
        result = trace(node, t1, t2);
    } else {
        result = 0;
    }
    return result;
}


int main(unsigned long long spu_id __attribute__ ((unused)), unsigned long long parm)
{
  tag_id = mfc_tag_reserve();
  spu_writech(MFC_WrTagMask, -1);
  
  printf("Data size: %d\n",  (int)sizeof(Node));

  // Input parameter parm is a pointer to the particle parameter context.
  // Fetch the context, waiting for it to complete.
	  
  spu_mfcdma32((void *)(&ctx), (unsigned int)parm, sizeof(spu_context), tag_id, MFC_GET_CMD);
  (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);

   int i = 0;
   int j = 0;
   for (j = ctx.y; j < ctx.y + ctx.dy; j++) {
        for (i = ctx.x; i < ctx.x + ctx.dx; i++) {
        	int res = render(
                	ctx.root,
                        i - ctx.width / 2,
                        j - ctx.heigth / 2,
                        ctx.width);
                result[i] = res;
        }
        spu_mfcdma32((void *)result, (unsigned int)(ctx.result + j * ctx.width), 
	                sizeof(int) * ctx.width, tag_id, MFC_PUT_CMD);
   }

  return 0;
}
