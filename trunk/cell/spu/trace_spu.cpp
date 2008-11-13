#include "stdafx.h"

#include "trace_spu.h"

int tag_id;
volatile trace_spu_params params __attribute__ ((aligned (16)));



int main(unsigned long long spu_id __attribute__ ((unused)), unsigned long long parm)
{
  printf("spu - %u %d\n", (uint)parm % 16, sizeof(params));

  tag_id = mfc_tag_reserve();
  spu_writech(MFC_WrTagMask, -1);
  
  spu_mfcdma32((void *)(&params), (unsigned int)parm, sizeof(trace_spu_params), tag_id, MFC_GET_CMD);
  (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);

  printf("y start: %d\n", params.start.y);

/*   int i = 0;
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
   }*/
/*  for (int y = params.start.y; y < params.end.y; ++y)
    for (int x = params.start.x; x < params.end.x; ++x)
    {
      int offs = y*params.viewSize.x + x;


    }*/

  return 0;
}
