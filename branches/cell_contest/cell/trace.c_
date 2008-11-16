#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <libspe2.h>
#include <pthread.h>
#include "data.h"

extern spe_program_handle_t trace_spu;
#define MAX_NODES               1000000
#define MAX_SPU_THREADS 	16

Node nodesBuff[MAX_NODES] __attribute__ ((aligned (128)));
int nodesCount = 0;

typedef struct ppu_pthread_data {
  spe_context_ptr_t spe_ctx;
  void *argp;
} ppu_pthread_data_t; 

Node * readNode(FILE *file) {
	if (nodesCount == MAX_NODES) {
		printf("Not enought nodes!!!\n");
		exit(0);
	}
        Node *result = nodesBuff + (nodesCount++);
        int i;
        for (i = 0; i < 8; i++) {
                int ch = fgetc(file);
                if (ch == 'b') {
                        result->type[i] = BRANCHING;
                        result->children[i] = (int)readNode(file);
                }
                if (ch == 'l') {
                        result->type[i] = LEAF;
                        int color = 0;
                        color += fgetc(file) << 16;
                        color += fgetc(file) << 8;
                        color += fgetc(file);
                        result->children[i] = color;
                }
                if (ch == 'e') {
                        result->type[i] = EMPTY;
                }
        }
        return result;
}

void *ppu_pthread_function(void *arg) {
    ppu_pthread_data_t *datap = (ppu_pthread_data_t *)arg;
    unsigned int entry = SPE_DEFAULT_ENTRY;
    if (spe_context_run(datap->spe_ctx, &entry, 0, datap->argp, NULL, NULL) < 0) {
        perror ("Failed running context");
        exit (1);
    }
    pthread_exit(NULL);
} 


int main()
{
  int i, spu_threads;
  ppu_pthread_data_t datas[MAX_SPU_THREADS];
  spe_context_ptr_t ctxs[MAX_SPU_THREADS];
  pthread_t threads[MAX_SPU_THREADS]; 
  spu_context ctx[MAX_SPU_THREADS] __attribute__ ((aligned (128)));

  /* Determine the number of SPE threads to create.
   */
  spu_threads = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
  if (spu_threads > MAX_SPU_THREADS) spu_threads = MAX_SPU_THREADS;

  FILE *f = fopen("out.cube", "r");
  if (f == NULL) {
    printf("No input file ");
    return 0;
  }
  fgetc(f);
  Node *root = readNode(f);
  printf("Nodes readed.\n");

  int width = 800;
  int height = 800;
  int result[800 * 800] __attribute__ ((aligned (128)));

  for(i = 0; i < spu_threads; i++) {
    ctx[i].root = root;
    ctx[i].width = width;
    ctx[i].heigth = height;
    ctx[i].dx = width;
    ctx[i].dy = height / spu_threads;
    ctx[i].x = 0;
    ctx[i].y = ctx[i].dy * i;
    ctx[i].result = result;
  }

  /* Create several SPE-threads to execute 'simple_spu'.
   */
  for(i=0; i<spu_threads; i++) {
    /* Create context */
    if ((ctxs[i] = spe_context_create (0, NULL)) == NULL) {
      perror ("Failed creating context");
      exit (1);
    }
    /* Load program into context */
    if (spe_program_load (ctxs[i], &trace_spu)) {
      perror ("Failed loading program");
      exit (1);
    }
    /* Initialize context run data */
    datas[i].spe_ctx = ctxs[i];
    datas[i].argp = (void *)&ctx[i]; 
    

    /* Create thread for each SPE context */
    if (pthread_create (&threads[i], NULL, &ppu_pthread_function, &datas[i]))  {
      perror ("Failed creating thread");
      exit (1);
    }
  }

  /* Wait for SPU-thread to complete execution.  */
  for (i=0; i<spu_threads; i++) {
    if (pthread_join (threads[i], NULL)) {
      perror("Failed pthread_join");
      exit (1);
    }

    /* Destroy context */
    if (spe_context_destroy (ctxs[i]) != 0) {
      perror("Failed destroying context");
      exit (1);
    }
  }
  
  FILE *out = fopen("input.txt", "w+");
  fprintf(out, "%d %d\n", width, height);
  for     (i = 0; i < width * height; i++) {
  	fprintf(out, "%d ", result[i]);
  }

  printf("\nThe program has successfully executed.\n");

  return (0);
}
