enum NodeType {
    LEAF,
    BRANCHING,
    EMPTY
};
                        
typedef struct {
    int type[8];
    int children[8];
} Node;

typedef struct {
  Node *root;
  int width;
  int heigth;
  int x;
  int y;
  int dx;
  int dy;
  int *result;
} spu_context;
