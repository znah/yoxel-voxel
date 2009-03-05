#ifndef VECTOR_H_
#define VECTOR_H_

union Vector3d {
	struct {
	double x;
	double y;
	double z;
	};
	double v[3];
};

typedef union Vector3d Vector;

double max(Vector *v) {
	if (v->x > v->y) {
		if (v->x > v->z) {
			return v->x;
        } else {
        	return v->z;
        }
	} else {
		if (v->y > v->z) {
			return v->y;
		} else {
			return v->z;
		}
	}
}

double min(Vector *v) {
	if (v->x < v->y) {
		if (v->x < v->z) {
			return v->x;
        } else {
        	return v->z;
        }
	} else {
		if (v->y < v->z) {
			return v->y;
		} else {
			return v->z;
		}
	}
}

int argMin(Vector *v) {
	if (v->x < v->y) {
		if (v->x < v->z) {
			return 0;
        } else {
        	return 2;
        }
	} else {
		if (v->y < v->z) {
			return 1;
		} else {
			return 2;
		}
	}
}

Vector *add(Vector * v1, Vector * v2) {
	v1->x += v2->x;
	v1->y += v2->y;
	v1->z += v2->z;
	return v1;
}

Vector *div2(Vector * v1, double v) {
	v1->x /= v;
	v1->y /= v;
	v1->z /= v;
	return v1;
}
    


#endif /*VECTOR_H_*/
