#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void init(int *u, int n) {
	int i, j, idx;

	int val = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			idx = j + i * n;
			u[idx] = val++;
		}
	}
}

void arr_div(int *u, int *global, int x, int y, int n) {
	int index = 0, i, j;
	for (i = x; i < x + n / 2; i++)
		for (j = y; j < y + n / 2; j++) {
			u[index++] = global[i * n + j];
		}
}

void print_array(int *u, int n) {
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			printf("%d ", u[i * n + j]);
		printf("\n");
	}
}

void print_array1d(int *u, int n) {
	int i;
	for (i = 0; i < n; i++)
		printf("%d ", u[i]);
	printf("\n");
}

int * extract_along_down(int *u, int x, int y, int n) {
	int i, index = 0;
	int * new = (int*) malloc(sizeof(int) * n);
	for (i = x; i < x + n; i++) {
		new[index++] = u[i * n + y];
	}
	return new;
}

int * extract_along_side(int *u, int x, int y, int n) {
	int j, index = 0;
	int * new = (int*) malloc(sizeof(int) * n);
	for (j = y; j < y + n; j++) {
		new[index++] = u[x * n + j];
	}
	return new;
}

int main(int argc, char *argv[]) {

	int npoints = atoi(argv[1]);
	int * test_array = (int *) malloc(npoints * npoints * sizeof(int));
	init(test_array, npoints);
	print_array(test_array, npoints);

	int * test_array_quad0 = (int *) malloc(
			npoints / 2 * npoints / 2 * sizeof(int));
	int * test_array_quad1 = (int *) malloc(
			npoints / 2 * npoints / 2 * sizeof(int));
	int * test_array_quad2 = (int *) malloc(
			npoints / 2 * npoints / 2 * sizeof(int));
	int * test_array_quad3 = (int *) malloc(
			npoints / 2 * npoints / 2 * sizeof(int));

	printf("\nQuadrant 0\n");
	arr_div(test_array_quad0, test_array, 0, 0, npoints);
	print_array(test_array_quad0, npoints / 2);

	printf("\nQuadrant 1\n");
	arr_div(test_array_quad1, test_array, 0, npoints / 2, npoints);
	print_array(test_array_quad1, npoints / 2);

	printf("\nQuadrant 2\n");
	arr_div(test_array_quad2, test_array, npoints / 2, 0, npoints);
	print_array(test_array_quad2, npoints / 2);

	printf("\nQuadrant 3\n");
	arr_div(test_array_quad3, test_array, npoints / 2, npoints / 2, npoints);
	print_array(test_array_quad3, npoints / 2);

	printf("\nRight border of quad 0\n");
	int * ext = extract_along_down(test_array_quad0, 0, npoints / 2 - 1,
			npoints / 2);
	print_array1d(ext, npoints / 2);

	printf("\nLeft border of quad 1\n");
	ext = extract_along_down(test_array_quad1, 0, 0, npoints / 2);
	print_array1d(ext, npoints / 2);

	printf("\nDown border of quad 0\n");
	ext = extract_along_side(test_array_quad0, npoints/2-1, 0, npoints / 2);
	print_array1d(ext, npoints / 2);

	printf("\nDown border of quad 1\n");
	ext = extract_along_side(test_array_quad1, npoints/2-1 , 0, npoints / 2);
	print_array1d(ext, npoints / 2);
	return 0;
}