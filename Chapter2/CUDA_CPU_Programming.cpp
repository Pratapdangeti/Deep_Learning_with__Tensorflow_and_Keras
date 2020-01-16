#include <stdio.h>
#define SIZE	1024

// vectorAdd function used with just void keyword at start indicates 
// this is just CPU function

void VectorAdd(int *a, int *b, int *c, int n)
{
	int i;
	// for loop is used as the function does run on cpu only
	for (i = 0; i < n; ++i)
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;

	// malloc is the memory allocation function for CPU programming

	a = (int *)malloc(SIZE * sizeof(int));
	b = (int *)malloc(SIZE * sizeof(int));
	c = (int *)malloc(SIZE * sizeof(int));

	// for loop used for creating some values for variables

	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	// initiating cpu function

	VectorAdd(a, b, c, SIZE);

	// print the result of first 10 result variable after addition
	printf("CPU Code\n");
	for (int i = 0; i < 10; ++i)
		printf("c[%d] = %d\n", i, c[i]);

	// memory needs to be freed post execution from CPU

	free(a);
	free(b);
	free(c);

	return 0;
}
