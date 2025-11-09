#include <arrayfire.h>
// Generate random data, sum and print the result.
int main(void)
{
    // Generate 10,000 random values
    af::array a = af::randu(10000);
 
    // Sum the values and copy the result to the CPU:
    double sum = af::sum<float>(a);
 
    printf("sum: %g\n", sum);
    return 0;
}