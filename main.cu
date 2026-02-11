#include <iostream>
#include <cstring>

// Declare the other functions (prototypes) so this file knows they exist
int blast_main();
int test_main();

int main(int argc, char** argv) {
    //// TEST
    //test_main();

    //// MAIN BLAST ALGORITHM
    blast_main();

    return 0;
}

// CODE EXECUTION:
// running the algorithm with nsight tool
// -> nsys profile -o report_name ./gpu_blast

// CREATING CSVs:
// Kernel execution
// -> nsys stats --report gpukernsum --format csv --output . report_name.nsys-rep
// Kernel Details
// -> ncu --csv --set detailed ./gpu_blast > kernel_analysis.csv
// Memory Transfer:
// -> nsys stats --report gpumemtimesum --format csv --output . report_name.nsys-rep
// Cuda API (cuaMalloc etc.)
// -> nsys stats --report cudaapisum --format csv --output . report_name.nsys-rep