#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <fstream>
#include <cstdio>
#include <iostream>

#define CHUNK_SIZE 16
#define N 128                // Grid size X
#define M 128                // Grid size Y
#define ITERATIONS 100000    // Number of iterations
#define DIFFUSION_FACTOR 0.5 // Diffusion factor
#define CELL_SIZE 0.01       // Cell size for the simulation


#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }

    
void initializeGrid(float *grid, int n, int m)
{
    for (int y = 0; y < m; ++y)
    {
        for (int x = 0; x < n; ++x)
        {
            // Initialize one quadrant to a high temp
            // and the rest to 0.
            if (y > m / 2 && x > n / 2)
            {
                grid[y * n + x] = 100.0f; // Temp in corner
            }
            else
            {
                grid[y * n + x] = 0.0f; // Temp in the rest
            }
        }
    }
}


__global__ void heatKernel(const float* input, float* output, const float dx2, const float dy2, const float dt){
    // chunk of the thread's block
    int chunk_row = blockIdx.y * blockDim.y;
    int chunk_col = blockIdx.x * blockDim.x;

    // exact coordinates of this block
    int global_row = chunk_row + threadIdx.y; // row
    int global_col = chunk_col + threadIdx.x; // column

    if (global_row >= N || global_col >= N) {return;} // prevent threads outisde of the matrix dimensions

    // declare shared memory
    __shared__ float curr[CHUNK_SIZE+2][CHUNK_SIZE+2]; // input chunk
    __shared__ float next[CHUNK_SIZE+2][CHUNK_SIZE+2]; // output chunk

    // each thread loads one element of the main chunk into shared memory
    curr[threadIdx.y+1][threadIdx.x+1] = input[global_row * N + global_col];

    // now we need to load the edge part of the chunk in order to have data for the overlapping logic
    // the base is [threadIdx.y+1][threadIdx.x+1]
    if (threadIdx.x==0){ // left border, offset 0 from row and offset -1 from column
        curr[threadIdx.y+1][threadIdx.x] = input[global_row * N + (global_col-1)];
    }
    if (threadIdx.x==CHUNK_SIZE-1){ // right border, offset 0 from row and offset +1 from column
        curr[threadIdx.y+1][threadIdx.x+2] = input[global_row * N + (global_col+1)];
    }
    if (threadIdx.y==0){ // top border, offset -1 from row and offset 0 from column
        curr[threadIdx.y][threadIdx.x+1] = input[(global_row-1) * N + global_col];
    }
    if (threadIdx.y==CHUNK_SIZE-1){ // bottom border, offset +1 from row and offset 0 from column
        curr[threadIdx.y+2][threadIdx.x+1] = input[(global_row+1) * N + global_col];
    }

    __syncthreads(); // make sure all elements are loaded by all threads before be proceed

    // skip threads which are located at the edge of the global input matrix, those cant be computed because on of the [left, right, below, above] elements would be missing
    if (global_row<=0 || global_col<=0 || global_row>=N-1 || global_col>=N-1) {return;}

    //DEBUGGING
    //if (threadIdx.x == 0 || threadIdx.x == CHUNK_SIZE-1 || threadIdx.y == 0 || threadIdx.y == CHUNK_SIZE-1) {return;}

    // compute the heat simulation approximation
    // auto left   = curr[threadIdx.y][threadIdx.x-1];
    // auto right  = curr[threadIdx.y][threadIdx.x+1];
    // auto below  = curr[threadIdx.y-1][threadIdx.x];
    // auto above  = curr[threadIdx.y+1][threadIdx.x];
    // auto center = curr[threadIdx.y][threadIdx.x];

    // Executing ONE iteration of heat flow
    auto left   = curr[threadIdx.y+1][threadIdx.x];
    auto right  = curr[threadIdx.y+1][threadIdx.x+2];
    auto below  = curr[threadIdx.y][threadIdx.x+1];
    auto above  = curr[threadIdx.y+2][threadIdx.x+1];
    auto center = curr[threadIdx.y+1][threadIdx.x+1];


    next[threadIdx.y+1][threadIdx.x+1] = center + DIFFUSION_FACTOR * dt *
                                           ((left - 2.0 * center + right) / dx2 +
                                            (above - 2.0 * center + below) / dy2);

    // synchronize before updating the curr array witht he newly computed elements
    __syncthreads();

    // thread writes back the element it owns
    output[global_row * N + global_col] = next[threadIdx.y+1][threadIdx.x+1]; // writing the result of the last iteration back to the correct place in the output array

    
    // wait till all outputs are written
    __syncthreads();
}

void writeGridAsText(const char* filename, float* grid, int width, int height)
{
    // 1. Open the output file stream.
    std::ofstream out(filename);
    if (!out)
    {
        // Use cerr for error messages
        std::cerr << "Failed to open file for grid text output: " << filename << "\n";
        return;
    }
    
    // Set output formatting for readability:
    // fixed: use fixed-point notation
    // setprecision(2): two decimal places
    // setw(7): set a minimum width of 7 characters for alignment
    out << std::fixed << std::setprecision(2);
    
    // 2. Iterate through rows (y) and columns (x) to format the output.
    for (int y = 0; y < height; ++y) // Outer loop handles rows (height)
    {
        for (int x = 0; x < width; ++x) // Inner loop handles columns (width)
        {
            // Calculate the 1D index: index = y * width + x
            float value = grid[y * width + x];
            
            // Write the value with padding
            out << std::setw(7) << value;
        }
        // Move to the next line after completing a row
        out << "\n";
    }

    // 3. Close the file.
    out.close();
    std::cout << "Successfully wrote grid contents to: " << filename << "\n";
}


int main()
{
    // Allocate memory for the grids
    float *h_in = (float *)malloc(N * M * sizeof(float));
    float *h_output = (float *)malloc(N * M * sizeof(float));
    float *d_in, *d_output;

    CHECK_CUDA(cudaMalloc(&d_in, N*M*sizeof(float)))
    CHECK_CUDA(cudaMalloc(&d_output, N*M*sizeof(float)))

    // Check for allocation failures
    if (h_in == NULL)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize the grids
    initializeGrid(h_in, N, M);

    writeGridAsText("heatmap_in.txt", h_in, N, M);

    CHECK_CUDA(cudaMemcpy(d_in, h_in, N*M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_output, h_output, N*M*sizeof(float), cudaMemcpyHostToDevice));

    float dx2 = CELL_SIZE * CELL_SIZE;
    float dy2 = CELL_SIZE * CELL_SIZE;
    float dt = dx2 * dy2 / (2.0 * DIFFUSION_FACTOR * (dx2 + dy2));

    // Run the heat simulation
    dim3 threadsPerBlock(CHUNK_SIZE, CHUNK_SIZE);
    dim3 blocksPerGrid(sqrt((N * M) / (CHUNK_SIZE*CHUNK_SIZE)), sqrt((N * M) / (CHUNK_SIZE*CHUNK_SIZE)));

    std::cout << "Launch kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";
    for (int i=0; i<ITERATIONS; i++){
        heatKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_output, dx2, dy2, dt);
        std::swap(d_in, d_output);  // the output from the current iteration becomes the input for the next one
    }

    CHECK_CUDA(cudaMemcpy(h_output, d_output, N*M * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a small section of the final grid for verification
    std::cout << "Final grid values (top-left corner):" << std::endl;
    for (int y = 0; y < 16; ++y)
    {
        for (int x = 0; x < 16; ++x)
        {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << h_output[y * N + x] << " ";
        }
        std::cout << std::endl;
    }

    // plot heatmap for interpretability
    writeGridAsText("heatmap_out.txt", h_output, N, M);

    // Free allocated memory
    cudaFree(d_in);
    cudaFree(d_output);
    free(h_in);
    free(h_output);

    return 0;
}
