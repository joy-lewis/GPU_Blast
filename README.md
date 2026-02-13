# GPU_Blast
GPU accelerated BLAST algorithm for DNA-sequence alignment.
### Build Option 1: How build via Makefile
- Step 1: unzip the data file `ncbi_data.tar.gz`
- Step 2: run `make`
- Step 3: run `./gpu_blast`
### Build Option 2: How to build via CLION

- Step 1: unzip the data file `ncbi_data.tar.gz`
- Step 2: The file `CMakeLists.txt` configures the run configuration. Here the only line that needs to be adjusted is line 4 (set(CMAKE_CUDA_COMPILER /home/....) where the user needs to put down the path to their nvcc version. 
- Step 3: Then in CLION Settings, configure CMakeApplication and select gpu_blast. Then the gpu_blast.cu can be run.


### Algorithm Hyperparameters
The algorithm hyperparameters are (there are more but we only describe the most important ones):
```ccp
DB_SIZE: 10  
K: 12
TILE_CHARS: 1024
MATCH_SCORE: 1  
MISMATCH_PENALTY: -1
```

This configuration gives us k-mers of length 12 which is a common value for BLAST algorithm implementations. The number of chars (DNA Bases) 
per tile is set to 1024, this is smaller than the number of threads per block, which is important because for for better performance (especially divergence) 
we need every thread to process not one but multiple k-mers of a tile. Because the tile size is so much smaller then the average database sequence sizes 
we also have significantly more tiles than available blocks which makes the for loop (where a single block processes multiple tiles) efficient.

Unlike for protein alignment where we have more complex structures, DNA only has 4 bases so a scoring of +1 for DNA base matches and -1 for
non-matches is sufficient during the extension process.

### Launch Configuration Parameters
Our launch configuration is as follows:
```ccp
NUM_THREADS_PER_BLOCK: 128
NUM_BLOCKS_PER_SM: 8
```

### Extract DNA sequences from fasta files
The [`read_fasta()`](gpu_blast.cu) function gets the raw NCBI data and parses the sequences into regular c++ char vectors 
to prepare them for the bit compression.

### Data Compression
In [`encoder()`](gpu_blast.cu) we drastically compress the DNA sequence data because DNA only consists of 4 unique characters (bases) which is why a 2 bit encoding is sufficient
to represent each base, instead of the larger ASCII encoding. Before we start the data transfer to the device we therefore compress all of our data in this manner.
[`decoder()`](gpu_blast.cu) decompresses those bit-encodings again for our test functions.

### Lookup Table
Function [`build_lookup_table_from_encoded()`](gpu_blast.cu) builds the lookup table for the short query sequence. 
This lookup table is necessary for the threads to know if their seed k-mer has a match in the query sequence and 
at what positon that matching k-mer sits at. This saves a lot of time since the same k-mers are looked up very frequently.

## BLAST DNA Sequence Alignment

In [`ungapped_extend()`](gpu_blast.cu) and [`blast()`](gpu_blast.cu) we implemented the BLAST algorithm. For details refer to the comments 
in the code. Here is an overview of how the algorithm functions:

**Extension:**
```
FUNCTION ungapped_extend(query_shared, db_tile, q_pos, db_pos)
Initialize score = K * MATCH_SCORE

    // Step 1: Leftward Extension
    WHILE bases exist to the left AND current_score > (best_score - X_DROP):
        Extract 2-bit DNA bases from shared memory
        IF bases match: current_score += MATCH_SCORE
        ELSE: current_score += MISMATCH_PENALTY
        
        Update best_score and left_extension_length
        IF current_score falls below (best_score - X_DROP): break (Early Exit)
        
    // Step 2: Rightward Extension
    Reset current_score to best_score found from the leftward phase
    WHILE bases exist to the right AND current_score > (best_score - X_DROP):
        Compare packed bases from shared memory
        Update current_score and best_score
        Update right_extension_length
        IF current_score falls below (best_score - X_DROP): break (Early Exit)
        
    RETURN best_score and extension coordinates
```
**Main BLAST Kernel:**
```
KERNEL blast_kernel(params)
// Step 1: Pre-Processing & Query Loading
Load Query into shared memory
Initialize local hit counter and buffer in shared memory

    // Step 2: Tile-Stride Loop
    FOR each database tile (stride by number of blocks):
        Reset local hit counter
        Load Database Tile into shared memory
        
        // Step 3: Thread-Stride K-mer Search
        FOR each database position in tile (stride by number of threads):
            Build k-mer key from tile data
            Lookup matching positions in the Query Table
            
            FOR each matching Query Position:
                EXECUTE ungapped_extend() using shared memory tiles
                
                IF best_score >= MIN_REPORT_SCORE:
                    Atomically increment local hit counter
                    IF counter < MAX: Store result in local shared memory buffer
        
        // Step 4: privatized global memory update
        IF thread_id == 0:
            Atomically reserve a block of space in global memory for all block hits
        
        Copy local results to global results array
```

## Optimisations implemented within the extension function and the blast kernel

### 1.1) Streams - What failed
First we didn't use any streams and iterated over each database sequence one by one. This worked but also caused long idle times
because due to the nature of the data, memory transfer are a bottleneck here.

### 1.2) Streams - Current state
Our algorithm is bounded by memory because we have very long sequences which need to be transferred from host to device. To hide the memory latency we implemented a 3-way concurrency where we overlap Host2Device, Kernel, Device2Host and after performing some CPU work to save the alignment results to some output file.

### 2.1) Coalescing - What failed
Because we compressed the bytes originally to an unint8_t array we had bad coalescing because threads in a warp would load only the 1 byte and leave a large gap of unused memory. That's why later when we load the 2-bit compressed sequences we do this using a word size of 4 byte to not leave any gaps.

### 2.2) Coalescing - Current state
By storing the sequence data in uint32_t arrays, each threads loads 4 bytes (16 DNA bases) from global into shared memory. This ensures that adjacent threads load adjacent memory blocks, without wasting any of the loaded memory.

### 3.1) Shared Memory - What failed
We put all the relevant data into shared memory from the start except for the local hit counter which we will discuss in section 6.1 (Privatization)

### 3.2) Shared Memory - Current state
Both the query sequence, the lookup table for the query and the database sequence tile for a block are moved to shared memory. All 3 objects are needed every time a thread performs and extension, so we have a high number of redundant memory accesses. Having them in shared memory reduces the latency of that.

### 4.1) Occupancy - What failed
Previously we also launched a fixed amount of blocks which performed very poorly in the occupancy metrics. Then we checked the lecture slides again and realised that we should hardcode a fixed block number but rather make it dependent on the hardware specs to maximize utilization.

### 4.2) Occupancy - Current state
To distribute work between blocks we partitioned the full DNA sequences into tiles of a fixed size and we distributed those tiles evenly among all blocks as follows: b_j processes X_(j+i*(N_b)) , ∀ i∈ {1,2,…,TileSize/N_t }. Because we have launched 8 blocks per SM, we have more blocks than SMs to hide memory latency if for example one block in a SM is busy with a data transfer others can start processing their assigned tiles.

### 5.1) Warp Divergence - What failed
We had the idea of a Thread-Stride logic from the start, but we still had pretty significant warp divergence because we had launched to threads per block comapred to the length of the tiles we hade for any given database sequence. 

### 5.1) Warp Divergence - Current state
A fundamental limitations of implementing the BLAST algorithm on GPU is warp divergence. It can happen that one thread gets a high scoring seed which makes the thread spend more time in the extension loop while another thread in the same warp might do no extensions. This warp divergence can’t be avoided fully due to the nature of the BLAST algorithm but we tried to reduce the likelihood from this happening by using the Thread-Stride logic described in the Parallel Algorithm section. Here we assign TileSize/N_t  > 1 many k-mer query seeds to each thread in a block. In our setup this number is 8, so that work is distributed more evenly.

### 6.1) Privatization - What failed
After finishing a successful extension, each thread in each block wrote that result to a buffer in global memory using atomic add, which increased the likelihood of stalling significantly, especially with our setup where we launched hundreds of blocks.

### 6.1) Privatization - Current state
Due to our tiling logic many blocks operate on the same sequence. To avoid atomic adds to global memory, we first let the threads increment a hit counter within the shared memory of their block, and once a tile is fully processed we then write that partial hit count to global memory. The tiling logic now helps us in avoiding atomic operations on global memory.
