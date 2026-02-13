# GPU_Blast
GPU accelerated BLAST algorithm for DNA-sequence alignment.

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

### 1.1) Streams - Current state
Our algorithm is bounded by memory because we have very long sequences which need to be transfered from host to device. To hide the memory latency we implemented a 3-way concurrency where we overlap Host2Device, Kernel, Device2Host and afterwards performing some CPU work to save the alignment results to some output file.

### 1.2) Streams - What failed
First we didn't use any streams and iterated over each database sequence one by one. This worked but also caused long idle times
because due to the nature of the data, memory transfer are a bottleneck here.

### 2.1) 


