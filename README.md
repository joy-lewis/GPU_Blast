# GPU_Blast
GPU accelerated BLAST algorithm for DNA-sequence alignment.

### Hyperparameters
The algorithm hyperparameters are (there are more but we only describe the most important ones)

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

Our launch configuration is as follows:
- NUM_THREADS_PER_BLOCK: 128
- NUM_BLOCKS_PER_SM: 8

### File extraction
The [read_fasta()](gpu_blast.cu) function gets the raw NCBI data and parses the sequences into regular c++ char vectors 
to prepare them for the bit compression.