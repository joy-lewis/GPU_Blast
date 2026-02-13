#include <vector>
#include <cstdint>
#include <utility>
#include <iostream>
#include <cassert>
#include <cstring>
#include <bitset>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>
#include <cctype>
#include <filesystem>
#include <algorithm>
#include "gpu_blast.h"
#include <cuda_runtime.h>


///////////
/////////// Define Algorithm Hyperparameters
///////////
#define DB_SIZE 10   // number of sequences in Database
#define K 12   // defines k-mer length // conditions: K>0
#define MAX_LOCAL_HITS 256   // maximum number of hits that fit into the local hit buffer within the shared memory of a block
#define TILE_CHARS 1024u // length of a DB sequence tile in characters (DNA bases)

#define MATCH_SCORE 1  // reward for finding a matching DNA base
#define MISMATCH_PENALTY -1  // reward for finding a non-matching DNA base
#define MIN_REPORT_SCORE K+1 // to only store the alignment results which are longer then the k-mer seed itself
#define X_DROP 4 // X-drop termination condition to keep the extension finite


///////////
/////////// Define Launch Configuration Parameters
///////////
#define NUM_THREADS_PER_BLOCK 128
#define NUM_BLOCKS_PER_SM 8


///////////
/////////// Sanity checks
///////////
#define CHECK_CUDA(call)                                            \
    {                                                               \
        cudaError_t err = (call);                                   \
        if ((call) != cudaSuccess)                                  \
        {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << ": "           \
        << cudaGetErrorString(err) << std::endl;                    \
        exit(EXIT_FAILURE);                                         \
        }                                                           \
    }


///////////
/////////// Extract DNA sequences from fasta files
///////////
std::vector<char> read_fasta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);

    std::vector<char> seq;
    std::string line;

    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '>') { // skip header
            continue;
        }
        for (unsigned char ch : line) { // parse line by line
            if (std::isspace(ch)) {
                continue;
            }
            ch = (unsigned char)std::toupper(ch);
            if (ch=='A' || ch=='C' || ch=='G' || ch=='T') { // all 4 possible DNA bases
                seq.push_back((char)ch);
            }
        }
    }
    return seq;
}

///////////
/////////// Data Compression
///////////
uint8_t encode_char(char c) {
    // Compressing the ASCII characters (4 unique DNA bases) to a 2-bit encoding
    switch (c) {
        case 'A': return 0b00;
        case 'C': return 0b01;
        case 'G': return 0b10;
        case 'T': return 0b11;
        default: assert(false); // should not happen
        }
    return 0;
}

char decode_bits(uint8_t b) {
    // Decompressing the 2-bit encoding back to an ASCII symbol
    switch (b) {
        case 0b00: return 'A';
        case 0b01: return 'C';
        case 0b10: return 'G';
        case 0b11: return 'T';
        default: assert(false); // should not happen
        }
    return 0;
}


void encoder(const std::vector<char>&input, size_t length, uint8_t* output) {
    // Converting DNA sequence to bit array
    // Layout for a 4 base word in an 8-bit encoding:
    //      base 0 is in bits 7–6,
    //      base 1 in bits 5–4,
    //      base 2 in bits 3–2,
    //      base 3 in bits 1–0

    size_t in_index = 0;
    size_t out_index = 0;

    while (in_index < length) {
        uint8_t new_byte = 0;

        // Store 4 bases into 1 byte, in the manner of MSB-first, i.e. left shift byte to make room for next DNA base
        for (int i = 0; i < 4; i++) {
            new_byte <<= 2;
            if (in_index < length) {
                uint8_t encoding = encode_char(input[in_index]);
                new_byte |= encoding;
                ++in_index;
            }
        }
        output[out_index++] = new_byte;
    }
}


void decoder(const uint8_t* input, size_t length, char* output) { //length == Number of characters that are encoded in the input bits (relevant because last byte might encode less than 4 characters)
    // Revert the compression from 2-bit back to ASCII characters

    int in_index = 0;
    int out_index = 0;

    while (out_index < length) { // as long as there are characters left we keep on decoding
        uint8_t byte = input[in_index];

        // Encode the next 4 characters
        for (int shift = 6; shift >= 0; shift -=2) {
            // We right shift by 6, 4, 2 and then 0 to have each bit pair at the rightmost edge of the byte
            // then we do AND with 0b11 to isolate the two rightmost bits we care about
            uint8_t s_byte = byte>>shift;
            s_byte = s_byte&0b11;

            if (out_index < length) {
                uint8_t decoding = decode_bits(s_byte);
                output[out_index++] = decoding;
            }
        }
        in_index++;
    }
}


///////////
/////////// Lookup Table Creation
///////////

// Here is an example describing the logic that this query lookup table is build on:
// indices 0...43 belong to k-mer i at 0x11111
// indices 44...46 belong to k-mer j at 0x22222

// offsets[k-mer i] == start index in positions array for k-mer i
// offsets[k-mer i+1] == end index in positions array for k-mer i

// "positions array are sorted query sequence positions"

// all possible k-mers are within [0, 2^(2*K)], because every character is 2 bits long (2*K) and there are 2 possible bits
// simplifies to [0, 4^K]

uint32_t base_at_msb(const uint8_t* encoded_dna, uint32_t i) {
    // This function extracts the two bits of the i-th DNA base from an 8-bit (byte) compression

    const uint32_t byte_idx = i >> 2;                // identify which byte in the encoded_dna array contains the i-th DNA base
    const uint32_t in_byte  = i & 3u;                // computes the DNA bases`s positon within the byte
    const uint32_t shift    = 6u - 2u * in_byte;     // 6,4,2,0 (MSB -> LSB), i.e. we shift the bit pair we are interested in to the right most positon
    return (encoded_dna[byte_idx] >> shift) & 0x3u;  // masking with 0x3u to extract the bit pair
}

uint32_t kmer_at_msb_bytes(const uint8_t* encoded_dna, uint32_t pos, uint32_t k) {
    // Returns a 2k-bit key where the leftmost base gets the most significant 2 bits in the encoding
    // Example: ACGT => 00 01 10 11 => 0b00011011

    uint32_t key = 0;
    for (uint32_t j = 0; j < k; ++j) {
        key = (key << 2) | base_at_msb(encoded_dna, pos + j);   // build the key from left to right
    }
    return key;
}


LookupTable build_lookup_table_from_encoded(const uint8_t* encoded_dna, uint32_t N, uint32_t k){
    // Builds the lookup table from the encoding `encoded_dna` for k-mer size K and a query sequence of length N

    if (k > 16) { // Sanity check for k. We can only fit k*2 bits in the keys of the lookup table
        throw std::invalid_argument("k too large for 32-bit key / table size");
    }

    const uint32_t L = N - k + 1u;      // number of k-mers in query
    const uint32_t M = 1u << (2u * k);  // number of possible k-mers (4^k)

    // 1) Count occurrences per k-mer
    std::vector<uint32_t> counts(M, 0);
    for (uint32_t i = 0; i < L; ++i) {
        const uint32_t key = kmer_at_msb_bytes(encoded_dna, i, k);
        ++counts[key];
    }

    // 2) Sum up the k-mer offset position with the hits to get the range of positons for this k-mer (as described in the example)
    std::vector<uint32_t> offsets(M + 1u, 0);
    for (uint32_t key = 0; key < M; ++key) {
        offsets[key + 1u] = offsets[key] + counts[key];
    }

    // 3) Fill positions (sorted per k-mer key because i increases)
    std::vector<uint32_t> cursor(offsets.begin(), offsets.begin() + M); // the end is the number of all possible k-mers
    std::vector<uint32_t> positions(L);

    for (uint32_t i = 0; i < L; ++i) {
        const uint32_t key = kmer_at_msb_bytes(encoded_dna, i, k);
        const uint32_t out = cursor[key]++;
        positions[out] = i;
    }

    return LookupTable{std::move(offsets), std::move(positions)};
}


LookupTableView lookup_table_to_device(const LookupTable& t, uint32_t** d_offsets_out, uint32_t** d_positions_out) {
    // Handles memory allocation on device and copies the lookup table contents over to the device

    // Allocating memory for the offsets and positions is enough, the rest of the struct are only single numbers
    uint32_t* d_offsets;
    uint32_t* d_positions;

    CHECK_CUDA(cudaMalloc(&d_offsets, t.offsets.size() * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_positions, t.positions.size() * sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(d_offsets, t.offsets.data(),
                          t.offsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_positions, t.positions.data(),
                          t.positions.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Save the pointers to the allocated memory to the lookup table struct
    // so we can hand it over to the kernel function later
    *d_offsets_out = d_offsets;
    *d_positions_out = d_positions;

    return LookupTableView{
        d_offsets,
        d_positions,
        (uint32_t)t.offsets.size(),
        (uint32_t)t.positions.size()
    };
}


///////////
/////////// BLAST DNA Alignment
///////////

__device__ __forceinline__ uint32_t base_at_msb_device(const uint8_t* encoded_dna, uint32_t i) {
    // *Same as above but only to be called from within the kernel*

    // This function extracts the two bits of the i-th DNA base from an 8-bit (byte) compression

    const uint32_t byte_idx = i >> 2;                // identify which byte in the encoded_dna array contains the i-th DNA base
    const uint32_t in_byte  = i & 3u;                // computes the DNA bases`s positon within the byte
    const uint32_t shift    = 6u - 2u * in_byte;     // 6,4,2,0 (MSB -> LSB), i.e. we shift the bit pair we are interested in to the right most positon
    return (encoded_dna[byte_idx] >> shift) & 0x3u;  // masking with 0x3u to extract the bit pair
}


__device__ __forceinline__  uint32_t kmer_at_msb_bytes_device(const uint8_t* encoded_dna, uint32_t pos, uint32_t k) {
    // *Same as above but only to be called from within the kernel*

    // Returns a 2k-bit key where the leftmost base gets the most significant 2 bits in the encoding
    // Example: ACGT => 00 01 10 11 => 0b00011011

    uint32_t key = 0;
    for (uint32_t j = 0; j < k; ++j) {
        key = (key << 2) | base_at_msb_device(encoded_dna, pos + j);   // build the key from left to right
    }
    return key;
}


// Shared-tile accessor with global fallback
struct Tile{
    const uint8_t* db_global;
    const uint8_t* tile_shared;  // packed bytes for tile
    uint32_t startChar;      // start char index in global db
    uint32_t nChars;          // number of valid chars in tile
};

__device__ __forceinline__ uint32_t tile_char_at(const Tile& acc, uint32_t dbCharIdx)
{
    // If inside tile, access shared; else access global
    if (dbCharIdx >= acc.startChar && dbCharIdx < (acc.startChar + acc.nChars)) {
        uint32_t local = dbCharIdx - acc.startChar;
        return base_at_msb_device(acc.tile_shared, local);
    } else {
        return base_at_msb_device(acc.db_global, dbCharIdx);
    }
}


// Simple ungapped extension around a seed
__device__ __forceinline__ void ungapped_extend(
    const uint8_t* q_shared,
    const Tile& dbAcc,
    uint32_t qLenChars,
    uint32_t dbLenChars,
    uint32_t qSeedPos,
    uint32_t dbSeedPos,
    int32_t& outBestScore,
    int32_t& outLeftExt,
    int32_t& outRightExt){

    // Seed is exact match by construction
    int32_t score = (int32_t)K * MATCH_SCORE;
    int32_t best  = score;

    int32_t bestLeft  = 0;
    int32_t bestRight = 0;

    // Left extension
    int32_t cur = score;
    int32_t leftExt = 0;
    int32_t iQ  = (int32_t)qSeedPos - 1;
    int32_t iDB = (int32_t)dbSeedPos - 1;
    while (iQ >= 0 && iDB >= 0) {
        const uint32_t qb  = base_at_msb_device(q_shared, (uint32_t)iQ);
        const uint32_t dbb = tile_char_at(dbAcc, (uint32_t)iDB);
        cur += (qb == dbb) ? MATCH_SCORE : MISMATCH_PENALTY;
        leftExt++;

        if (cur > best) {
            best = cur;
            bestLeft = leftExt;
        }
        if (cur < best - X_DROP) break;
        --iQ; --iDB;
    }

    // Right extension
    cur = best; // we continue from best-so-far of previous left extension
    int32_t rightExt = 0;
    iQ  = (int32_t)(qSeedPos + K);
    iDB = (int32_t)(dbSeedPos + K);
    while (iQ < (int32_t)qLenChars && iDB < (int32_t)dbLenChars) {
        const uint32_t qb  = base_at_msb_device(q_shared, (uint32_t)iQ);
        const uint32_t dbb = tile_char_at(dbAcc, (uint32_t)iDB);
        cur += (qb == dbb) ? MATCH_SCORE : MISMATCH_PENALTY;
        rightExt++;

        if (cur > best) {
            best = cur;
            bestRight = rightExt;
        }
        if (cur < best - X_DROP) break;

        ++iQ; ++iDB;
    }

    outBestScore = best;
    outLeftExt   = bestLeft;
    outRightExt  = bestRight;
}


// Main BLAST kernel who is responsible for the tiling process
__global__ void blast(KernelParamsView params)
{
    // To not exceed our thread budget
    if (blockDim.x != NUM_THREADS_PER_BLOCK) return;

    const uint32_t qLen = params.query.nChars;
    const uint32_t dbLen = params.database.nChars;
    const uint32_t qBytes = params.query.nBytes;

    // Shared memory layout:
    // [ query_bytes ][ tile_bytes ]
    extern __shared__ uint8_t shared_mem[];

    // 1. Declare Privatized Shared Variables [cite: 83, 151]
    __shared__ uint32_t blockHitCount;
    __shared__ uint32_t globalBaseIdx;
    __shared__ Hit localHitBuffer[MAX_LOCAL_HITS];

    // Round qBytes up to the nearest multiple of 4 to ensure the NEXT pointer is aligned
    const uint32_t qBytesAligned = (params.query.nBytes + 3) & ~0x03;
    uint8_t* q_sh = shared_mem;
    uint8_t* tile_sh = q_sh + qBytesAligned; // tile_sh is now guaranteed 4-byte aligned

    // 1) Load query into shared (all blocks do this in parallel)
    for (uint32_t i = threadIdx.x; i < qBytes; i += blockDim.x) {
        q_sh[i] = params.query.seq[i];
    }
    __syncthreads(); // wait till the full query is loaded into shared memory

    // 2) Every block has its pre-determined set of tiles: tileId = blockIdx.x
    //  which gets updated in each iteration with tileId = tileId + gridDim.x
    const uint32_t tilesTotal = (dbLen + TILE_CHARS - 1u) / TILE_CHARS;

    // incrementing the tileId by the grid dimension X ensures that we only process a DB tile sequence only once
    // -> this implements the tile-stride logic
    for (uint32_t tileId = (uint32_t) blockIdx.x; tileId < tilesTotal; tileId += (uint32_t) gridDim.x) {

        // Reset local counter for the new tile
        if (threadIdx.x == 0) blockHitCount = 0;
        __syncthreads();

        const uint32_t tileStartChar = tileId * TILE_CHARS;
        const uint32_t tileChars = min(TILE_CHARS, dbLen - tileStartChar);
        const uint32_t tileBytes = (tileChars + 3u) >> 2;
        const uint32_t tileStartByte = tileStartChar >> 2;

        // 2a) loading this tile’s encoded bytes into shared memory using vectorized loads
        // This addresses the "UncoalescedGlobalAccess" bottleneck in your Nsight profile.
        // each thread loads 4 bytes (16 DNA bases) which ensures that adjacent threads load adjacent memory
        uint32_t* tile_sh_32 = (uint32_t*)tile_sh;
        uint32_t* db_global_32 = (uint32_t*)&params.database.seq[tileStartByte];

        // we divide tileBytes by 4 because each thread now loads 4 bytes (32 bits, 16 DNA bases) at once
        // -> this implements the k-mer stride logic within a tile
        for (uint32_t i = threadIdx.x; i < (tileBytes + 3) / 4; i += blockDim.x) {
            tile_sh_32[i] = db_global_32[i]; // Coalesced 32-bit load
        }
        __syncthreads(); // Wait until the entire tile is loaded into shared memory

        // Define current tile
        Tile dbAcc;
        dbAcc.db_global = params.database.seq;
        dbAcc.tile_shared = tile_sh;
        dbAcc.startChar = tileStartChar;
        dbAcc.nChars = tileChars;

        // 3) Each thread processes start positions in this tile: start = tileStartChar + tid
        // then += blockDim.x (i.e. each thread pulls its next seed with a offset of NUM_THREADS_PER_BLOCK)
        for (uint32_t dbPos = tileStartChar + (uint32_t) threadIdx.x;
             dbPos < tileStartChar + tileChars;
             dbPos += (uint32_t) blockDim.x) {
            // sanity checks
            if (dbPos + K > dbLen) continue;
            if (K == 0 || qLen < K) continue;

            // Build k-mer key from database sequence at dbPos.
            // Uses shared tile for speed when inside tile; but kmer_at expects a pointer.
            // So we compute by reading character-by-character via tile_char_at().
            uint32_t key = 0;
            //#pragma unroll
            for (uint32_t j = 0; j < 32; ++j) {
                if (j >= K) break;
                key = (key << 2) | tile_char_at(dbAcc, dbPos + j);
            }

            // Finding matches for this k-mer in query lookup table
            if (key + 1u >= params.lView.nOffsets) continue; // to avoid out of bounds errors
            const uint32_t start = params.lView.offsets[key];
            const uint32_t end   = params.lView.offsets[key + 1u];

            // We perform an extension for each match position in query
            for (uint32_t idx = start; idx < end; ++idx) {
                if (idx >= params.lView.nPositions) break;
                const uint32_t qPos = params.lView.positions[idx];

                if (qPos + K > qLen) continue; // to avoid out of bounds errors

                int32_t bestScore, leftExt, rightExt;
                ungapped_extend(q_sh, dbAcc, qLen, dbLen, qPos, dbPos,
                                bestScore, leftExt, rightExt);

                if (bestScore >= MIN_REPORT_SCORE) {
                    // 1. Privatized Atomic: Add to shared memory counter [cite: 26, 34]
                    uint32_t localIdx = atomicAdd(&blockHitCount, 1u);

                    // 2. Staging: Store hit in shared memory buffer [cite: 151, 155]
                    if (localIdx < MAX_LOCAL_HITS) {
                        Hit h;
                        h.db_pos = dbPos;
                        h.q_pos = qPos;
                        h.bestScore = bestScore;
                        h.leftExt = leftExt;
                        h.rightExt = rightExt;
                        localHitBuffer[localIdx] = h;
                    }
                }
            }
        }

        // 2. COMMIT THE TILE RESULTS ONCE
        __syncthreads(); // Wait for all threads to finish the ENTIRE tile [cite: 141, 158]

        if (threadIdx.x == 0 && blockHitCount > 0) {
            uint32_t numToCopy = min(blockHitCount, (uint32_t)MAX_LOCAL_HITS);
            globalBaseIdx = atomicAdd(params.hitCount, numToCopy); // Single global atomic per tile [cite: 45]
        }
        __syncthreads();

        // 3. COALESCED WRITE TO GLOBAL
        uint32_t numToCopy = min(blockHitCount, (uint32_t)MAX_LOCAL_HITS);
        for (uint32_t i = threadIdx.x; i < numToCopy; i += blockDim.x) {
            if (globalBaseIdx + i < params.maxHits) {
                params.hits[globalBaseIdx + i] = localHitBuffer[i]; // Coalesced access [cite: 112, 119]
            }
        }
        __syncthreads();
    }
}


// Updated to accept host pointers instead of device pointers
void save_results(uint32_t hitCount, uint32_t maxHits, const Hit* h_hits_src, int si) {
    // 4) hitCount is already passed as a value from host pinned memory
    // No cudaMemcpy needed here!

    // Copy only as many hits as fit in the allocated buffer
    uint32_t nToCopy = std::min(hitCount, maxHits);

    // 5) Access data directly from the host pinned pointer (h_hits_src)
    // We create a local vector to sort without modifying the original pinned buffer
    std::vector<Hit> sorted_hits(nToCopy);
    if (nToCopy > 0) {
        std::memcpy(sorted_hits.data(), h_hits_src, nToCopy * sizeof(Hit));

        // sort alignments by length (Theme: Performance Analysis)
        std::sort(sorted_hits.begin(), sorted_hits.end(),
                  [](const Hit& a, const Hit& b) {
                      if (a.bestScore != b.bestScore) return a.bestScore > b.bestScore;
                      if (a.db_pos != b.db_pos) return a.db_pos < b.db_pos;
                      return a.q_pos < b.q_pos;
                  });
    }

    // Save results to txt file (Theme: CUDA Program Structure [cite: 225])
    namespace fs = std::filesystem;
    fs::create_directories("results");

    std::string outName = (fs::path("results") /
                               ("blast_results_sequence" + std::to_string(si) + ".txt")).string();
    std::ofstream out(outName, std::ios::out);

    if (out) {
        out << "DB sequence index: " << si << "\n";
        out << "Device hitCount: " << hitCount << "\n";
        out << "Hits written: " << nToCopy << "\n";
        out << "Columns: db_pos q_pos bestScore leftExt rightExt\n";

        for (uint32_t i = 0; i < nToCopy; ++i) {
            const Hit& h = sorted_hits[i];
            out << h.db_pos << " " << h.q_pos << " " << h.bestScore << " "
                << h.leftExt << " " << h.rightExt << "\n";
        }
    }
    else {
        out << "No results!";
    }
}


////////////////////////////
////////// MAIN FUNCTION //
///////////////////////////

/// 3-way concurrent loop
///
///
///

int blast_main() {
    // 1. Hardware Discovery (Do this once)
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    // We launch 16 blocks per SM to ensure high occupancy and more blocks then SMs available to avoid idle standing SMs
    const int blocksPerSM = NUM_BLOCKS_PER_SM;
    dim3 blocksPerGrid(props.multiProcessorCount * blocksPerSM, 1);
    dim3 threadsPerBlock(NUM_THREADS_PER_BLOCK, 1);

    // Add this after Hardware Discovery (Step 4)
    std::cout << "[GPU Config] Using Device: " << props.name << "\n";
    std::cout << "[GPU Config] MultiProcessors: " << props.multiProcessorCount << "\n";
    std::cout << "[GPU Config] Launching " << blocksPerGrid.x << " blocks of " << threadsPerBlock.x << " threads.\n";

    // 2. Query Setup (Pinned Memory) to hide data transfer cost
    std::vector<char> query_seq = read_fasta("ncbi_data/query.fasta");
    uint32_t q_nChars = query_seq.size();
    uint32_t q_nBytes = (q_nChars + 3) / 4;

    uint8_t* h_q_pinned;
    CHECK_CUDA(cudaMallocHost(&h_q_pinned, q_nBytes));
    encoder(query_seq, q_nChars, h_q_pinned);

    uint8_t* d_query;
    CHECK_CUDA(cudaMalloc(&d_query, q_nBytes));
    CHECK_CUDA(cudaMemcpy(d_query, h_q_pinned, q_nBytes, cudaMemcpyHostToDevice));
    SeqView q_view { d_query, q_nBytes, q_nChars };

    LookupTable lTable = build_lookup_table_from_encoded(h_q_pinned, q_nChars, K);
    uint32_t *d_off, *d_pos;
    LookupTableView lView = lookup_table_to_device(lTable, &d_off, &d_pos);

    // 3. PRE-PROCESSING: Encode all 10 sequences into Pinned Memory
    // This avoids the synchronous bottlenecks
    uint8_t* h_db_pinned_array[DB_SIZE];
    uint32_t db_nBytes_array[DB_SIZE];
    uint32_t db_nChars_array[DB_SIZE];

    for (int si = 1; si <= DB_SIZE; si++) {
        std::string path = "ncbi_data/sequence" + std::to_string(si) + ".fasta";
        std::vector<char> db_seq = read_fasta(path);
        db_nChars_array[si-1] = db_seq.size();
        db_nBytes_array[si-1] = (db_nChars_array[si-1] + 3) / 4;

        CHECK_CUDA(cudaMallocHost(&h_db_pinned_array[si-1], db_nBytes_array[si-1]));
        encoder(db_seq, db_nChars_array[si-1], h_db_pinned_array[si-1]);
    }

    // 4. Concurrency Setup (Streams and Multi-Buffers)
    const int nStreams = 4;
    cudaStream_t streams[nStreams];
    uint8_t* d_db_buffers[nStreams];
    Hit* d_hits_buffers[nStreams];
    uint32_t* d_hitCount_buffers[nStreams];

    // Pinned result buffers (one set per sequence for final processing)
    uint32_t MAX_HITS = 1u << 20;
    uint32_t* h_hitCounts;
    Hit** h_all_hits;
    CHECK_CUDA(cudaMallocHost(&h_hitCounts, DB_SIZE * sizeof(uint32_t)));
    CHECK_CUDA(cudaMallocHost(&h_all_hits, DB_SIZE * sizeof(Hit*)));

    for (int i = 0; i < nStreams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
        CHECK_CUDA(cudaMalloc(&d_db_buffers[i], 100 * 1024 * 1024)); // 100MB
        CHECK_CUDA(cudaMalloc(&d_hits_buffers[i], MAX_HITS * sizeof(Hit)));
        CHECK_CUDA(cudaMalloc(&d_hitCount_buffers[i], sizeof(uint32_t)));
    }
    for (int si = 0; si < DB_SIZE; si++) {
        CHECK_CUDA(cudaMallocHost(&h_all_hits[si], MAX_HITS * sizeof(Hit)));
    }

    // 5. DISPATCH LOOP: Pure Asynchronous Work [Lecture 4]
    const uint32_t qBytesAligned = (q_view.nBytes + 3) & ~0x03;
    const uint32_t tileBytesMax = (TILE_CHARS + 3u) >> 2;
    size_t shmemBytes = (size_t)qBytesAligned + (size_t)tileBytesMax;

    for (int si = 1; si <= DB_SIZE; si++) {
        int sIdx = (si - 1) % nStreams;
        int arrIdx = si - 1;

        std::cout << "Sending Sequence " << si << " in stream" << sIdx << std::endl;

        SeqView db_view { d_db_buffers[sIdx], db_nBytes_array[arrIdx], db_nChars_array[arrIdx] };
        KernelParamsView params { q_view, db_view, lView, d_hits_buffers[sIdx], d_hitCount_buffers[sIdx], MAX_HITS };

        // Ensure stream is ready from previous work
        cudaStreamSynchronize(streams[sIdx]);
        cudaMemsetAsync(d_hitCount_buffers[sIdx], 0, sizeof(uint32_t), streams[sIdx]);

        // H2D -> KERNEL -> D2H Pipeline
        cudaMemcpyAsync(d_db_buffers[sIdx], h_db_pinned_array[arrIdx], db_nBytes_array[arrIdx],
                        cudaMemcpyHostToDevice, streams[sIdx]);

        blast<<<blocksPerGrid, threadsPerBlock, shmemBytes, streams[sIdx]>>>(params);

        cudaMemcpyAsync(&h_hitCounts[arrIdx], d_hitCount_buffers[sIdx], sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, streams[sIdx]);
        cudaMemcpyAsync(h_all_hits[arrIdx], d_hits_buffers[sIdx], MAX_HITS * sizeof(Hit),
                        cudaMemcpyDeviceToHost, streams[sIdx]);
    }

    // 6. Final Synchronization and Processing
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "\n--- Final Results ---\n";
    for (int si = 1; si <= DB_SIZE; si++) {
        save_results(h_hitCounts[si-1], MAX_HITS, h_all_hits[si-1], si);
        std::cout << "Sequence " << si << ": Found " << h_hitCounts[si-1] << " hits. Results saved to results/ folder.\n";
    }



    // 7. Cleanup (Code omitted for brevity, ensure all FreeHost and Free calls are included)
    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_db_buffers[i]);
        cudaFree(d_hits_buffers[i]);
        cudaFree(d_hitCount_buffers[i]);
    }

    for (int si = 0; si < DB_SIZE; si++) {
        cudaFreeHost(h_db_pinned_array[si]); // Freeing pinned DB memory [cite: 212]
        cudaFreeHost(h_all_hits[si]);        // Freeing pinned hits memory
    }

    cudaFreeHost(h_q_pinned);  // Freeing pinned query memory [cite: 212]
    cudaFreeHost(h_hitCounts);
    cudaFree(d_query);
    cudaFree(d_off);
    cudaFree(d_pos);

    return 0;
}
