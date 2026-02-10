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


/////////////////////////////////////
/////////// Define Hyperparameters //
/////////////////////////////////////
#define DB_SIZE 10 // number of sequences in Database
#define K 12 // defines k-mer length // conditions: K>0

#define NUM_THREADS_PER_BLOCK 256
#define NUM_BLOCKS 4

// Tile length in "characters" (DNA bases), cant be larger then (total-shared-memory-query-memory*NUM_BLOCKS) / NUM_BLOCKS
// Example: 16384 chars => 4096 bytes.
#define TILE_CHARS 16384u //todo:optimizable

// Scoring function for matches during extension. +1 for match, -1 for non-match
#define MATCH_SCORE 1
#define MISMATCH_PENALTY -1

// Minimal score to report
#define MIN_REPORT_SCORE K+1 // this way we only record matches where the alignment is larger then the seed k-mer

// X-drop termination condition to keep the extension finite
#define X_DROP 4


////////////////////////////
/////////// Sanity checks //
////////////////////////////
#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
    std::cerr << "CUDA error at " << __LINE__ << std::endl;     \
    exit(EXIT_FAILURE);                                         \
    }


//TODO:VERY IMPORTANT!!
//use the length of the original sequence, to know when to stop processing the
//encoded binary sequence, because the last few bits are all zero padded.
//Same goes for when we decode the final alignment. We need to pass along the length of valid bit pairs (i.e. characters)
//so we dont decode all bits in the last byte of the uint8_t array.

/////////////////////////////////////////////
/////// Extract DNA sequence from fasta file
/////////////////////////////////////////////
std::vector<char> read_fasta(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open: " + path);

    std::vector<char> seq;
    std::string line;

    int nBytes = 0;
    int nChars = 0;

    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '>') continue; // skip header
        for (unsigned char ch : line) {
            if (std::isspace(ch)) continue;
            ch = (unsigned char)std::toupper(ch);
            if (ch=='A' || ch=='C' || ch=='G' || ch=='T')
                seq.push_back((char)ch);
        }
    }
    return seq;
}


/////////////////////////////////
///////////// Data Compression //
/////////////////////////////////
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

// Converting DNA sequence to bit array
// Layout for 8-bit array: base 0 is in bits 7–6 of byte 0, base 1 in bits 5–4, base 2 in bits 3–2, base 3 in bits 1–0
void encoder(const std::vector<char>& input, size_t length, uint8_t* output) {
    size_t in_index = 0;
    size_t out_index = 0;

    while (in_index < length) {
        uint8_t new_byte = 0;

        // pack 4 bases into 1 byte, MSB-first
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

// Converting but string to DNA sequence
void decoder(const uint8_t* input, size_t length, char* output) { //length==number of characters that are encoded in the input bits (relevant because last byte might encode less then 4 characters)
    int in_index = 0;
    int out_index = 0;

    while (out_index < length) { // as long as there are characters left we keep on decoding
        uint8_t byte = input[in_index];

        // Encode the next 4 characters
        for (int shift = 6; shift >= 0; shift -=2) {
            // we right shift by 6, 4, 2 and then 0 to have each bit pair at the rightmost edge of the byte
            // then we do AND with 0b11 to isolate the two rightmost bits we care about
            uint8_t sbyte = byte>>shift;
            sbyte = sbyte&0b11;

            if (out_index < length) {
                uint8_t decoding = decode_bits(sbyte);
                output[out_index++] = decoding;
            }
        }
        in_index++;
    }
}


///////////////////////////////////
////////// Lookup Table Creation //
///////////////////////////////////
///
//logic:
// “indices 100..143 belong to k-mer 0x11111”
// “indices 144..146 belong to k-mer 0x22222”

// "offsets[k-mer] = start index in positions[]"
// "offsets[k-mer+1] = end index in positions[]"

// "positions array are sorted query sequence positions"

// all possible k-mers are within [0, 2^(2*K)], because every character is 2 bits long (2*K) and there are 2 possible bits
// simplifies to [0, 4^K]

uint32_t base_at_msb4(const uint8_t* encoded_dna, uint32_t i)
{
    //Purpose of this function:
    // 1.	Which byte contains base i
    // 2.	Where inside that byte the 2 bits for that base live
    // 3.	Shift + mask to extract those 2 bits
    const uint32_t byte_idx = i >> 2;           // i / 4
    const uint32_t in_byte  = i & 3u;           // 0..3
    const uint32_t shift    = 6u - 2u * in_byte; // 6,4,2,0 (MSB -> LSB)
    return (encoded_dna[byte_idx] >> shift) & 0x3u;
}

// Returns a 2k-bit key where the leftmost base is the most significant 2 bits.
// Example: ACGT => 00 01 10 11 => 0b00011011
uint32_t kmer_at_msb_bytes(const uint8_t* encoded_dna,
                                         uint32_t pos, uint32_t k)
{
    uint32_t key = 0;
    for (uint32_t j = 0; j < k; ++j) {
        key = (key << 2) | base_at_msb4(encoded_dna, pos + j);
    }
    return key;
}


LookupTable build_lookup_table_from_encoded(
    const uint8_t* encoded_dna,  // packed 2-bit characters (4 per byte, MSB-first)
    uint32_t N,                  // total number of characters in the query
    uint32_t k                   // k-mer length in characters
){

    if (2u * k >= 32u) {
        // because we store keys in uint32_t and compute M = 1<<(2k)
        throw std::invalid_argument("k too large for 32-bit key / table size");
    }

    const uint32_t L = N - k + 1u;      // number of k-mers in query
    const uint32_t M = 1u << (2u * k);  // number of possible k-mers (4^k)

    // 1) count occurrences
    std::vector<uint32_t> counts(M, 0);
    for (uint32_t i = 0; i < L; ++i) {
        const uint32_t key = kmer_at_msb_bytes(encoded_dna, i, k);
        ++counts[key];
    }

    // 2) prefix sum -> offsets (exclusive scan)
    std::vector<uint32_t> offsets(M + 1u, 0);
    for (uint32_t key = 0; key < M; ++key) {
        offsets[key + 1u] = offsets[key] + counts[key];
    }

    // 3) fill positions (sorted per key because i increases)
    std::vector<uint32_t> cursor(offsets.begin(), offsets.begin() + M); // size M
    std::vector<uint32_t> positions(L);

    for (uint32_t i = 0; i < L; ++i) {
        const uint32_t key = kmer_at_msb_bytes(encoded_dna, i, k);
        const uint32_t out = cursor[key]++;
        positions[out] = i;
    }

    return LookupTable{std::move(offsets), std::move(positions)};
}


// Handles memory allocation on device and copies the lookup table contents over to the device
LookupTableView lookup_table_to_device(const LookupTable& t, uint32_t** d_offsets_out, uint32_t** d_positions_out) {
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

    // returns device pointers
    *d_offsets_out = d_offsets;
    *d_positions_out = d_positions;

    return LookupTableView{
        d_offsets,
        d_positions,
        (uint32_t)t.offsets.size(),
        (uint32_t)t.positions.size()
    };
}


////////////////////////////
////////// DNA Alignment //
///////////////////////////


// leftmost 2-bit character (DNA base) extraction
// Similar logic to what we used for the lookup table creation
__device__ __forceinline__ uint32_t base_at_msb4_dev(const uint8_t* encoded_dna, uint32_t iChar) {
    const uint32_t byte_idx = iChar >> 2;            // /4
    const uint32_t in_byte  = iChar & 3u;            // 0..3
    const uint32_t shift    = 6u - 2u * in_byte;     // 6,4,2,0
    return (encoded_dna[byte_idx] >> shift) & 0x3u;
}

// k-mer extraction
__device__ __forceinline__ uint32_t kmer_at_msb_bytes_dev(const uint8_t* encoded_dna,
                                                          uint32_t posChar) {
    uint32_t key = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 32; ++j) {  // unroll upper bound; break at K
        if (j >= K) break;
        key = (key << 2) | base_at_msb4_dev(encoded_dna, posChar + j);
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
        return base_at_msb4_dev(acc.tile_shared, local);
    } else {
        return base_at_msb4_dev(acc.db_global, dbCharIdx);
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
        const uint32_t qb  = base_at_msb4_dev(q_shared, (uint32_t)iQ);
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
        const uint32_t qb  = base_at_msb4_dev(q_shared, (uint32_t)iQ);
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

    // Shared memory layout:
    // [ query_bytes ][ tile_bytes ]
    extern __shared__ uint8_t shared_mem[];
    uint8_t* q_sh = shared_mem;

    const uint32_t qBytes = params.query.nBytes;
    const uint32_t tileBytesMax = (TILE_CHARS + 3u) >> 2; // /4 rounded up
    uint8_t* tile_sh = q_sh + qBytes;

    // 1) Load query into shared (all blocks do this in parallel)
    for (uint32_t i = threadIdx.x; i < qBytes; i += blockDim.x) {
        q_sh[i] = params.query.seq[i];
    }
    __syncthreads(); // wait till the full query is loaded into shared memory

    // 2) Every block has its pre-determined set of tiles: tileId = blockIdx.x
    //  which gets updated in each iteration with tileId = tileId + gridDim.x
    const uint32_t tilesTotal = (dbLen + TILE_CHARS - 1u) / TILE_CHARS;

    for (uint32_t tileId = (uint32_t) blockIdx.x; tileId < tilesTotal; tileId += (uint32_t) gridDim.x) {

        const uint32_t tileStartChar = tileId * TILE_CHARS;
        const uint32_t tileChars = min(TILE_CHARS, dbLen - tileStartChar);
        const uint32_t tileBytes = (tileChars + 3u) >> 2;

        // 2a) Load this tile’s encoded bytes into shared memory
        const uint32_t tileStartByte = tileStartChar >> 2; // /4
        for (uint32_t i = threadIdx.x; i < tileBytes; i += blockDim.x) {
            tile_sh[i] = params.database.seq[tileStartByte + i];
        }
        __syncthreads(); // wait till tiles are fully loaded

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
            #pragma unroll
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
                    // Append to global output
                    uint32_t outIdx = atomicAdd(params.hitCount, 1u);
                    if (outIdx < params.maxHits) { // out of memory check
                        Hit h;
                        h.db_pos     = dbPos;
                        h.q_pos      = qPos;
                        h.bestScore  = bestScore;
                        h.leftExt    = leftExt;
                        h.rightExt   = rightExt;
                        params.hits[outIdx] = h;
                    }
                }
            }
        }

        __syncthreads(); // before next tile load
    }
}


void save_results(const uint32_t* d_hitCount, uint32_t maxHits, const Hit* d_hits, int si) {
    // 4) Copy back hitCount
    uint32_t h_hitCount = 0;
    CHECK_CUDA(cudaMemcpy(&h_hitCount, d_hitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy only as many hits as fit in the allocated buffer
    uint32_t nToCopy = std::min(h_hitCount, maxHits);

    // 5) Copy back hits
    std::vector<Hit> h_hits(nToCopy);
    if (nToCopy > 0) {
        CHECK_CUDA(cudaMemcpy(h_hits.data(), d_hits, nToCopy * sizeof(Hit), cudaMemcpyDeviceToHost));

        // sort alignments by length
        std::sort(h_hits.begin(), h_hits.end(),
                  [](const Hit& a, const Hit& b) {
                      if (a.bestScore != b.bestScore) return a.bestScore > b.bestScore;
                      if (a.db_pos != b.db_pos) return a.db_pos < b.db_pos;
                      return a.q_pos < b.q_pos;
                  });
    }

    // save results to txt file

    namespace fs = std::filesystem;
    fs::create_directories("results"); // no-op if it already exists

    std::string outName = (fs::path("results") /
                               ("blast_results_sequence" + std::to_string(si) + ".txt")).string();
    std::ofstream out(outName, std::ios::out);

    if (!out) {
        std::cerr << "Failed to open output file: " << outName << "\n";
    } else {
        out << "DB sequence index: " << si << "\n";
        out << "Device hitCount (may exceed cap): " << h_hitCount << "\n";
        out << "Hits written (clamped to cap): " << nToCopy << "\n";
        out << "Columns: db_pos q_pos bestScore leftExt rightExt\n";

        for (uint32_t i = 0; i < nToCopy; ++i) {
            const Hit& h = h_hits[i];
            out << h.db_pos << " "
                << h.q_pos << " "
                << h.bestScore << " "
                << h.leftExt << " "
                << h.rightExt << "\n";
        }
    }
}

////////////////////////////
////////// MAIN FUNCTION //
///////////////////////////
int blast_main() {
    // Load DNA sequences from database
    char query_name[32];
    char db_name[32];

    // Read in the seqeunce from file
    snprintf(query_name, sizeof(query_name), "ncbi_data/query.fasta");
    std::vector<char> query_seq = read_fasta(query_name);
    const int query_nChars = query_seq.size();

    // Encode query sequence to 2-bit encoding
    const int query_nBytes = (query_nChars + 3) / 4; // 4 bases per byte
    uint8_t* query_encoder_out = (uint8_t*) std::malloc(sizeof(uint8_t) * query_nBytes);
    encoder(query_seq, query_nChars, query_encoder_out);

    // Allocate memory on device for query sequence and lookup table
    uint8_t* d_query;
    CHECK_CUDA(cudaMalloc(&d_query, query_nBytes));
    CHECK_CUDA(cudaMemcpy(d_query, query_encoder_out, query_nBytes, cudaMemcpyHostToDevice));

    SeqView q {
        d_query,
        (uint32_t) query_nBytes,
        (uint32_t) query_nChars
    };

    // Create lookup table for query sequence
    LookupTable lTable = build_lookup_table_from_encoded(query_encoder_out, query_nChars, K);
    // Send lookup table to device
    uint32_t *d_offsets, *d_positions;
    LookupTableView lView = lookup_table_to_device(lTable, &d_offsets, &d_positions);

    // Process DB sequences one by one
    for (int si=1; si<=DB_SIZE; si++) {
        // Read in the sequence from file
        snprintf(db_name, sizeof(db_name), "ncbi_data/sequence%d.fasta", si);
        std::vector<char> db_seq = read_fasta(db_name);
        const int db_nChars = db_seq.size();

        // Encode database sequence to 2-bit encoding
        const int db_nBytes = (db_nChars + 3) / 4; // 4 bases per byte
        uint8_t* db_encoder_out = (uint8_t*) std::malloc(sizeof(uint8_t) * db_nBytes);
        encoder(db_seq, db_nChars, db_encoder_out);

        // Allocate memory on device for database sequence
        uint8_t* d_db;
        CHECK_CUDA(cudaMalloc(&d_db, db_nBytes));
        CHECK_CUDA(cudaMemcpy(d_db, db_encoder_out, db_nBytes, cudaMemcpyHostToDevice));

        SeqView db {
            d_db,
            (uint32_t) db_nBytes,
            (uint32_t) db_nChars
        };

        // Allocate memory for outputs on device
        uint32_t MAX_HITS = 1u << 20; // roughly over 1 million, must be enough memory for counter
        Hit* d_hits;
        uint32_t* d_hitCount;
        CHECK_CUDA(cudaMalloc(&d_hits, MAX_HITS * sizeof(Hit)));
        CHECK_CUDA(cudaMalloc(&d_hitCount, sizeof(uint32_t)));
        CHECK_CUDA(cudaMemset(d_hitCount, 0, sizeof(uint32_t)));

        // Define parameters for the kernel function
        KernelParamsView params {
            q,
            db,
            lView,
            d_hits,
            d_hitCount,
        MAX_HITS
        };

        // Handing both the query sequence and the database sequence in a shared struct directly over to the kernel function AS VALUE
        dim3 threadsPerBlock(256, 1);
        dim3 blocksPerGrid(4, 1);
        std::cout << "Launching kernel with " << blocksPerGrid.x * blocksPerGrid.y << " blocks each with " << threadsPerBlock.x * threadsPerBlock.y << " threads\n";

        const uint32_t qBytes = q.nBytes;
        const uint32_t tileBytesMax = (TILE_CHARS + 3u) >> 2;   // bytes needed for TILE_CHARS
        size_t shmemBytes = (size_t)qBytes + (size_t)tileBytesMax;

        blast<<<blocksPerGrid, threadsPerBlock, shmemBytes>>>(params); // shmemBytes ensures we dont use more shared memory then we have available
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        save_results(d_hitCount, MAX_HITS, d_hits, si);

        // sanity check
        uint32_t h_hitCount = 0;
        CHECK_CUDA(cudaMemcpy(&h_hitCount, d_hitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::cout << "DB " << si << " hitCount=" << h_hitCount << " (cap " << MAX_HITS << ")\n";

        free(db_encoder_out);
        CHECK_CUDA(cudaFree(d_db));
        CHECK_CUDA(cudaFree(d_hits));
        CHECK_CUDA(cudaFree(d_hitCount));
    }
    free(query_encoder_out);
    CHECK_CUDA(cudaFree(d_query));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_positions));

    return 0;
}
