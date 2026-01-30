#include <vector>
#include <cstdint>
#include <utility>
#include <iostream>
#include <cassert>
#include <cstring>
#include <bitset>
#include <cmath>
#include "gpu_blast.h"
#include <cuda_runtime.h>


/////////////////////////////////////
/////////// Define Hyperparameters //
/////////////////////////////////////
#define DB_SIZE 10 // number of sequences in Database
#define K 8 // defines k-mer length

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

// Converting DNA sequence to bit string
void encoder(const char* input, size_t length, uint8_t* output) { //length=number of characters
    int in_index = 0;
    int out_index = 0;

    while (in_index < length) { // as long as there are characters left we keep on encoding
        uint8_t new_byte = 0;

        // Encode the next 4 characters
        for (int i = 0; i < 4; i++) {
            new_byte <<= 2;
            if (in_index < length) {
                uint8_t encoding = encode_char(input[in_index]);
                new_byte |= encoding; // insert the two new bits into the existing byte
                in_index++;
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


/////////////////////////////
/////////// Data Transfer ///
/////////////////////////////
void transfer_to_device() {
    // Moving a full sequence from host to device global memory
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

// all possible k-mers are within [0, 2^(2*K)], because every character is 2 bits long (2*K) and there are 2 possible bits
// simplifies to [0, 4^K]

void build_lookup_table() {
    // Keys: all unique k-mers; Values
    int num_entries = pow(4, K);

    // we process the query sequence left-to-right to record all k-mer positons while keeping those positions sorted

    // count occurances
    const uint32_t M = 1u << (2*k);
    const uint32_t L = N - k + 1;

    std::vector<uint32_t> counts(M, 0);

    for (uint32_t i = 0; i < L; ++i) {
        uint32_t key = kmer_at(seq2bit, i, k);  // your 2-bit extraction
        counts[key]++;
    }

    // prefix sum counts -> offsets
    std::vector<uint32_t> offsets(M + 1, 0);
    for (uint32_t key = 0; key < M; ++key) {
        offsets[key + 1] = offsets[key] + counts[key];
    }

    // fill positons (sorted)
    std::vector<uint32_t> cursor = offsets;      // copy offsets[0..M-1]
    std::vector<uint32_t> positions(L);

    for (uint32_t i = 0; i < L; ++i) {
        uint32_t key = kmer_at(seq2bit, i, k);
        uint32_t out = cursor[key]++;            // write then increment
        positions[out] = i;                      // i increases => sorted per key
    }

    // for each key the indice list is: positions[offsets[key] .. offsets[key+1])

    /*
     * Operations on the kernel:
    * start = d_offsets[key];
end   = d_offsets[key+1];
for (t = start; t < end; ++t) {
    qpos = d_positions[t];
    ...
}
     *
     */
}



////////////////////////////
////////// DNA Alignment //
///////////////////////////
void launch_kernels() {
    // Launching the kernels, i.e. blocks on the GPU device
}



int main() {
    const char* query = "";
    //todo: move query and query hash table from host to device global memory


    // Process DB sequences one by one
    for (int si=0; si<DB_SIZE; si++) {
        //todo: send full DB sequence si over from Host to Device global memory

        //todo: free memory of sequence si on device

    }
    //todo: free memory of query and and lookup table on device


    return 0;
}
