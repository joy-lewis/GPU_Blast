#include <vector>
#include <cstdint>
#include <utility>
#include <iostream>
#include <cassert>
#include <cstring>
#include <bitset>
#include <cmath>
#include <stdexcept>
#include "gpu_blast.h"
#include <cuda_runtime.h>


/////////////////////////////////////
/////////// Define Hyperparameters //
/////////////////////////////////////
#define DB_SIZE 10 // number of sequences in Database
#define K 8 // defines k-mer length

// conditions: K>0

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

// Converting DNA sequence to bit array
// Layout for 8-bit array: base 0 is in bits 7–6 of byte 0, base 1 in bits 5–4, base 2 in bits 3–2, base 3 in bits 1–0
void encoder(const char* input, size_t length, uint8_t* output) { //length=number of characters
    int in_index = 0;
    int out_index = 0;

    while (in_index < length) { // as long as there are characters left we keep on encoding
        uint8_t new_byte = 0;

        // Encode the next 4 characters in a MSB-first manner
        for (int i = 0; i < 4; i++) {
            new_byte <<= 2; // shift the existing encoding two positions to the left to make space for the next bit pair
            if (in_index < length) {
                uint8_t encoding = encode_char(input[in_index]);
                new_byte |= encoding; // insert the two new bits into the left side of the existing byte
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
    const uint8_t* encoded_dna,  // packed 2-bit bases (4 per byte, MSB-first)
    uint32_t N,                  // total number of bases in the query
    uint32_t k                   // k-mer length in bases
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


////////////////////////////
////////// DNA Alignment //
///////////////////////////
void launch_kernels() {
    // Launching the kernels, i.e. blocks on the GPU device
}

/*
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
*/