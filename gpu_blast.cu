#include <vector>
#include <cstdint>
#include <utility>
#include <iostream>
#include "gpu_blast.h"
#include <cassert>

#define CHUNK_SIZE 2
#define N 2              // Grid size X
#define M 2             // Grid size Y

#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
    std::cerr << "CUDA error at " << __LINE__ << std::endl;     \
    exit(EXIT_FAILURE);                                         \
    }

// Data Compression
uint8_t encode_char(char c) {
    // Compressing the ASCII characters (4 unique DNA bases) to a 2-bit encoding
    switch (c) {
        case 'A': return 0b00;
        case 'C': return 0b01;
        case 'G': return 0b10;
        case 'T': return 0b11;
        default: assert(false) // should not happen
        }
}

char decode_bits(uint8_t b) {
    // Decompressing the 2-bit encoding back to an ASCII symbol
    switch (b) {
        case 0b00: return 'A';
        case 0b01: return 'C';
        case 0b10: return 'G';
        case 0b11: return 'T';
        default: assert(false) // should not happen
        }
}

// Converting DNA sequence to bit string
std::vector<uint8_t> encoder(const char* input, size_t length) { //length=number of characters
    size_t out_size = (length + 3)/4; // 4 characters fit in each byte
    std::vector<uint8_t> output(out_size); // vector which will hold the resulting encoding

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
    return output;
}

// Converting but string to DNA sequence
std::vector<uint8_t> decoder(const uint8_t* input, size_t length) { //length==number of characters that are encoded in the input bits (relevant because last byte might encode less then 4 characters)
    size_t out_size = length; // 4 characters fit in each byte
    std::vector<char> output(out_size); // vector which will hold the resulting decoding

    int in_index = 0;
    int out_index = 0;

    while (out_index < length) { // as long as there are characters left we keep on decoding
        uint8_t byte = input[in_index];

        // Encode the next 4 characters
        for (int shift = 6; shift >= 0; shift -=2) {
            // we right shift by 6, 4, 2 and then 0 to have each bit pair at the rightmost edge of the byte
            byte >>= shift;
            byte &= 0b11; // to isolate the two rightmost bits we care about
            if (out_index < length) {
                uint8_t decoding = decode_bits[byte];
                output[out_index++] = decoding;
            }
        }
        in_index++;
    }
}


int main() {
    !!VERY IMPORTANT: use the length of the original sequence, to know when to stop processing the
    encoded binary sequence, becasue the last few bits are all zero padded.
    Same goes for when we decode the final alignment. We need to pass along the length of valid bit pairs (i.e. characters)
    so we dont decode all bits in the last byte of the uint8_t array.

}