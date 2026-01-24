#include <vector>
#include <cstdint>
#include <utility>
#include <iostream>

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
std::vector<uint8_t> decoder(const uint8_t* input, size_t length) { //length==original length of the character based DNA sequence
    size_t out_size = length; // 4 characters fit in each byte
    std::vector<char> output(out_size); // vector which will hold the resulting decoding

    int in_index = 0;
    int out_index = 0;

    //todo: implement the rest of the decoding logic below

    while (in_index < length) { // as long as there are characters left we keep on encoding
        uint8_t new_byte = 0;

        // Encode the next 4 characters
        for (int i = 0; i < 4; i++) {
            new_byte <<= 2;
            if (in_index < length) {
                uint8_t encoding = decode_bits[input[in_index]];
                new_byte |= encoding; // insert the two new bits into the existing byte
                in_index++;
            }
        }
        output[out_index++] = new_byte;
    }
}


int main() {
    !!VERY IMPORTANT: use the length of the original sequence, to know when to stop processing the
    encoded binary sequence, becasue the last few bits are all zero padded.
    Same goes for when we decode the final alignment back on CPU after GPU sent it over

}