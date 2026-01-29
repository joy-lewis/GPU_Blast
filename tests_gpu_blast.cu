//
// Created by Joy de Carvalho Lucas on 25.01.26.
//

#include <iostream>
#include <cstring>
#include <bitset>
#include <cuda_runtime.h>
#include "gpu_blast.h"


int main() {
    const char* dna = "ACGTACGT";
    size_t dna_length = strlen(dna);
    size_t encoder_out_size = (dna_length + 3)/4; // 4 characters fit in each byte
    uint8_t* encoder_out = (uint8_t*) malloc(sizeof(uint8_t)*encoder_out_size); // vector which will hold the resulting encoding

    // Encode
    encoder(dna, dna_length, encoder_out);

    // Decode
    //size_t out_size = length; // 4 characters fit in each byte
    //std::vector<char> output(out_size); // vector which will hold the resulting decoding

    int encoder_out_length = 8; // in the real algorithm this has to be given by the GPU because the alignments (to be decoded) can havy any length
    char* decoder_out = (char*) malloc(sizeof(char)*encoder_out_length);
    decoder(encoder_out, encoder_out_length, decoder_out);

    std::cout << "Original: " << dna << "\n";

    std::cout << "Encoded (bits): ";
    for (size_t i = 0; i < encoder_out_size; i++) {
        std::bitset<8> bits(encoder_out[i]);
        std::cout << bits << " ";
    }
    std::cout << "\n";

    std::cout << "Decoded:  ";
    for (size_t i=0; i<encoder_out_length; i++) {
        std::cout << decoder_out[i];
    }
    std::cout << "\n";

    return 0;
}

//run: nvcc gpu_blast.cu tests_gpu_blast.cpp -o test