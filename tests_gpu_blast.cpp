//
// Created by Joy de Carvalho Lucas on 25.01.26.
//

#include <iostream>
#include <cstring>
#include "gpu_blast.h"

int main() {
    const char* dna = "ACGTACGT";

    auto encoded = encoder(dna, strlen(dna));
    auto decoded = decoder(encoded, strlen(dna));

    std::cout << "Original: " << dna << "\n";
    std::cout << "Decoded:  ";

    for (char c : decoded) {
        std::cout << c;
    }
    std::cout << "\n";

    return 0;
}

//run: nvcc gpu_blast.cu tests_gpu_blast.cpp -o test