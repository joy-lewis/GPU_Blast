//
// Created by Joy de Carvalho Lucas on 25.01.26.
//

#include <iostream>
#include <cstring>
#include <bitset>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include "gpu_blast.h"


// Optional: decode a 2-bit base to char for printing (must match your encoder mapping!)
static inline char base_code_to_char(uint32_t code)
{
    static const char map[4] = {'A','C','G','T'};
    return map[code & 3u];
}

static void print_kmer_from_encoded(const uint8_t* encoded_dna, uint32_t pos, uint32_t k)
{
    for (uint32_t j = 0; j < k; ++j) {
        std::cout << base_code_to_char(base_at_msb4(encoded_dna, pos + j));
    }
}

int test_main() {
    std::vector<char> dna = {'A', 'C', 'G', 'T', 'A', 'C', 'G', 'T'};
    const int dna_length = dna.size();

    const size_t encoder_out_size = (dna_length + 3) / 4; // 4 bases per byte
    uint8_t* encoder_out = (uint8_t*) std::malloc(sizeof(uint8_t) * encoder_out_size);

    // Encode using your existing function
    encoder(dna, dna_length, encoder_out);
    std::cout << "Bytes used: " << encoder_out_size << "\n";

    // Decode using your existing function (your decoder expects a char length)
    const int decoder_out_length = (int)dna_length;
    char* decoder_out = (char*) std::malloc(sizeof(char) * decoder_out_length);
    decoder(encoder_out, decoder_out_length, decoder_out);

    std::cout << "Original: ";
    std::cout.write(dna.data(), dna.size());
    std::cout << std::endl; // Optional newline

    std::cout << "Encoded (bits): ";
    for (size_t i = 0; i < encoder_out_size; i++) {
        std::bitset<8> bits(encoder_out[i]);
        std::cout << bits << " ";
    }
    std::cout << "\n";

    std::cout << "Decoded:  ";
    for (int i = 0; i < decoder_out_length; i++) {
        std::cout << decoder_out[i];
    }
    std::cout << "\n\n";

    // -----------------------------
    // NEW: Build and test lookup table
    // -----------------------------
    const uint32_t k = 2;
    const uint32_t N = (uint32_t)dna_length;
    const uint32_t L = N - k + 1u;
    const uint32_t M = 1u << (2u * k);

    LookupTableView lut = build_lookup_table_from_encoded(encoder_out, N, k);

    std::cout << "Lookup table test (k=" << k << ")\n";
    std::cout << "N=" << N << ", L=" << L << ", M=" << M << "\n";

    // Print only keys that occur (for readability)
    for (uint32_t key = 0; key < M; ++key) {
        const uint32_t start = lut.offsets[key];
        const uint32_t end   = lut.offsets[key + 1u];
        if (start == end) continue;

        std::cout << "key=" << key << " occurs " << (end - start) << " times at positions: ";
        for (uint32_t t = start; t < end; ++t) {
            uint32_t pos = lut.positions[t];
            std::cout << pos << " (";
            print_kmer_from_encoded(encoder_out, pos, k);
            std::cout << ") ";
        }
        std::cout << "\n";
    }

    // Expected for "ACGTACGT" with k=2:
    // AC at positions [0,4]
    // CG at positions [1,5]
    // GT at positions [2,6]
    // TA at positions [3]
    const uint32_t key_AC = 1;   // 00 01 -> 0b0001
    const uint32_t key_CG = 6;   // 01 10 -> 0b0110
    const uint32_t key_GT = 11;  // 10 11 -> 0b1011
    const uint32_t key_TA = 12;  // 11 00 -> 0b1100

    auto slice_equals = [&](uint32_t key, const std::vector<uint32_t>& expected) {
        uint32_t start = lut.offsets[key];
        uint32_t end   = lut.offsets[key + 1u];
        assert((end - start) == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            assert(lut.positions[start + (uint32_t)i] == expected[i]);
        }
    };

    slice_equals(key_AC, {0, 4});
    slice_equals(key_CG, {1, 5});
    slice_equals(key_GT, {2, 6});
    slice_equals(key_TA, {3});

    // Also sanity: total positions length must be L
    assert(lut.positions.size() == L);
    // And last offset must equal L
    assert(lut.offsets.back() == L);

    std::cout << "\nLookup table assertions passed ✅\n";

    std::free(encoder_out);
    std::free(decoder_out);
    return 0;
}

// compile: nvcc gpu_blast.cu tests_gpu_blast.cu -o test