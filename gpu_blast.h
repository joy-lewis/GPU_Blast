//
// Created by Joy de Carvalho Lucas on 25.01.26.
//

#pragma once
#include <vector>
#include <cstdint>

uint8_t encode_char(char c);
char decode_bits(uint8_t b);

void encoder(const std::vector<char>& input, size_t length, uint8_t* output);
void decoder(const uint8_t* input, size_t length, char* output);

struct LookupTable {
    std::vector<uint32_t> offsets; // size = M+1
    std::vector<uint32_t> positions; // size = L
};

uint32_t base_at_msb4(const uint8_t* encoded_dna, uint32_t i);
uint32_t kmer_at_msb_bytes(const uint8_t* encoded_dna, uint32_t pos, uint32_t k);

LookupTable build_lookup_table_from_encoded(const uint8_t* encoded_dna,
                                            uint32_t N, uint32_t k);