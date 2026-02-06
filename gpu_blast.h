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

// Structs that hold the relevant device pointers for the query and database sequences
struct LookupTable { // pure lookup table with host vectors
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> positions;
};

struct LookupTableView { // lookup table with address pointers of device
    const uint32_t* offsets;    // size = M+1
    const uint32_t* positions;  // size = L
    uint32_t nOffsets;          // M+1
    uint32_t nPositions;        // L
};

struct SeqView {
    const uint8_t* seq;   // device pointer to packed 2-bit DNA
    uint32_t nBytes;      // bytes allocated for seq
    uint32_t nChars;      // number of DNA bases
};

struct KernelParamsView {
    SeqView query;
    LookupTableView lView;
    uint32_t K;
};

uint32_t base_at_msb4(const uint8_t* encoded_dna, uint32_t i);
uint32_t kmer_at_msb_bytes(const uint8_t* encoded_dna, uint32_t pos, uint32_t k);

LookupTable build_lookup_table_from_encoded(const uint8_t* encoded_dna,
                                           uint32_t N, uint32_t k);

LookupTableView lookup_table_to_device(const LookupTable& t,
                                       uint32_t** d_offsets_out,
                                       uint32_t** d_positions_out);