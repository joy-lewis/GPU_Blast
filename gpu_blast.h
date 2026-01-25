//
// Created by Joy de Carvalho Lucas on 25.01.26.
//

#pragma once
#include <vector>
#include <cstdint>

uint8_t encode_char(char c);
char decode_bits(uint8_t b);

std::vector<uint8_t> encoder(const char* input, size_t length);
std::vector<char> decoder(const std::vector<uint8_t>& input, size_t length);