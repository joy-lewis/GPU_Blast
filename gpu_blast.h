//
// Created by Joy de Carvalho Lucas on 25.01.26.
//

#pragma once
#include <vector>
#include <cstdint>

uint8_t encode_char(char c);
char decode_bits(uint8_t b);

void encoder(const char* input, size_t length, uint8_t* output);
void decoder(const uint8_t* input, size_t length, char* output);