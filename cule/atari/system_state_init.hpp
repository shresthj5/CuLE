#pragma once

#include <cule/config.hpp>

#include <cule/atari/types/types.hpp>

namespace cule
{
namespace atari
{
namespace detail
{

template<typename State_t>
static CULE_ANNOTATION void initialize_power_on_riot_ram(State_t& s)
{
    constexpr uint8_t stella_power_on_riot_ram[128] = {
        184, 150, 192,  73, 203,  85, 200, 127, 251, 180,  69, 102,  43,  52,  16, 202,
        185, 206, 255, 143,  79,  68, 161, 138,   4, 197, 205, 252, 117, 231,  10, 173,
         45, 155,  96, 158, 170,  23,  30,  84, 249, 132,  81,  62,  52, 151, 124,  94,
         80, 182, 134, 133, 190, 119,  69, 139, 109,   4, 181, 238,  66,  57, 193,  75,
        156,  51,  82,  71,  73, 179, 238, 107, 112, 172,  36, 107, 212, 254, 254, 214,
         23, 220,  56, 154,  73, 142,  33, 172, 214,  28,  13,  92, 218, 173,  11, 119,
        163, 108,  40,  56,  81, 197, 137,  92,  90,  22, 152,  85, 149,  91, 175,  74,
        175, 153,  13,  75, 236,  10, 186,  94, 167, 189, 230, 247, 171,  20, 111, 192,
    };
    uint8_t* ram_bytes = reinterpret_cast<uint8_t*>(s.ram);
    for(size_t i = 0; i < 128; ++i)
    {
        ram_bytes[i] = stella_power_on_riot_ram[i];
    }
}

template<typename State_t>
static CULE_ANNOTATION void initialize_f8sc_reset_ram(State_t& s)
{
    constexpr uint8_t stella_f8sc_reset_ram[128] = {
        139, 236, 141,  97, 233, 177, 178,  81,  78,   7,  14, 191,  19,  37,  79, 154,
        112, 210,   5, 152, 196,  95,  65, 171,  36, 230,  59, 123,  25, 234, 167, 215,
          8,   7, 134,  49, 201, 187, 231, 241, 236,  36, 163, 237, 145,  84, 242, 248,
        113, 122, 111, 217, 153, 104, 218, 221, 229, 159,  48,  24, 213, 104, 249,  52,
        190,   0,  61, 180,  18,   6,  27, 139,  49, 134, 255, 167, 195,  51, 131, 190,
          5, 183,  22, 155,  69, 105, 161,  47,  41, 201, 155,  52, 137, 146, 220, 245,
         79,  98, 217,  74, 170, 234,  91,  38,  78,  70, 158, 115,  21,  27, 237, 169,
        127, 123, 193, 231,  44,  14,  84, 109,  74,  92,  70,  72, 174, 238, 137,  35,
    };
    uint8_t* ram_bytes = reinterpret_cast<uint8_t*>(s.ram) + 128;
    for(size_t i = 0; i < 128; ++i)
    {
        ram_bytes[i] = stella_f8sc_reset_ram[i];
    }
}

} // namespace detail
} // namespace atari
} // namespace cule
