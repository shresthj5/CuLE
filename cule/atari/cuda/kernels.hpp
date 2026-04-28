#pragma once

#include <cule/config.hpp>

#include <cule/atari/state.hpp>
#include <cule/atari/rom.hpp>
#include <cule/atari/preprocess.hpp>
#include <cule/atari/frame_state_helpers.hpp>
#include <cule/atari/prng.hpp>

#include <cule/atari/cuda/frame_state.hpp>
#include <cule/atari/cuda/state.hpp>

#include <agency/agency.hpp>
#include <agency/cuda.hpp>

#ifndef CULE_ATARI_ENV_BLOCK_SIZE
#define CULE_ATARI_ENV_BLOCK_SIZE 32
#endif

namespace cule
{
namespace atari
{
namespace cuda
{

namespace detail
{

constexpr uint32_t MAX_RAM_BYTES_PER_ENV = 256;
constexpr uint32_t MAX_RAM_WORDS_PER_ENV = MAX_RAM_BYTES_PER_ENV / sizeof(uint32_t);
constexpr uint32_t ENV_RAM_LAYOUT_BLOCK_SIZE = CULE_ATARI_ENV_BLOCK_SIZE;

__device__ inline uint32_t ram_word_offset(const uint32_t env_index,
                                           const uint32_t word_index,
                                           const uint32_t ram_words_per_env)
{
    const uint32_t block_index = env_index / ENV_RAM_LAYOUT_BLOCK_SIZE;
    const uint32_t lane_index = env_index % ENV_RAM_LAYOUT_BLOCK_SIZE;
    return (ram_words_per_env * ENV_RAM_LAYOUT_BLOCK_SIZE * block_index) +
           (word_index * ENV_RAM_LAYOUT_BLOCK_SIZE) +
           lane_index;
}

template<size_t RAM_WORDS_PER_ENV>
__device__ inline uint32_t ram_word_offset(const uint32_t env_index,
                                           const uint32_t word_index)
{
    return ram_word_offset(env_index, word_index, RAM_WORDS_PER_ENV);
}

__device__ inline void load_env_ram_words(const uint8_t* ram_buffer,
                                          const uint32_t env_index,
                                          const uint32_t ram_words_per_env,
                                          uint32_t* ram_words)
{
    const uint32_t* ram_words_buffer = reinterpret_cast<const uint32_t*>(ram_buffer);
    for(uint32_t i = 0; i < ram_words_per_env; ++i)
    {
        ram_words[i] = ram_words_buffer[ram_word_offset(env_index, i, ram_words_per_env)];
    }
}

__device__ inline void store_env_ram_words(uint8_t* ram_buffer,
                                           const uint32_t env_index,
                                           const uint32_t ram_words_per_env,
                                           const uint32_t* ram_words)
{
    uint32_t* ram_words_buffer = reinterpret_cast<uint32_t*>(ram_buffer);
    for(uint32_t i = 0; i < ram_words_per_env; ++i)
    {
        ram_words_buffer[ram_word_offset(env_index, i, ram_words_per_env)] = ram_words[i];
    }
}

template<size_t RAM_WORDS_PER_ENV>
__device__ inline void load_env_ram_words(const uint8_t* ram_buffer,
                                          const uint32_t env_index,
                                          uint32_t* ram_words)
{
    static_assert(RAM_WORDS_PER_ENV <= MAX_RAM_WORDS_PER_ENV,
                  "RAM_WORDS_PER_ENV exceeds CuLE's maximum Atari RAM allocation");

    const uint32_t* ram_words_buffer = reinterpret_cast<const uint32_t*>(ram_buffer);
    #pragma unroll
    for(int32_t i = 0; i < int32_t(RAM_WORDS_PER_ENV); ++i)
    {
        ram_words[i] = ram_words_buffer[ram_word_offset<RAM_WORDS_PER_ENV>(env_index, i)];
    }
}

template<size_t RAM_WORDS_PER_ENV>
__device__ inline void store_env_ram_words(uint8_t* ram_buffer,
                                           const uint32_t env_index,
                                           const uint32_t* ram_words)
{
    static_assert(RAM_WORDS_PER_ENV <= MAX_RAM_WORDS_PER_ENV,
                  "RAM_WORDS_PER_ENV exceeds CuLE's maximum Atari RAM allocation");

    uint32_t* ram_words_buffer = reinterpret_cast<uint32_t*>(ram_buffer);
    #pragma unroll
    for(int32_t i = 0; i < int32_t(RAM_WORDS_PER_ENV); ++i)
    {
        ram_words_buffer[ram_word_offset<RAM_WORDS_PER_ENV>(env_index, i)] = ram_words[i];
    }
}

} // namespace detail

template<typename State_t, size_t NT>
__launch_bounds__(NT) __global__
void initialize_frame_states_kernel(const uint32_t noop_reset_steps,
                                    State_t* cached_states_buffer,
                                    frame_state* cached_frame_states_buffer,
                                    const uint32_t* pf_base,
                                    const uint8_t* p0_base,
                                    const uint8_t* p1_base,
                                    const uint8_t* m0_base,
                                    const uint8_t* m1_base,
                                    const uint8_t* bl_base)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= noop_reset_steps)
    {
        return;
    }

    frame_state& fs  = cached_frame_states_buffer[global_index];
    fs.CurrentPFMask = &playfield_accessor(0, 0) + (fs.CurrentPFMask - pf_base);
    fs.CurrentP0Mask = &player_mask_accessor(0, 0, 0, 0) + (fs.CurrentP0Mask - p0_base);
    fs.CurrentP1Mask = &player_mask_accessor(0, 0, 0, 0) + (fs.CurrentP1Mask - p1_base);
    fs.CurrentM0Mask = &missle_accessor(0, 0, 0, 0) + (fs.CurrentM0Mask - m0_base);
    fs.CurrentM1Mask = &missle_accessor(0, 0, 0, 0) + (fs.CurrentM1Mask - m1_base);
    fs.CurrentBLMask = &ball_accessor(0, 0, 0) + (fs.CurrentBLMask - bl_base);

    State_t& s = cached_states_buffer[global_index];
    s.CurrentPFMask = fs.CurrentPFMask;
    s.CurrentP0Mask = fs.CurrentP0Mask;
    s.CurrentP1Mask = fs.CurrentP1Mask;
    s.CurrentM0Mask = fs.CurrentM0Mask;
    s.CurrentM1Mask = fs.CurrentM1Mask;
    s.CurrentBLMask = fs.CurrentBLMask;
}

template<typename State_t, size_t NT, size_t RAM_WORDS_PER_ENV>
__launch_bounds__(NT) __global__
void initialize_states_kernel(const uint32_t num_envs,
                              const uint32_t noop_reset_steps,
                              State_t* states_buffer,
                              uint8_t* ram_buffer,
                              const State_t* cached_states_buffer,
                              const uint8_t* cached_ram_buffer,
                              uint32_t* tia_update_buffer,
                              const uint32_t* cached_tia_update_buffer,
                              uint8_t* frame_buffer,
                              uint8_t* previous_frame_buffer,
                              uint8_t* reset_screen_buffer,
                              const uint8_t* cached_frame_buffer,
                              const uint8_t* cached_previous_frame_buffer,
                              const uint8_t* cached_reset_screen_buffer,
                              uint32_t* rand_states_buffer,
                              uint32_t* cache_index_buffer,
                              frame_state* frame_states_buffer,
                              frame_state* cached_frame_states_buffer)
{
    static_assert(RAM_WORDS_PER_ENV <= detail::MAX_RAM_WORDS_PER_ENV,
                  "RAM_WORDS_PER_ENV exceeds CuLE's maximum Atari RAM allocation");

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= num_envs)
    {
        return;
    }

    prng gen(rand_states_buffer[global_index]);
    const size_t cache_index = gen.sample() % noop_reset_steps;
    cache_index_buffer[global_index] = cache_index;

    uint32_t ram[RAM_WORDS_PER_ENV];

    states_buffer[global_index] = cached_states_buffer[cache_index];
    frame_states_buffer[global_index] = cached_frame_states_buffer[cache_index];

    uint32_t * ram_int = (uint32_t*) cached_ram_buffer + (RAM_WORDS_PER_ENV * cache_index);

    #pragma loop unroll
    for(int32_t i = 0; i < int32_t(RAM_WORDS_PER_ENV); i++)
    {
        ram[i] = ram_int[i];
    }

    detail::store_env_ram_words<RAM_WORDS_PER_ENV>(ram_buffer, global_index, ram);

    const uint32_t* cached_tia = cached_tia_update_buffer + (ENV_UPDATE_SIZE * cache_index);
    uint32_t* tia = tia_update_buffer + (ENV_UPDATE_SIZE * global_index);
    #pragma unroll 1
    for(int32_t i = 0; i < int32_t(ENV_UPDATE_SIZE); ++i)
    {
        tia[i] = cached_tia[i];
    }
    tia[0] = 0;

    const uint8_t* cached_frame = cached_frame_buffer + (cache_index * 300 * SCREEN_WIDTH);
    uint8_t* frame = frame_buffer + (global_index * 300 * SCREEN_WIDTH);
    const uint8_t* cached_previous_frame = cached_previous_frame_buffer + (cache_index * 300 * SCREEN_WIDTH);
    uint8_t* previous_frame = previous_frame_buffer + (global_index * 300 * SCREEN_WIDTH);
    #pragma unroll 1
    for(int32_t i = 0; i < int32_t(300 * SCREEN_WIDTH); ++i)
    {
        frame[i] = cached_frame[i];
        previous_frame[i] = cached_previous_frame[i];
    }

    if(reset_screen_buffer != nullptr)
    {
        const uint8_t* visible_frame =
            cached_reset_screen_buffer != nullptr
                ? cached_reset_screen_buffer + (cache_index * 300 * SCREEN_WIDTH)
                : (cached_frame_states_buffer[cache_index].frameBufferIndex == 0
                       ? cached_frame
                       : cached_previous_frame);
        uint8_t* reset_screen = reset_screen_buffer + (global_index * 300 * SCREEN_WIDTH);
        #pragma unroll 1
        for(int32_t i = 0; i < int32_t(300 * SCREEN_WIDTH); ++i)
        {
            reset_screen[i] = visible_frame[i];
        }
    }
}

template<typename State_t, size_t NT, size_t RAM_WORDS_PER_ENV>
__launch_bounds__(NT) __global__
void reset_kernel(const uint32_t num_envs,
                  const size_t noop_reset_steps,
                  State_t* states_buffer,
                  uint8_t* ram_buffer,
                  const State_t* cached_states_buffer,
                  const uint8_t* cached_ram_buffer,
                  uint32_t* tia_update_buffer,
                  const uint32_t* cached_tia_update_buffer,
                  uint8_t* frame_buffer,
                  uint8_t* previous_frame_buffer,
                  uint8_t* reset_screen_buffer,
                  const uint8_t* cached_frame_buffer,
                  const uint8_t* cached_previous_frame_buffer,
                  const uint8_t* cached_reset_screen_buffer,
                  frame_state* frame_states_buffer,
                  frame_state* cached_frame_states_buffer,
                  uint32_t* cache_index_buffer,
                  uint32_t* rand_states_buffer)
{
    static_assert(RAM_WORDS_PER_ENV <= detail::MAX_RAM_WORDS_PER_ENV,
                  "RAM_WORDS_PER_ENV exceeds CuLE's maximum Atari RAM allocation");

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= num_envs)
    {
        return;
    }

    if(states_buffer[global_index].tiaFlags[FLAG_ALE_TERMINAL])
    {
        prng gen(rand_states_buffer[global_index]);
        const size_t sample = gen.sample();
        const size_t cache_index = sample % noop_reset_steps;

        states_buffer[global_index] = cached_states_buffer[cache_index];
        states_buffer[global_index].tiaFlags.set(FLAG_ALE_TERMINAL);
        cache_index_buffer[global_index] = cache_index;
        frame_states_buffer[global_index] = cached_frame_states_buffer[cache_index];

        uint32_t ram[RAM_WORDS_PER_ENV];
        uint32_t * ram_int = ((uint32_t*) cached_ram_buffer) + (RAM_WORDS_PER_ENV * cache_index);

        #pragma loop unroll
        for(int32_t i = 0; i < int32_t(RAM_WORDS_PER_ENV); i++)
        {
            ram[i] = ram_int[i];
        }

        detail::store_env_ram_words<RAM_WORDS_PER_ENV>(ram_buffer, global_index, ram);

        const uint32_t* cached_tia = cached_tia_update_buffer + (ENV_UPDATE_SIZE * cache_index);
        uint32_t* tia = tia_update_buffer + (ENV_UPDATE_SIZE * global_index);
        #pragma unroll 1
        for(int32_t i = 0; i < int32_t(ENV_UPDATE_SIZE); ++i)
        {
            tia[i] = cached_tia[i];
        }
        tia[0] = 0;

        const uint8_t* cached_frame = cached_frame_buffer + (cache_index * 300 * SCREEN_WIDTH);
        uint8_t* frame = frame_buffer + (global_index * 300 * SCREEN_WIDTH);
        const uint8_t* cached_previous_frame = cached_previous_frame_buffer + (cache_index * 300 * SCREEN_WIDTH);
        uint8_t* previous_frame = previous_frame_buffer + (global_index * 300 * SCREEN_WIDTH);
        #pragma unroll 1
        for(int32_t i = 0; i < int32_t(300 * SCREEN_WIDTH); ++i)
        {
            frame[i] = cached_frame[i];
            previous_frame[i] = cached_previous_frame[i];
        }

        if(reset_screen_buffer != nullptr)
        {
            const uint8_t* visible_frame =
                cached_reset_screen_buffer != nullptr
                    ? cached_reset_screen_buffer + (cache_index * 300 * SCREEN_WIDTH)
                    : (cached_frame_states_buffer[cache_index].frameBufferIndex == 0
                           ? cached_frame
                           : cached_previous_frame);
            uint8_t* reset_screen = reset_screen_buffer + (global_index * 300 * SCREEN_WIDTH);
            #pragma unroll 1
            for(int32_t i = 0; i < int32_t(300 * SCREEN_WIDTH); ++i)
            {
                reset_screen[i] = visible_frame[i];
            }
        }
    }
}

template<typename State_t, typename Environment_t, size_t NT>
__launch_bounds__(NT) __global__
void step_kernel(const uint32_t num_envs,
                 const bool fire_reset,
                 State_t* states_buffer,
                 uint8_t* ram_buffer,
                 uint32_t* tia_update_buffer,
                 const Action* player_a_buffer,
                 const Action* player_b_buffer,
                 bool* done_buffer)
{
    static_assert(Environment_t::RAM_WORDS_PER_ENV <= detail::MAX_RAM_WORDS_PER_ENV,
                  "RAM_WORDS_PER_ENV exceeds CuLE's maximum Atari RAM allocation");

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if((global_index >= num_envs) || done_buffer[global_index])
    {
        return;
    }

    states_buffer += global_index;

    uint32_t ram[Environment_t::RAM_WORDS_PER_ENV];

    State_t s;

    {
        state_store_load_helper(s, *states_buffer);
        detail::load_env_ram_words<Environment_t::RAM_WORDS_PER_ENV>(ram_buffer, global_index, ram);
    }

    Action player_a_action = ACTION_NOOP;
    Action player_b_action = ACTION_NOOP;

    if(player_a_buffer != nullptr)
    {
        player_a_action = player_a_buffer[global_index];
    }
    if(player_b_buffer != nullptr)
    {
        player_b_action = player_b_buffer[global_index];
    }

    if(fire_reset && s.tiaFlags[FLAG_ALE_LOST_LIFE])
    {
        player_a_action = ACTION_FIRE;
        s.tiaFlags.clear(FLAG_ALE_LOST_LIFE);
    }

    s.ram = ram;
    s.rom = gpu_rom;
    s.tia_update_buffer = tia_update_buffer + (ENV_UPDATE_SIZE * global_index);

    Environment_t::act(s, player_a_action, player_b_action);

    // Match the CPU step functor: preprocess replays a bounded per-step TIA
    // stream, so partial frames need an explicit sentinel before stale data.
    if(s.tia_update_buffer != nullptr)
    {
        *s.tia_update_buffer++ = uint32_t(0xFD);
    }

    {
        state_store_load_helper(*states_buffer, s);
        detail::store_env_ram_words<Environment_t::RAM_WORDS_PER_ENV>(ram_buffer, global_index, ram);
    }
}

template<typename State_t, typename ALE_t, size_t NT, size_t RAM_WORDS_PER_ENV>
__launch_bounds__(NT) __global__
void get_data_kernel(const int32_t num_envs,
                     const bool episodic_life,
                     State_t* states_buffer,
                     const uint8_t* ram_buffer,
                     bool* done_buffer,
                     float* rewards_buffer,
                     int32_t* lives_buffer)
{
    static_assert(RAM_WORDS_PER_ENV <= detail::MAX_RAM_WORDS_PER_ENV,
                  "RAM_WORDS_PER_ENV exceeds CuLE's maximum Atari RAM allocation");

    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if((global_index >= num_envs) || done_buffer[global_index])
    {
        return;
    }

    State_t& s = states_buffer[global_index];
    uint32_t ram[RAM_WORDS_PER_ENV];
    detail::load_env_ram_words<RAM_WORDS_PER_ENV>(ram_buffer, global_index, ram);
    s.ram = ram;

    const bool terminal = s.tiaFlags[FLAG_ALE_TERMINAL];
    const uint32_t old_lives = lives_buffer[global_index];
    const uint32_t new_lives = terminal ? 0 : ALE_t::getLives(s);
    lives_buffer[global_index] = new_lives;

    const bool lost_life = new_lives < old_lives;
    s.tiaFlags.template change<FLAG_ALE_LOST_LIFE>(lost_life);

    rewards_buffer[global_index] += ALE_t::getRewards(s);
    done_buffer[global_index] |= terminal || (episodic_life && lost_life);

    s.score = ALE_t::getScore(s);
}

template<typename State_t, size_t NT>
__launch_bounds__(NT) __global__
void process_kernel(const uint32_t num_envs,
                    const bool last_frame,
                    const bool render_frame,
                    const uint32_t* tia_update_buffer,
                    const uint32_t* cached_tia_update_buffer,
                    const uint32_t* cache_index_buffer,
                    State_t* states_buffer,
                    frame_state* frame_states_buffer,
                    uint8_t* frame_buffer,
                    uint8_t* previous_frame_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= num_envs)
    {
        return;
    }

    frame_states_buffer += global_index;

    frame_state fs;
    state_store_load_helper(fs, *frame_states_buffer);
    const State_t& s = states_buffer[global_index];
    fs.srcBuffer = tia_update_buffer + (global_index * ENV_UPDATE_SIZE);
    const bool is_terminal = s.tiaFlags[FLAG_ALE_TERMINAL];
    const bool is_started  = s.tiaFlags[FLAG_ALE_STARTED];

    if(last_frame && is_started && is_terminal)
    {
        states_buffer[global_index].tiaFlags.clear(FLAG_ALE_TERMINAL);
        fs.srcBuffer = cached_tia_update_buffer + (cache_index_buffer[global_index] * ENV_UPDATE_SIZE);
    }
    fs.cpuCycles = s.cpuCycles;
    uint8_t* primary_frame_buffer = render_frame ? frame_buffer : nullptr;
    uint8_t* secondary_frame_buffer = render_frame ? previous_frame_buffer : nullptr;
    preprocess::bindFrameBuffers(fs,
                                 primary_frame_buffer == nullptr ? nullptr : &primary_frame_buffer[global_index * 300 * SCREEN_WIDTH],
                                 secondary_frame_buffer == nullptr ? nullptr : &secondary_frame_buffer[global_index * 300 * SCREEN_WIDTH]);

    preprocess::state_to_buffer(fs, s.clockAtLastUpdate);

    state_store_load_helper(*frame_states_buffer, fs);
}

template<size_t NT>
__launch_bounds__(NT) __global__
void apply_palette_kernel(const int32_t num_envs,
                          const int32_t screen_height,
                          const int32_t num_channels,
                          uint8_t* dst_buffer,
                          const frame_state* frame_states_buffer,
                          const uint8_t* src_buffer,
                          const uint8_t* previous_frame_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index < num_envs * screen_height * SCREEN_WIDTH)
    {
        const uint32_t state_index = global_index / (screen_height * SCREEN_WIDTH);
        const uint8_t* source = frame_states_buffer[state_index].frameBufferIndex == 0 ? src_buffer : previous_frame_buffer;

        // slide the start index of the src_buffer forward to account for the
        // mismatch between the number of PAL and NTSC rows
        source += state_index * SCREEN_WIDTH * (300 - screen_height);

        int32_t color = source[global_index];
        dst_buffer += num_channels * global_index;

        if(num_channels == 3)
        {
            int32_t rgb = gpu_NTSCPalette[color];
            dst_buffer[0] = uint8_t(rgb >> 16);  // r
            dst_buffer[1] = uint8_t(rgb >>  8);  // g
            dst_buffer[2] = uint8_t(rgb >>  0);  // b
        }
        else
        {
            dst_buffer[0] = uint8_t(gpu_NTSCPalette[color + 1] & 0xFF);
        }
    }
}

template<size_t NT>
__launch_bounds__(NT) __global__
void apply_palette_gray4_kernel(const int32_t num_envs,
                                const int32_t screen_height,
                                uint8_t* dst_buffer,
                                const frame_state* frame_states_buffer,
                                const uint8_t* src_buffer,
                                const uint8_t* previous_frame_buffer)
{
    constexpr uint32_t PIXELS_PER_THREAD = 4;
    const uint32_t pixels_per_env = screen_height * SCREEN_WIDTH;
    const uint32_t words_per_env = pixels_per_env / PIXELS_PER_THREAD;
    const uint32_t global_word_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_word_index < uint32_t(num_envs) * words_per_env)
    {
        const uint32_t state_index = global_word_index / words_per_env;
        const uint32_t word_index = global_word_index - (state_index * words_per_env);
        const uint8_t* source =
            frame_states_buffer[state_index].frameBufferIndex == 0 ? src_buffer : previous_frame_buffer;
        source += (state_index * 300 * SCREEN_WIDTH) + (word_index * PIXELS_PER_THREAD);

        const uint32_t colors = *reinterpret_cast<const uint32_t*>(source);
        const uint32_t gray =
            ((gpu_NTSCPalette[(colors & 0x000000FFU) + 1] & 0xFFU) << 0) |
            ((gpu_NTSCPalette[((colors >> 8) & 0x000000FFU) + 1] & 0xFFU) << 8) |
            ((gpu_NTSCPalette[((colors >> 16) & 0x000000FFU) + 1] & 0xFFU) << 16) |
            ((gpu_NTSCPalette[((colors >> 24) & 0x000000FFU) + 1] & 0xFFU) << 24);

        reinterpret_cast<uint32_t*>(dst_buffer)[global_word_index] = gray;
    }
}

template<size_t NT>
__launch_bounds__(NT) __global__
void apply_palette_screen_kernel(const int32_t num_envs,
                                 const int32_t screen_height,
                                 const int32_t num_channels,
                                 uint8_t* dst_buffer,
                                 const uint8_t* screen_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index < num_envs * screen_height * SCREEN_WIDTH)
    {
        const uint32_t state_index = global_index / (screen_height * SCREEN_WIDTH);
        const uint32_t pixel_index = global_index % (screen_height * SCREEN_WIDTH);
        const uint8_t* source = screen_buffer + (state_index * 300 * SCREEN_WIDTH);

        const int32_t color = source[pixel_index];
        dst_buffer += num_channels * global_index;

        if(num_channels == 3)
        {
            int32_t rgb = gpu_NTSCPalette[color];
            dst_buffer[0] = uint8_t(rgb >> 16);
            dst_buffer[1] = uint8_t(rgb >>  8);
            dst_buffer[2] = uint8_t(rgb >>  0);
        }
        else
        {
            dst_buffer[0] = uint8_t(gpu_NTSCPalette[color + 1] & 0xFF);
        }
    }
}

template<size_t NT>
__launch_bounds__(NT) __global__
void apply_rescale_kernel(const int32_t num_envs,
                          const int32_t screen_height,
                          uint8_t * dst_buffer,
                          const frame_state* frame_states_buffer,
                          const uint8_t * src_buffer,
                          const uint8_t * previous_frame_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index < (num_envs * SCALED_SCREEN_SIZE))
    {
        // slide the start index of the src_buffer forward to account for the
        // mismatch between the number of PAL and NTSC rows
        const uint32_t state_index = global_index / SCALED_SCREEN_SIZE;
        const uint8_t* source = frame_states_buffer[state_index].frameBufferIndex == 0 ? src_buffer : previous_frame_buffer;
        source += state_index * SCREEN_WIDTH * (300 - screen_height);

        const float S_R = float(screen_height) / 84.0f;
        const float S_C = float(SCREEN_WIDTH) / 84.0f;

        const size_t row  = std::floor(float(global_index) / 84.0f);
        const size_t col  = global_index % 84;

        const float rf = (0.5f + row) * S_R;
        const float cf = (0.5f + col) * S_C;
        const size_t r = std::floor(rf - 0.5f);
        const size_t c = std::floor(cf - 0.5f);

        const float delta_R = rf - (0.5f + r);
        const float delta_C = cf - (0.5f + c);

        const float color_0_0 = gpu_NTSCPalette[source[(r + 0) * SCREEN_WIDTH + (c + 0)] + 1] & 0xFF;
        const float color_0_1 = gpu_NTSCPalette[source[(r + 1) * SCREEN_WIDTH + (c + 0)] + 1] & 0xFF;
        const float color_1_0 = gpu_NTSCPalette[source[(r + 0) * SCREEN_WIDTH + (c + 1)] + 1] & 0xFF;
        const float color_1_1 = gpu_NTSCPalette[source[(r + 1) * SCREEN_WIDTH + (c + 1)] + 1] & 0xFF;

        const float value = (color_0_0 * (1.0f - delta_R) * (1.0f - delta_C)) +
                            (color_0_1 * delta_R * (1.0f - delta_C)) +
                            (color_1_0 * (1.0f - delta_R) * delta_C) +
                            (color_1_1 * delta_R * delta_C);
        dst_buffer[global_index] = value + 0.5f;
    }
}

template<size_t NT>
__launch_bounds__(NT) __global__
void apply_rescale_screen_kernel(const int32_t num_envs,
                                 const int32_t screen_height,
                                 uint8_t * dst_buffer,
                                 const uint8_t * screen_buffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index < (num_envs * SCALED_SCREEN_SIZE))
    {
        const uint32_t state_index = global_index / SCALED_SCREEN_SIZE;
        const uint8_t* source = screen_buffer + (state_index * 300 * SCREEN_WIDTH);

        const float S_R = float(screen_height) / 84.0f;
        const float S_C = float(SCREEN_WIDTH) / 84.0f;

        const size_t row  = std::floor(float(global_index % SCALED_SCREEN_SIZE) / 84.0f);
        const size_t col  = global_index % 84;

        const float rf = (0.5f + row) * S_R;
        const float cf = (0.5f + col) * S_C;
        const size_t r = std::floor(rf - 0.5f);
        const size_t c = std::floor(cf - 0.5f);

        const float delta_R = rf - (0.5f + r);
        const float delta_C = cf - (0.5f + c);

        const float color_0_0 = gpu_NTSCPalette[source[(r + 0) * SCREEN_WIDTH + (c + 0)] + 1] & 0xFF;
        const float color_0_1 = gpu_NTSCPalette[source[(r + 1) * SCREEN_WIDTH + (c + 0)] + 1] & 0xFF;
        const float color_1_0 = gpu_NTSCPalette[source[(r + 0) * SCREEN_WIDTH + (c + 1)] + 1] & 0xFF;
        const float color_1_1 = gpu_NTSCPalette[source[(r + 1) * SCREEN_WIDTH + (c + 1)] + 1] & 0xFF;

        const float value = (color_0_0 * (1.0f - delta_R) * (1.0f - delta_C)) +
                            (color_0_1 * delta_R * (1.0f - delta_C)) +
                            (color_1_0 * (1.0f - delta_R) * delta_C) +
                            (color_1_1 * delta_R * delta_C);
        dst_buffer[global_index] = value + 0.5f;
    }
}

/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
template<size_t NT>
__launch_bounds__(NT) __global__
void action_kernel(const uint32_t num_envs,
                   const uint32_t minimal_actions_size,
                   const uint32_t num_entries,
                   const Action* minimal_actions_ptr,
                   uint32_t* rand_states_ptr,
                   Action* actionsBuffer)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= num_envs)
    {
        return;
    }

    prng gen(rand_states_ptr[global_index]);

    for(int i = global_index; i < num_entries; i += num_envs)
    {
        actionsBuffer[i] = minimal_actions_ptr[gen.sample() % minimal_actions_size];
    }
}

template<typename State_t, size_t NT>
__launch_bounds__(NT) __global__
void get_states_kernel(const uint32_t num_envs,
                       const int32_t* indices,
                       const State_t* states_buffer,
                       const frame_state* frame_states_buffer,
                       State_t* output_states_buffer,
                       frame_state* output_frame_states_buffer,
                       const uint8_t* ram_buffer,
                       const uint32_t ram_words_per_env,
                       uint8_t* output_states_ram)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= num_envs)
    {
        return;
    }

    const size_t index = indices[global_index];
    const State_t& s = states_buffer[index];
    State_t& t = output_states_buffer[global_index];

    t = s;
    output_frame_states_buffer[global_index] = frame_states_buffer[index];

    output_states_ram += 256 * global_index;
    uint32_t ram_words[detail::MAX_RAM_WORDS_PER_ENV];
    uint32_t* output_ram_words = reinterpret_cast<uint32_t*>(output_states_ram);
    detail::load_env_ram_words(ram_buffer, index, ram_words_per_env, ram_words);

    for(int32_t i = 0; i < int32_t(ram_words_per_env); ++i)
    {
        output_ram_words[i] = ram_words[i];
    }
}

template<typename State_t, size_t NT>
__launch_bounds__(NT) __global__
void set_states_kernel(const uint32_t num_envs,
                       const int32_t* indices,
                       State_t* states_buffer,
                       frame_state* frame_states_buffer,
                       const State_t* input_states_buffer,
                       const frame_state* input_frame_states_buffer,
                       uint8_t* ram_buffer,
                       const uint32_t ram_words_per_env,
                       const uint8_t* input_states_ram)
{
    const uint32_t global_index = (NT * blockIdx.x) + threadIdx.x;

    if(global_index >= num_envs)
    {
        return;
    }

    const size_t index = indices[global_index];
    const State_t& s = input_states_buffer[global_index];
    State_t& t = states_buffer[index];
    frame_state& tf = frame_states_buffer[index];

    t.A = s.A;
    t.X = s.X;
    t.Y = s.Y;
    t.SP = s.SP;
    t.PC = s.PC;
    t.addr = s.addr;
    t.value = s.value;
    t.noise = s.noise;

    t.cpuCycles = s.cpuCycles;
    t.bank = s.bank;

    t.resistance = s.resistance;

    t.GRP = s.GRP;
    t.HM = s.HM;
    t.PF = s.PF;
    t.POS = s.POS;
    t.CurrentGRP0 = s.CurrentGRP0;
    t.CurrentGRP1 = s.CurrentGRP1;

    t.collision = s.collision;
    t.clockWhenFrameStarted = s.clockWhenFrameStarted;
    t.clockAtLastUpdate = s.clockAtLastUpdate;
    t.dumpDisabledCycle = s.dumpDisabledCycle;
    t.VSYNCFinishClock = s.VSYNCFinishClock;
    t.lastHMOVEClock = s.lastHMOVEClock;
    t.displayYStart = s.displayYStart;
    t.displayHeight = s.displayHeight;

    t.riotData = s.riotData;
    t.cyclesWhenTimerSet = s.cyclesWhenTimerSet;
    t.cyclesWhenInterruptReset = s.cyclesWhenInterruptReset;

    t.sysFlags = s.sysFlags;
    t.tiaFlags = s.tiaFlags;

    t.frameData = s.frameData;
    t.bootProgress = s.bootProgress;
    t.bootPhase = s.bootPhase;
    t.rand = s.rand;
    t.score = s.score;
    t.M0CosmicArkCounter = s.M0CosmicArkCounter;

    t.CurrentPFMask = &playfield_accessor(0, 0);
    t.CurrentP0Mask = &player_mask_accessor(0, 0, 0, 0);
    t.CurrentP1Mask = &player_mask_accessor(0, 0, 0, 0);
    t.CurrentM0Mask = &missle_accessor(0, 0, 0, 0);
    t.CurrentM1Mask = &missle_accessor(0, 0, 0, 0);
    t.CurrentBLMask = &ball_accessor(0, 0, 0);

    state_store_load_helper(tf, input_frame_states_buffer[global_index]);
    refresh_frame_state_masks(tf);

    input_states_ram += 256 * global_index;
    const uint32_t* input_ram_words = reinterpret_cast<const uint32_t*>(input_states_ram);

    detail::store_env_ram_words(ram_buffer, index, ram_words_per_env, input_ram_words);
}

} // end namespace cuda
} // end namespace atari
} // end namespace cule
