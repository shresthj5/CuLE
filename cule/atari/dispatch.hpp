#pragma once

#include <cule/config.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/ale.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/environment.hpp>
#include <cule/atari/functors.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/joystick.hpp>
#include <cule/atari/m6502.hpp>
#include <cule/atari/paddles.hpp>
#include <cule/atari/palettes.hpp>
#include <cule/atari/png.hpp>
#include <cule/atari/prng.hpp>
#include <cule/atari/preprocess.hpp>
#include <cule/atari/rom.hpp>

#include <agency/agency.hpp>

#include <random>

namespace cule
{
namespace atari
{
namespace dispatch
{

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
reset(ExecutionPolicy&& policy,
      Wrapper& wrap,
      uint32_t* seedBuffer)
{
    agency::vector<uint32_t, agency::allocator<uint32_t>> rand_temp_buffer;

    if(seedBuffer == nullptr)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        rand_temp_buffer.resize(wrap.size(), 0);
        seedBuffer = rand_temp_buffer.data();

        std::generate_n(seedBuffer, wrap.size(), [&]{ return rng(); });
    }

    agency::bulk_invoke(policy(1),
                        initialize_functor<Environment>{},
                        wrap.rom_indices_ptr,
                        &wrap.cart,
                        wrap.cached_ram_ptr,
                        wrap.cached_states_ptr,
                        wrap.cached_frame_states_ptr,
                        wrap.rand_states_ptr,
                        seedBuffer);

    size_t boot_frames = 0;
    constexpr size_t MAX_EXTRA_BOOT_FRAMES = 4096;
    while((boot_frames < ENV_BASE_FRAMES) ||
          !wrap.cached_states_ptr[0].tiaFlags[FLAG_ALE_STARTED])
    {
        agency::bulk_invoke(policy(1),
                            step_functor<Environment>{},
                            true,
                            wrap.cached_states_ptr,
                            wrap.cached_tia_update_ptr,
                            nullptr,
                            nullptr,
                            nullptr);
        agency::bulk_invoke(policy(1),
                            preprocess_functor<Environment>{},
                            true,
                            wrap.cached_tia_update_ptr,
                            nullptr,
                            wrap.cached_states_ptr,
                            wrap.cache_index_ptr,
                            wrap.cached_frame_states_ptr,
                            wrap.cached_frame_ptr,
                            wrap.cached_previous_frame_ptr);

        if(!wrap.cached_states_ptr[0].tiaFlags[FLAG_ALE_STARTED] &&
           (wrap.cached_states_ptr[0].bootPhase == BOOT_DONE))
        {
            wrap.cached_states_ptr[0].tiaFlags.set(FLAG_ALE_STARTED);
        }
        ++boot_frames;
        CULE_ASSERT(boot_frames <= (ENV_BASE_FRAMES + MAX_EXTRA_BOOT_FRAMES),
                    "ALE reset did not reach a started state for "
                        << wrap.cart.game_name() << " within "
                        << (ENV_BASE_FRAMES + MAX_EXTRA_BOOT_FRAMES)
                        << " frames");
    }

    for (size_t i = 1; i < wrap.noop_reset_steps; i++)
    {
        wrap.cached_states_ptr[i] = wrap.cached_states_ptr[i - 1];
        wrap.cached_frame_states_ptr[i] = wrap.cached_frame_states_ptr[i - 1];

        wrap.cached_states_ptr[i].ram = (uint32_t*) (wrap.cached_ram_ptr + (wrap.cart.ram_size() * i));
        std::copy(wrap.cached_states_ptr[i - 1].ram,
                  wrap.cached_states_ptr[i - 1].ram + (wrap.cart.ram_size() / sizeof(uint32_t)),
                  wrap.cached_states_ptr[i].ram);
        std::copy(wrap.cached_frame_ptr + ((i - 1) * 300 * SCREEN_WIDTH),
                  wrap.cached_frame_ptr + (i * 300 * SCREEN_WIDTH),
                  wrap.cached_frame_ptr + (i * 300 * SCREEN_WIDTH));
        std::copy(wrap.cached_previous_frame_ptr + ((i - 1) * 300 * SCREEN_WIDTH),
                  wrap.cached_previous_frame_ptr + (i * 300 * SCREEN_WIDTH),
                  wrap.cached_previous_frame_ptr + (i * 300 * SCREEN_WIDTH));

        agency::bulk_invoke(policy(1),
                            step_functor<Environment>{},
                            true,
                            wrap.cached_states_ptr + i,
                            wrap.cached_tia_update_ptr + (i * ENV_UPDATE_SIZE),
                            nullptr,
                            nullptr,
                            nullptr);
        agency::bulk_invoke(policy(1),
                            preprocess_functor<Environment>{},
                            true,
                            wrap.cached_tia_update_ptr + (i * ENV_UPDATE_SIZE),
                            nullptr,
                            wrap.cached_states_ptr + i,
                            wrap.cache_index_ptr + i,
                            wrap.cached_frame_states_ptr + i,
                            wrap.cached_frame_ptr + (i * 300 * SCREEN_WIDTH),
                            wrap.cached_previous_frame_ptr + (i * 300 * SCREEN_WIDTH));
    }

    for (size_t i = 0; i < wrap.size(); i++)
    {
        prng gen(wrap.rand_states_ptr[i]);
        const size_t index = gen.sample() % wrap.noop_reset_steps;
        wrap.states_ptr[i] = wrap.cached_states_ptr[index];
        wrap.states_ptr[i].ram = (uint32_t *) (wrap.ram_ptr + (wrap.cart.ram_size() * i));
        std::copy(wrap.cached_states_ptr[index].ram,
                  wrap.cached_states_ptr[index].ram + (wrap.cart.ram_size() / sizeof(uint32_t)),
                  wrap.states_ptr[i].ram);
        wrap.cache_index_ptr[i] = index;
        std::copy(wrap.cached_tia_update_ptr + (index * ENV_UPDATE_SIZE),
                  wrap.cached_tia_update_ptr + ((index + 1) * ENV_UPDATE_SIZE),
                  wrap.tia_update_ptr + (i * ENV_UPDATE_SIZE));
        wrap.frame_states_ptr[i] = wrap.cached_frame_states_ptr[index];
        std::copy(wrap.cached_frame_ptr + (index * 300 * SCREEN_WIDTH),
                  wrap.cached_frame_ptr + ((index + 1) * 300 * SCREEN_WIDTH),
                  wrap.frame_ptr + (i * 300 * SCREEN_WIDTH));
        std::copy(wrap.cached_previous_frame_ptr + (index * 300 * SCREEN_WIDTH),
                  wrap.cached_previous_frame_ptr + ((index + 1) * 300 * SCREEN_WIDTH),
                  wrap.previous_frame_ptr + (i * 300 * SCREEN_WIDTH));
        wrap.tia_update_ptr[i * ENV_UPDATE_SIZE] = 0;
    }
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
reset_states(ExecutionPolicy&& policy,
             Wrapper& wrap)
{
    agency::bulk_invoke(policy(wrap.size()),
                        reset_functor<Environment>{},
                        &wrap.cart,
                        wrap.ram_ptr,
                        wrap.tia_update_ptr,
                        wrap.frame_ptr,
                        wrap.previous_frame_ptr,
                        wrap.noop_reset_steps,
                        wrap.states_ptr,
                        wrap.cached_states_ptr,
                        wrap.cached_ram_ptr,
                        wrap.cached_tia_update_ptr,
                        wrap.frame_states_ptr,
                        wrap.cached_frame_states_ptr,
                        wrap.cached_frame_ptr,
                        wrap.cached_previous_frame_ptr,
                        wrap.cache_index_ptr,
                        wrap.rand_states_ptr);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
get_states(ExecutionPolicy&& policy,
           Wrapper& wrap,
           const size_t num_states,
           const int32_t* indices,
           typename Wrapper::State_t* output_states,
           frame_state* output_frame_states,
           uint8_t*)
{
    agency::bulk_invoke(policy(num_states),
                        get_states_functor{},
                        indices,
                        wrap.states_ptr,
                        wrap.frame_states_ptr,
                        output_states,
                        output_frame_states);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
set_states(ExecutionPolicy&& policy,
           Wrapper& wrap,
           const size_t num_states,
           const int32_t* indices,
           const typename Wrapper::State_t* input_states,
           const frame_state* input_frame_states,
           const uint8_t*)
{
    agency::bulk_invoke(policy(num_states),
                        set_states_functor{},
                        indices,
                        &wrap.cart,
                        wrap.ram_ptr,
                        wrap.states_ptr,
                        wrap.frame_states_ptr,
                        input_states,
                        input_frame_states);
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
step(ExecutionPolicy&& policy,
     Wrapper& wrap,
     const bool fire_reset,
     const Action* playerABuffer,
     const Action* playerBBuffer,
     bool* doneBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        step_functor<Environment>{},
                        fire_reset,
                        wrap.states_ptr,
                        wrap.tia_update_ptr,
                        (Action*) playerABuffer,
                        (Action*) playerBBuffer,
                        doneBuffer);
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
get_data(ExecutionPolicy&& policy,
         Wrapper& wrap,
         const bool episodic_life,
         bool* doneBuffer,
         float* rewardsBuffer,
         int32_t* livesBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        get_data_functor<Environment>{},
                        episodic_life,
                        wrap.states_ptr,
                        doneBuffer,
                        rewardsBuffer,
                        livesBuffer);
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
preprocess(ExecutionPolicy&& policy,
           Wrapper& wrap,
           const bool last_frame,
           const uint32_t* tiaBuffer,
           uint8_t* frameBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        preprocess_functor<Environment>{},
                        last_frame,
                        tiaBuffer,
                        wrap.cached_tia_update_ptr,
                        wrap.states_ptr,
                        wrap.cache_index_ptr,
                        wrap.frame_states_ptr,
                        frameBuffer,
                        wrap.previous_frame_ptr);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
generate_frames(ExecutionPolicy&& policy,
                Wrapper& wrap,
                const bool rescale,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        generate_frame_functor{},
                        num_channels,
                        wrap.cart.screen_height(),
                        rescale,
                        wrap.frame_states_ptr,
                        wrap.frame_ptr,
                        wrap.previous_frame_ptr,
                        imageBuffer);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
generate_random_actions(ExecutionPolicy&& policy,
                        Wrapper& wrap,
                        Action* actionsBuffer,
                        const size_t N)
{
    const size_t num_entries = N == 0 ? wrap.size() : N;

    agency::bulk_invoke(policy(wrap.size()),
                        random_actions_functor{},
                        wrap.cart.minimal_actions().size(),
                        num_entries,
                        wrap.minimal_actions_ptr,
                        wrap.rand_states_ptr,
                        actionsBuffer);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
save_images(ExecutionPolicy&& policy,
            Wrapper& wrap,
            const bool rescale,
            const size_t num_channels,
            const size_t frame_index,
            const uint8_t* imageBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        png_functor{},
                        frame_index,
                        num_channels,
                        wrap.cart.screen_height(),
                        rescale,
                        imageBuffer);
}

} // end namespace dispatch
} // end namespace atari
} // end namespace cule

#ifdef __CUDACC__
#include <cule/atari/cuda/dispatch.hpp>
#endif
