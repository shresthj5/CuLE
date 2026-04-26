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
#include <cule/atari/ram.hpp>
#include <cule/atari/rom.hpp>

#include <agency/agency.hpp>

#include <random>
#include <vector>

namespace cule
{
namespace atari
{
namespace dispatch
{

inline bool
requires_frame_complete_reset_state(const rom& cart)
{
    switch(cart.game_id())
    {
    case games::GAME_ASSAULT:
        return true;
    default:
        return false;
    }
}

template<typename State_t>
void
apply_ale_v5_post_reset_normalization(const rom& cart,
                                      State_t& s)
{
    switch(cart.game_id())
    {
    case games::GAME_ASSAULT:
        // Official ALE reset consistently lands Assault with these startup
        // bytes cleared/seeded, while CuLE's generic boot sequence can leave
        // the stale pre-start values around. Those bytes drive later live-play
        // state and are the source of the deterministic step-31/32 drift.
        ram::write(s.ram, 0xB9, uint8_t(0x00));
        ram::write(s.ram, 0xE6, uint8_t(0xCF));
        break;
    default:
        break;
    }
}

inline double
stella_next_double(std::mt19937& rng,
                   uint64_t* draw_count = nullptr)
{
    if(draw_count != nullptr)
    {
        ++(*draw_count);
    }

    return static_cast<double>(rng()) /
           (static_cast<double>(std::mt19937::max()) + 1.0);
}

template<typename State_t>
void
terminate_cached_tia_stream(State_t& s,
                            uint32_t* tia_update_begin)
{
    if(s.tia_update_buffer == nullptr)
    {
        return;
    }

    if((s.tia_update_buffer == tia_update_begin) ||
       (*(s.tia_update_buffer - 1) != uint32_t(0xFD)))
    {
        *s.tia_update_buffer++ = uint32_t(0xFD);
    }
}

template<typename Environment,
         typename Wrapper>
void
capture_cached_reset_screen(Wrapper& wrap,
                            const size_t cache_index = 0)
{
    if((wrap.cached_reset_screen_ptr == nullptr) ||
       (wrap.cached_frame_ptr == nullptr) ||
       (wrap.cached_previous_frame_ptr == nullptr) ||
       (wrap.cached_frame_states_ptr == nullptr))
    {
        return;
    }

    const uint8_t* visible_frame =
        wrap.cached_frame_states_ptr[cache_index].frameBufferIndex == 0
            ? wrap.cached_frame_ptr
            : wrap.cached_previous_frame_ptr;
    std::copy(visible_frame + (cache_index * 300 * SCREEN_WIDTH),
              visible_frame + ((cache_index + 1) * 300 * SCREEN_WIDTH),
              wrap.cached_reset_screen_ptr + (cache_index * 300 * SCREEN_WIDTH));
}

template<typename Environment,
         typename Wrapper>
void
replay_cached_frame(Wrapper& wrap)
{
    auto& s = wrap.cached_states_ptr[0];
    auto& fs = wrap.cached_frame_states_ptr[0];

    fs.srcBuffer = wrap.cached_tia_update_ptr;
    fs.cpuCycles = s.cpuCycles;

    preprocess::bindFrameBuffers(fs,
                                 wrap.cached_frame_ptr,
                                 wrap.cached_previous_frame_ptr);
    preprocess::state_to_buffer(fs, s.clockAtLastUpdate);
}

template<typename Environment,
         typename Wrapper>
void
raw_cached_emulate_steps(Wrapper& wrap,
                         const Action player_a_action,
                         const Action player_b_action,
                         const bool select_pressed,
                         const size_t num_steps,
                         const bool count_public_frame_each_step)
{
    using ALE_t = typename Environment::ALE_t;
    using Controller_t = typename Environment::Controller_t;

    auto& s = wrap.cached_states_ptr[0];
    s.tia_update_buffer = wrap.cached_tia_update_ptr;

    ALE_t::noopIllegalActions(s);
    Environment::setSelectPressed(s, select_pressed);
    Controller_t::set_actions(s, player_a_action, player_b_action);

    for(size_t step = 0; step < num_steps; ++step)
    {
        Environment::emulate(s);

        if(count_public_frame_each_step)
        {
            Environment::increment(s);
        }
    }

    terminate_cached_tia_stream(s, wrap.cached_tia_update_ptr);

    replay_cached_frame<Environment>(wrap);
}

template<typename Environment,
         typename Wrapper>
void
raw_cached_reset_step(Wrapper& wrap,
                      const Action player_a_action,
                      const Action player_b_action,
                      const bool select_pressed,
                      const bool count_public_frame)
{
    raw_cached_emulate_steps<Environment>(wrap,
                                          player_a_action,
                                          player_b_action,
                                          select_pressed,
                                          1,
                                          count_public_frame);
}

template<typename Environment,
         typename Wrapper>
void
perform_ale_style_reset_act(Wrapper& wrap,
                            const Action requested_player_a_action,
                            std::mt19937& rng,
                            Action& last_player_a_action,
                            Action& last_player_b_action,
                            uint64_t& sticky_draw_count)
{
    const uint32_t frame_skip = wrap.reset_frame_skip == 0 ? 1U : wrap.reset_frame_skip;
    const double repeat_action_probability =
        static_cast<double>(wrap.reset_repeat_action_probability);

    for(uint32_t frame = 0; frame < frame_skip; ++frame)
    {
        if(stella_next_double(rng, &sticky_draw_count) >= repeat_action_probability)
        {
            last_player_a_action = requested_player_a_action;
        }
        if(stella_next_double(rng, &sticky_draw_count) >= repeat_action_probability)
        {
            last_player_b_action = ACTION_NOOP;
        }

        raw_cached_reset_step<Environment>(wrap,
                                           last_player_a_action,
                                           last_player_b_action,
                                           false,
                                           true);
    }
}

template<typename Environment,
         typename Wrapper>
void
perform_doubledunk_scripted_sticky_frame(Wrapper& wrap,
                                         const Action requested_player_a_action,
                                         std::mt19937& rng,
                                         Action& last_player_a_action,
                                         Action& last_player_b_action,
                                         uint64_t& sticky_draw_count)
{
    const double repeat_action_probability =
        static_cast<double>(wrap.reset_repeat_action_probability);

    if(stella_next_double(rng, &sticky_draw_count) >= repeat_action_probability)
    {
        last_player_a_action = requested_player_a_action;
    }
    if(stella_next_double(rng, &sticky_draw_count) >= repeat_action_probability)
    {
        last_player_b_action = ACTION_NOOP;
    }

    raw_cached_reset_step<Environment>(wrap,
                                       last_player_a_action,
                                       last_player_b_action,
                                       false,
                                       true);
}

template<typename Environment,
         typename Wrapper>
void
perform_cached_soft_reset(Wrapper& wrap)
{
    raw_cached_emulate_steps<Environment>(wrap,
                                          ACTION_RESET,
                                          ACTION_NOOP,
                                          false,
                                          ENV_RESET_FRAMES / 2,
                                          false);
}

template<typename Environment,
         typename Wrapper>
void
perform_generic_cached_boot_step(Wrapper& wrap)
{
    auto& s = wrap.cached_states_ptr[0];
    s.tia_update_buffer = wrap.cached_tia_update_ptr;

    Environment::act(s, ACTION_NOOP, ACTION_NOOP);

    terminate_cached_tia_stream(s, wrap.cached_tia_update_ptr);

    replay_cached_frame<Environment>(wrap);
}

template<typename Environment,
         typename Wrapper>
void
perform_doubledunk_go_down(Wrapper& wrap,
                           std::mt19937& rng,
                           Action& last_player_a_action,
                           Action& last_player_b_action,
                           uint64_t& sticky_draw_count)
{
    auto& s = wrap.cached_states_ptr[0];
    const uint8_t previous_selection = ram::read(s.ram, 0xB0);

    size_t attempts = 0;
    while(previous_selection == ram::read(s.ram, 0xB0))
    {
        perform_doubledunk_scripted_sticky_frame<Environment>(wrap,
                                                              ACTION_DOWN,
                                                              rng,
                                                              last_player_a_action,
                                                              last_player_b_action,
                                                              sticky_draw_count);
        perform_doubledunk_scripted_sticky_frame<Environment>(wrap,
                                                              ACTION_NOOP,
                                                              rng,
                                                              last_player_a_action,
                                                              last_player_b_action,
                                                              sticky_draw_count);
        ++attempts;
        CULE_ASSERT(attempts <= 256,
                    "DoubleDunk reset failed to advance the menu selection");
    }
}

template<typename Environment,
         typename Wrapper>
void
perform_doubledunk_option_adjustment(Wrapper& wrap,
                                     std::mt19937& rng,
                                     Action& last_player_a_action,
                                     Action& last_player_b_action,
                                     const uint8_t bit_of_interest,
                                     const bool enabled,
                                     uint64_t& sticky_draw_count)
{
    auto& s = wrap.cached_states_ptr[0];
    size_t attempts = 0;
    while(((ram::read(s.ram, 0x80) & bit_of_interest) == bit_of_interest) != enabled)
    {
        perform_doubledunk_scripted_sticky_frame<Environment>(
            wrap,
            enabled ? ACTION_RIGHT : ACTION_LEFT,
            rng,
            last_player_a_action,
            last_player_b_action,
            sticky_draw_count);
        perform_doubledunk_scripted_sticky_frame<Environment>(wrap,
                                                              ACTION_NOOP,
                                                              rng,
                                                              last_player_a_action,
                                                              last_player_b_action,
                                                              sticky_draw_count);
        ++attempts;
        CULE_ASSERT(attempts <= 256,
                    "DoubleDunk reset failed to converge on the requested mode bits");
    }
}

template<typename Environment,
         typename Wrapper>
void
perform_exact_doubledunk_reset_sequence(Wrapper& wrap,
                                        std::mt19937& rng,
                                        uint64_t& sticky_draw_count,
                                        const bool capture_reset_screen)
{
    auto& s = wrap.cached_states_ptr[0];
    Action last_player_a_action = ACTION_NOOP;
    Action last_player_b_action = ACTION_NOOP;

    // StellaEnvironment::pressSelect() performs one select-asserted raw update,
    // then one NOOP act() and increments the public frame once.
    raw_cached_reset_step<Environment>(wrap,
                                       ACTION_NOOP,
                                       ACTION_NOOP,
                                       true,
                                       false);
    raw_cached_reset_step<Environment>(wrap,
                                       ACTION_NOOP,
                                       ACTION_NOOP,
                                       false,
                                       true);

    // Follow ALE's DoubleDunkSettings::setMode() logic exactly so sticky-action
    // randomness can affect menu traversal in the same places as upstream.
    constexpr uint32_t DEFAULT_DOUBLE_DUNK_MODE = 0U;

    perform_doubledunk_go_down<Environment>(
        wrap, rng, last_player_a_action, last_player_b_action, sticky_draw_count);
    perform_doubledunk_go_down<Environment>(
        wrap, rng, last_player_a_action, last_player_b_action, sticky_draw_count);

    perform_doubledunk_option_adjustment<Environment>(
        wrap,
        rng,
        last_player_a_action,
        last_player_b_action,
        0x08,
        (DEFAULT_DOUBLE_DUNK_MODE & 0x1U) != 0U,
        sticky_draw_count);

    perform_doubledunk_go_down<Environment>(
        wrap, rng, last_player_a_action, last_player_b_action, sticky_draw_count);
    perform_doubledunk_option_adjustment<Environment>(
        wrap,
        rng,
        last_player_a_action,
        last_player_b_action,
        0x10,
        (DEFAULT_DOUBLE_DUNK_MODE & 0x2U) != 0U,
        sticky_draw_count);

    perform_doubledunk_go_down<Environment>(
        wrap, rng, last_player_a_action, last_player_b_action, sticky_draw_count);
    perform_doubledunk_option_adjustment<Environment>(
        wrap,
        rng,
        last_player_a_action,
        last_player_b_action,
        0x04,
        (DEFAULT_DOUBLE_DUNK_MODE & 0x4U) != 0U,
        sticky_draw_count);

    perform_doubledunk_go_down<Environment>(
        wrap, rng, last_player_a_action, last_player_b_action, sticky_draw_count);
    perform_doubledunk_option_adjustment<Environment>(
        wrap,
        rng,
        last_player_a_action,
        last_player_b_action,
        0x20,
        (DEFAULT_DOUBLE_DUNK_MODE & 0x8U) != 0U,
        sticky_draw_count);

    perform_cached_soft_reset<Environment>(wrap);
    last_player_a_action = ACTION_NOOP;
    last_player_b_action = ACTION_NOOP;

    perform_doubledunk_scripted_sticky_frame<Environment>(wrap,
                                                          ACTION_UPFIRE,
                                                          rng,
                                                          last_player_a_action,
                                                          last_player_b_action,
                                                          sticky_draw_count);
    perform_doubledunk_scripted_sticky_frame<Environment>(wrap,
                                                          ACTION_NOOP,
                                                          rng,
                                                          last_player_a_action,
                                                          last_player_b_action,
                                                          sticky_draw_count);

    // StellaEnvironment::reset() applies a second softReset() after setMode(),
    // then replays the ROM's starting action list via raw emulate() calls.
    perform_cached_soft_reset<Environment>(wrap);
    raw_cached_reset_step<Environment>(wrap,
                                       Environment::getStartAction(s),
                                       ACTION_NOOP,
                                       false,
                                       false);
    if(capture_reset_screen)
    {
        capture_cached_reset_screen<Environment>(wrap);
    }

    Environment::setBootProgress(s, 0);
    Environment::setBootPhase(s, BOOT_DONE);
    s.tiaFlags.set(FLAG_ALE_STARTED);
}

template<typename Environment,
         typename Wrapper>
void
perform_exact_doubledunk_reset(Wrapper& wrap,
                               const uint32_t ale_seed)
{
    auto& s = wrap.cached_states_ptr[0];
    size_t boot_frames = 0;
    constexpr size_t MAX_BOOT_FRAMES_TO_MODE_SETUP = ENV_BASE_FRAMES + 4096;
    while((Environment::getBootPhase(s) != BOOT_MODE_SETUP) ||
          (Environment::getBootProgress(s) != 0))
    {
        perform_generic_cached_boot_step<Environment>(wrap);
        ++boot_frames;
        CULE_ASSERT(boot_frames <= MAX_BOOT_FRAMES_TO_MODE_SETUP,
                    "DoubleDunk reset did not reach BOOT_MODE_SETUP");
    }

    // Gymnasium's ALE v5 wrapper seeds ALE, then calls loadROM() and finally
    // reset_game(). The visible reset therefore starts from the post-loadROM
    // sticky RNG state, while ALE's wrapper-level screen bookkeeping has also
    // observed the hidden loadROM reset. Consume that hidden reset first, then
    // restore only emulator state/RAM/TIA input stream before replaying the
    // visible reset. Leaving frame_state/framebuffer history intact mirrors the
    // separate ALE m_screen/processScreen snapshot path used for reset().
    const state state_backup = wrap.cached_states_ptr[0];
    const std::vector<uint8_t> ram_backup(
        wrap.cached_ram_ptr,
        wrap.cached_ram_ptr + wrap.cart.ram_size());
    const std::vector<uint32_t> tia_backup(
        wrap.cached_tia_update_ptr,
        wrap.cached_tia_update_ptr + ENV_UPDATE_SIZE);

    std::mt19937 rng(ale_seed);
    uint64_t sticky_draw_count = 0;
    perform_exact_doubledunk_reset_sequence<Environment>(wrap, rng, sticky_draw_count, false);

    wrap.cached_states_ptr[0] = state_backup;
    std::copy(ram_backup.begin(), ram_backup.end(), wrap.cached_ram_ptr);
    std::copy(tia_backup.begin(), tia_backup.end(), wrap.cached_tia_update_ptr);

    perform_exact_doubledunk_reset_sequence<Environment>(wrap, rng, sticky_draw_count, true);
    wrap.reset_sticky_rng_draws = sticky_draw_count;
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
reset(ExecutionPolicy&& policy,
      Wrapper& wrap,
      uint32_t* seedBuffer,
      const uint32_t* aleSeedBuffer)
{
    agency::vector<uint32_t, agency::allocator<uint32_t>> rand_temp_buffer;
    wrap.reset_sticky_rng_draws = 0;

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
    const bool require_frame_complete_reset =
        requires_frame_complete_reset_state(wrap.cart);
    const bool use_exact_doubledunk_reset =
        wrap.ale_reset_semantics &&
        (wrap.cart.game_id() == games::GAME_DOUBLE_DUNK) &&
        (wrap.reset_repeat_action_probability > 0.0f) &&
        (aleSeedBuffer != nullptr);

    if(use_exact_doubledunk_reset)
    {
        perform_exact_doubledunk_reset<Environment>(wrap, aleSeedBuffer[0]);
    }
    else while((boot_frames < ENV_BASE_FRAMES) ||
               !wrap.cached_states_ptr[0].tiaFlags[FLAG_ALE_STARTED] ||
               (require_frame_complete_reset &&
                wrap.cached_states_ptr[0].tiaFlags[FLAG_TIA_PARTIAL]))
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
                            wrap.cached_previous_frame_ptr,
                            true);

        if(!wrap.cached_states_ptr[0].tiaFlags[FLAG_ALE_STARTED] &&
           (wrap.cached_states_ptr[0].bootPhase == BOOT_DONE))
        {
            wrap.cached_states_ptr[0].tiaFlags.set(FLAG_ALE_STARTED);
        }
        ++boot_frames;
        CULE_ASSERT(boot_frames <= (ENV_BASE_FRAMES + MAX_EXTRA_BOOT_FRAMES),
                    "ALE reset did not reach the required started state for "
                        << wrap.cart.game_name() << " within "
                        << (ENV_BASE_FRAMES + MAX_EXTRA_BOOT_FRAMES)
                        << " frames");
    }

    if(wrap.ale_reset_semantics)
    {
        apply_ale_v5_post_reset_normalization(wrap.cart, wrap.cached_states_ptr[0]);
    }
    if(!use_exact_doubledunk_reset)
    {
        capture_cached_reset_screen<Environment>(wrap, 0);
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
                            wrap.cached_previous_frame_ptr + (i * 300 * SCREEN_WIDTH),
                            true);
        capture_cached_reset_screen<Environment>(wrap, i);
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
        if(wrap.reset_screen_ptr != nullptr)
        {
            const uint8_t* reset_screen =
                wrap.cached_reset_screen_ptr != nullptr
                    ? wrap.cached_reset_screen_ptr
                    : (wrap.cached_frame_states_ptr[index].frameBufferIndex == 0
                           ? wrap.cached_frame_ptr
                           : wrap.cached_previous_frame_ptr);
            std::copy(reset_screen + (index * 300 * SCREEN_WIDTH),
                      reset_screen + ((index + 1) * 300 * SCREEN_WIDTH),
                      wrap.reset_screen_ptr + (i * 300 * SCREEN_WIDTH));
        }
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
                        wrap.reset_screen_ptr,
                        wrap.noop_reset_steps,
                        wrap.states_ptr,
                        wrap.cached_states_ptr,
                        wrap.cached_ram_ptr,
                        wrap.cached_tia_update_ptr,
                        wrap.frame_states_ptr,
                        wrap.cached_frame_states_ptr,
                        wrap.cached_frame_ptr,
                        wrap.cached_previous_frame_ptr,
                        wrap.cached_reset_screen_ptr,
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
           uint8_t* frameBuffer,
           const bool render_frame)
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
                        wrap.previous_frame_ptr,
                        render_frame);
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
generate_reset_screen_frames(ExecutionPolicy&& policy,
                             Wrapper& wrap,
                             const bool rescale,
                             const size_t num_channels,
                             uint8_t* imageBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        generate_screen_frame_functor{},
                        num_channels,
                        wrap.cart.screen_height(),
                        rescale,
                        wrap.reset_screen_ptr,
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
