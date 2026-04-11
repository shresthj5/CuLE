#include <array>
#include <cstdint>
#include <iostream>
#include <string>

#include <cule/cule.hpp>
#include <cule/atari/prng.hpp>

namespace
{
using Environment = cule::atari::environment<cule::atari::ROM_4K>;
using Accessor = typename Environment::Accessor_t;
using ALE = typename Environment::ALE_t;
using Controller = typename Environment::Controller_t;
using State = cule::atari::state;

constexpr uint32_t kSystemRandomSeed = 4753849U;

struct TraceState
{
    State state{};
    std::array<uint32_t, Environment::RAM_WORDS_PER_ENV> ram_words{};
};

const char* boot_phase_name(const uint8_t phase)
{
    using namespace cule::atari;
    switch(BOOT_PHASE(phase))
    {
    case BOOT_NOOP:
        return "BOOT_NOOP";
    case BOOT_RESET1:
        return "BOOT_RESET1";
    case BOOT_MODE_SETUP:
        return "BOOT_MODE_SETUP";
    case BOOT_RESET2:
        return "BOOT_RESET2";
    case BOOT_START_ACTIONS:
        return "BOOT_START_ACTIONS";
    case BOOT_DONE:
        return "BOOT_DONE";
    default:
        return "BOOT_UNKNOWN";
    }
}

TraceState initialize_state(const cule::atari::rom& cart)
{
    TraceState trace_state{};
    State& s = trace_state.state;

    s.ram = trace_state.ram_words.data();
    s.rom = cart.data();
    s.tia_update_buffer = nullptr;

    uint32_t rand_state = 0U;
    cule::atari::prng gen(rand_state);
    gen.initialize(kSystemRandomSeed);
    s.rand = gen.sample();

    cule::atari::detail::initialize_power_on_riot_ram(s);
    Accessor::initialize(s);

    Controller::set_flags(
        s,
        cart.use_paddles(),
        cart.swap_paddles() || cart.swap_ports(),
        cart.player_left_difficulty_B(),
        cart.player_right_difficulty_B());

    ALE::set_id(s, cart.game_id());
    s.tiaFlags.template change<cule::atari::FLAG_TIA_IS_NTSC>(cart.is_ntsc());
    s.tiaFlags.template change<cule::atari::FLAG_TIA_HMOVE_ALLOW>(cart.allow_hmove_blanks());
    s.tiaFlags.template change<cule::atari::FLAG_TIA_Y_SHIFT>(cart.game_id() != cule::atari::games::GAME_UP_N_DOWN);
    s.displayYStart = cart.screen_y_start();
    s.displayHeight = cart.screen_height();

    s.tiaFlags.set(cule::atari::FLAG_ALE_TERMINAL);
    s.tiaFlags.set(cule::atari::FLAG_ALE_STARTED);
    s.tiaFlags.clear(cule::atari::FLAG_ALE_LOST_LIFE);

    Environment::setStartNumber(s, ALE::getStartingActionCount(cart.game_id()));
    Environment::setStartAction(s, ALE::getStartingAction(cart.game_id()));
    Environment::setAuxState(s, 0);
    Environment::setBootProgress(s, 0);
    Environment::setBootPhase(s, cule::atari::BOOT_NOOP);

    return trace_state;
}

void print_state(const int step, State& s)
{
    const auto* ram_bytes = reinterpret_cast<const uint8_t*>(s.ram);
    std::cout
        << "step=" << step
        << " phase=" << boot_phase_name(s.bootPhase)
        << " progress=" << s.bootProgress
        << " started=" << int(s.tiaFlags[cule::atari::FLAG_ALE_STARTED])
        << " partial=" << int(s.tiaFlags[cule::atari::FLAG_TIA_PARTIAL])
        << " frame=" << Environment::getFrameNumber(s)
        << " cpuCycles=" << s.cpuCycles
        << " PC=" << static_cast<unsigned>(s.PC)
        << " A=" << int(s.A)
        << " X=" << int(s.X)
        << " Y=" << int(s.Y)
        << " timer=" << int(SELECT_FIELD(s.riotData, cule::atari::FIELD_RIOT_TIMER))
        << " cyclesWhenInterruptReset=" << s.cyclesWhenInterruptReset
        << " ram57=" << int(ram_bytes[57])
        << " ram102=" << int(ram_bytes[102])
        << "\n";
}
} // namespace

int main(int argc, char** argv)
{
    const int steps = (argc > 1) ? std::stoi(argv[1]) : 90;
    const std::string rom_path =
        (argc > 2)
            ? std::string(argv[2])
            : std::string("/home/shres/projects/ICML/CuLE/.venv/lib/python3.12/site-packages/ale_py/roms/assault.bin");
    const std::string game_name = (argc > 3) ? std::string(argv[3]) : std::string("assault");
    cule::atari::rom cart(rom_path, game_name);
    TraceState trace_state = initialize_state(cart);
    State& s = trace_state.state;

    print_state(-1, s);
    for(int step = 0; step < steps; ++step)
    {
        Environment::act(s, cule::atari::ACTION_NOOP, cule::atari::ACTION_NOOP);
        print_state(step, s);
    }

    return 0;
}
