#pragma once

#include <cule/config.hpp>

#include <cule/atari/accessors.hpp>
#include <cule/atari/actions.hpp>
#include <cule/atari/ale.hpp>
#include <cule/atari/m6532.hpp>
#include <cule/atari/ram.hpp>
#include <cule/atari/tia.hpp>

#include <agency/agency.hpp>

#include <iomanip>
#include <random>

namespace cule
{
namespace atari
{

template<typename, typename> struct interrupt;
template<typename, typename, typename> struct mmc;
template<typename, typename, typename> struct m6502;
template<int32_t> struct rom_accessor;
template<typename> struct stack;

template<int32_t ROM_FORMAT>
struct environment
{

static constexpr int32_t ROM_FORMAT_VALUE = ROM_FORMAT;
static constexpr uint32_t RAM_BYTES_PER_ENV = (ROM_FORMAT == ROM_F8SC) ? 256 : 128;
static constexpr uint32_t RAM_WORDS_PER_ENV = RAM_BYTES_PER_ENV / sizeof(uint32_t);

using Accessor_t = rom_accessor<ROM_FORMAT>;
using ALE_t = ale;
using Controller_t = controller;

using M6532_t = m6532<Controller_t>;
using TIA_t = tia<M6532_t, Controller_t>;

using MMC_t = mmc<Accessor_t, M6532_t, TIA_t>;
using Stack_t = stack<MMC_t>;
using Interrupt_t = interrupt<MMC_t, Stack_t>;
using M6502_t = m6502<MMC_t, Stack_t, Interrupt_t>;

template<typename State_t>
static
CULE_ANNOTATION
void increment(State_t& s)
{
    INC_FIELD(s.frameData, FIELD_FRAME_NUMBER);
}

template<typename State_t>
static
CULE_ANNOTATION
void incrementBootProgress(State_t& s)
{
    ++s.bootProgress;
}

template<typename State_t>
static
CULE_ANNOTATION
void setFrameNumber(State_t& s, const int frame_number)
{
    UPDATE_FIELD(s.frameData, FIELD_FRAME_NUMBER, frame_number);
}

template<typename State_t>
static
CULE_ANNOTATION
void setBootProgress(State_t& s, const uint16_t boot_progress)
{
    s.bootProgress = boot_progress;
}

template<typename State_t>
static
CULE_ANNOTATION
void setBootPhase(State_t& s, const BOOT_PHASE boot_phase)
{
    s.bootPhase = boot_phase;
}

template<typename State_t>
static
CULE_ANNOTATION
int getFrameNumber(State_t& s)
{
    return SELECT_FIELD(s.frameData, FIELD_FRAME_NUMBER);
}

template<typename State_t>
static
CULE_ANNOTATION
uint16_t getBootProgress(State_t& s)
{
    return s.bootProgress;
}

template<typename State_t>
static
CULE_ANNOTATION
BOOT_PHASE getBootPhase(State_t& s)
{
    return BOOT_PHASE(s.bootPhase);
}

template<typename State_t>
static
CULE_ANNOTATION
void setStartNumber(State_t& s, int num_actions)
{
    UPDATE_FIELD(s.frameData, FIELD_START_NUMBER, num_actions);
}

template<typename State_t>
static
CULE_ANNOTATION
int getStartNumber(State_t& s)
{
    return SELECT_FIELD(s.frameData, FIELD_START_NUMBER);
}

template<typename State_t>
static
CULE_ANNOTATION
void setStartAction(State_t& s, const Action& starting_action)
{
    UPDATE_FIELD(s.frameData, FIELD_START_ACTION, starting_action);
}

template<typename State_t>
static
CULE_ANNOTATION
Action getStartAction(State_t& s)
{
    return Action(SELECT_FIELD(s.frameData, FIELD_START_ACTION));
}

template<typename State_t>
static
CULE_ANNOTATION
void setAuxState(State_t& s, const uint8_t value)
{
    UPDATE_FIELD(s.frameData, FIELD_ALE_AUX, value);
}

template<typename State_t>
static
CULE_ANNOTATION
uint8_t getAuxState(State_t& s)
{
    return SELECT_FIELD(s.frameData, FIELD_ALE_AUX);
}

/** Resets the system to its start state. */
template<typename State_t>
static
CULE_ANNOTATION
void reset(State_t& s)
{
    // Reset ALE
    ALE_t::reset(s);

    // Reset the paddles
    Controller_t::reset(s);

    // Stella resets hardware devices in system attach order before the CPU:
    // M6532, TIA, cartridge, then the processor.
    M6532_t::reset(s);

    TIA_t::reset(s);

    // Reset cartridge power-on state such as active banks and SC RAM.
    Accessor_t::reset(s);

    // Reset the processor after device reset so the reset vector is read from
    // the finalized cartridge/bus state.
    M6502_t::reset(s);

    // Reset the frame number
    setFrameNumber(s, 0);
    setBootProgress(s, 0);
    setBootPhase(s, BOOT_NOOP);
}

/** Actually emulates the emulator for a given number of steps. */
template<typename State_t>
static
CULE_ANNOTATION
bool emulate(State_t& s)
{
    if(!s.tiaFlags[FLAG_TIA_PARTIAL])
    {
        TIA_t::startFrame(s);
    }

    s.tiaFlags.set(FLAG_TIA_PARTIAL);

    // update paddle position at every step
    Controller_t::applyAction(s);
    M6502_t::run(s);

    if(!s.tiaFlags[FLAG_TIA_PARTIAL])
    {
        TIA_t::finishFrame(s);
    }

    // ALE evaluates terminal/lives state after every mediaSource().update()
    // call, not only after a completed TIA frame.
    ALE_t::setTerminal(s);

    return !s.tiaFlags[FLAG_TIA_PARTIAL];
}

template<typename State_t>
static
CULE_ANNOTATION
void setSelectPressed(State_t& s, const bool select_pressed)
{
    s.sysFlags.template change<FLAG_SW_SELECT_OFF>(!select_pressed);
}

template<typename State_t>
static
CULE_ANNOTATION
bool configure_mode_boot_step(State_t& s,
                              Action& action,
                              bool& select_pressed,
                              bool& count_public_frame)
{
    const games::GAME_TYPE game_id = ALE_t::get_id(s);
    const uint16_t progress = getBootProgress(s);

    select_pressed = false;
    count_public_frame = false;
    action = ACTION_NOOP;

    switch(game_id)
    {
    case games::GAME_BATTLE_ZONE:
    {
        if(progress == 0)
        {
            if(cule::atari::ram::read(s.ram, 0xA1) == 0x01)
            {
                setBootProgress(s, 10);
                action = ACTION_RESET;
            }
            else
            {
                setBootProgress(s, 1);
                select_pressed = true;
            }
        }
        else if(progress <= 2)
        {
            select_pressed = true;
        }
        else if(progress == 3)
        {
            count_public_frame = true;
        }
        else
        {
            action = ACTION_RESET;
        }
        return true;
    }
    case games::GAME_NAME_THIS_GAME:
    {
        if(progress == 0)
        {
            if(cule::atari::ram::read(s.ram, 0xDE) == 0x08)
            {
                setBootProgress(s, 10);
                action = ACTION_RESET;
            }
            else
            {
                setBootProgress(s, 1);
                select_pressed = true;
            }
        }
        else if(progress <= 2)
        {
            select_pressed = true;
        }
        else if(progress == 3)
        {
            count_public_frame = true;
        }
        else
        {
            action = ACTION_RESET;
        }
        return true;
    }
    case games::GAME_GRAVITAR:
    {
        if(progress == 0)
        {
            if(cule::atari::ram::read(s.ram, 0x80) == 0x00)
            {
                setBootProgress(s, 20);
                action = ACTION_RESET;
            }
            else
            {
                setBootProgress(s, 1);
                select_pressed = true;
            }
        }
        else if(progress <= 10)
        {
            select_pressed = true;
        }
        else if(progress == 11)
        {
            count_public_frame = true;
        }
        else
        {
            action = ACTION_RESET;
        }
        return true;
    }
    case games::GAME_MS_PACMAN:
    {
        if(progress == 0)
        {
            if((cule::atari::ram::read(s.ram, 0x99) == 0x01) &&
               (cule::atari::ram::read(s.ram, 0xA1) == 0x01))
            {
                setBootProgress(s, 20);
                action = ACTION_RESET;
            }
            else
            {
                setBootProgress(s, 1);
                select_pressed = true;
            }
        }
        else if(progress <= 10)
        {
            select_pressed = true;
        }
        else if(progress == 11)
        {
            count_public_frame = true;
        }
        else
        {
            action = ACTION_RESET;
        }
        return true;
    }
    case games::GAME_YARS_REVENGE:
    {
        if(progress <= 1)
        {
            select_pressed = true;
        }
        else if(progress == 2)
        {
            count_public_frame = true;
        }
        else if(progress == 3)
        {
            if(cule::atari::ram::read(s.ram, 0xE3) == 0x00)
            {
                setBootProgress(s, 20);
                action = ACTION_RESET;
            }
            else
            {
                setBootProgress(s, 4);
                select_pressed = true;
            }
        }
        else if(progress == 4)
        {
            select_pressed = true;
        }
        else if(progress == 5)
        {
            count_public_frame = true;
        }
        else
        {
            action = ACTION_RESET;
        }
        return true;
    }
    case games::GAME_DOUBLE_DUNK:
    {
        switch(progress)
        {
        case 0:
            select_pressed = true;
            break;
        case 1:
        case 3:
        case 5:
        case 7:
        case 9:
        case 11:
            count_public_frame = true;
            break;
        case 2:
        case 4:
        case 6:
        case 8:
        case 10:
            action = ACTION_DOWN;
            count_public_frame = true;
            break;
        case 12:
        case 13:
        case 14:
        case 15:
            action = ACTION_RESET;
            break;
        case 16:
            action = ACTION_UPFIRE;
            count_public_frame = true;
            break;
        case 17:
            count_public_frame = true;
            break;
        default:
            return false;
        }
        return true;
    }
    default:
        return false;
    }
}

template<typename State_t>
static
CULE_ANNOTATION
void advance_mode_boot_step(State_t& s,
                            const uint16_t progress)
{
    const games::GAME_TYPE game_id = ALE_t::get_id(s);

    switch(game_id)
    {
    case games::GAME_BATTLE_ZONE:
    case games::GAME_NAME_THIS_GAME:
    {
        if((progress == 1) || (progress == 2))
        {
            incrementBootProgress(s);
        }
        else if(progress == 3)
        {
            setBootProgress(s, 0);
        }
        else if(progress >= 10)
        {
            incrementBootProgress(s);
            if(getBootProgress(s) == 14)
            {
                setBootProgress(s, 0);
                setBootPhase(s, BOOT_RESET2);
            }
        }
        break;
    }
    case games::GAME_GRAVITAR:
    case games::GAME_MS_PACMAN:
    {
        if(progress >= 1 && progress <= 10)
        {
            incrementBootProgress(s);
        }
        else if(progress == 11)
        {
            setBootProgress(s, 0);
        }
        else if(progress >= 20)
        {
            incrementBootProgress(s);
            if(getBootProgress(s) == 24)
            {
                setBootProgress(s, 0);
                setBootPhase(s, BOOT_RESET2);
            }
        }
        break;
    }
    case games::GAME_YARS_REVENGE:
    {
        if(progress <= 1)
        {
            incrementBootProgress(s);
        }
        else if(progress == 2)
        {
            setBootProgress(s, 3);
        }
        else if(progress == 4)
        {
            setBootProgress(s, 5);
        }
        else if(progress == 5)
        {
            setBootProgress(s, 3);
        }
        else if(progress >= 20)
        {
            incrementBootProgress(s);
            if(getBootProgress(s) == 24)
            {
                setBootProgress(s, 0);
                setBootPhase(s, BOOT_RESET2);
            }
        }
        break;
    }
    case games::GAME_DOUBLE_DUNK:
    {
        incrementBootProgress(s);
        if(getBootProgress(s) == 18)
        {
            setBootProgress(s, 0);
            setBootPhase(s, BOOT_RESET2);
        }
        break;
    }
    default:
    {
        setBootProgress(s, 0);
        setBootPhase(s, BOOT_RESET2);
        break;
    }
    }
}

/** This applies an action exactly one time step. Helper function to act(). */
template<typename State_t>
static
CULE_ANNOTATION
void act(State_t& s, const Action& player_a_action, const Action& player_b_action = ACTION_NOOP)
{
    if (ALE_t::isTerminal(s) && ALE_t::isStarted(s))
    {
        reset(s);
    }

    // Convert illegal actions into NOOPs; actions such as reset are always illegal
    ALE_t::noopIllegalActions(s);

    const bool booting = !ALE_t::isStarted(s);
    bool count_public_frame = false;
    bool select_pressed = false;
    Action action = player_a_action;
    Action player_b_internal_action = player_b_action;
    BOOT_PHASE boot_phase = getBootPhase(s);
    uint16_t boot_progress = getBootProgress(s);

    if(booting)
    {
        switch(boot_phase)
        {
        case BOOT_NOOP:
            action = ACTION_NOOP;
            break;
        case BOOT_RESET1:
        case BOOT_RESET2:
            action = ACTION_RESET;
            break;
        case BOOT_START_ACTIONS:
            action = getStartAction(s);
            break;
        case BOOT_MODE_SETUP:
        {
            if(!configure_mode_boot_step(s, action, select_pressed, count_public_frame))
            {
                setBootPhase(s, BOOT_RESET2);
                boot_phase = BOOT_RESET2;
                action = ACTION_RESET;
            }
            break;
        }
        case BOOT_DONE:
            break;
        }
        boot_phase = getBootPhase(s);
        boot_progress = getBootProgress(s);
        player_b_internal_action = ACTION_NOOP;
    }

    setSelectPressed(s, select_pressed);
    if(booting)
    {
        Controller_t::set_actions(s, action, player_b_internal_action);
    }
    else
    {
        Controller_t::set_actions(s, action, player_b_internal_action);
    }

    const bool finished_frame = emulate(s);
    if(booting)
    {
        if(count_public_frame)
        {
            increment(s);
        }

        switch(boot_phase)
        {
        case BOOT_NOOP:
            incrementBootProgress(s);
            if(getBootProgress(s) == ENV_NOOP_FRAMES)
            {
                setBootProgress(s, 0);
                setBootPhase(s, BOOT_RESET1);
            }
            break;
        case BOOT_RESET1:
            incrementBootProgress(s);
            if(getBootProgress(s) == ENV_RESET_FRAMES / 2)
            {
                ALE_t::reset(s);
                setBootProgress(s, 0);
                setBootPhase(s, BOOT_MODE_SETUP);
            }
            break;
        case BOOT_MODE_SETUP:
            advance_mode_boot_step(s, boot_progress);
            break;
        case BOOT_RESET2:
            incrementBootProgress(s);
            if(getBootProgress(s) == ENV_RESET_FRAMES / 2)
            {
                setBootProgress(s, 0);
                setBootPhase(s, getStartNumber(s) > 0 ? BOOT_START_ACTIONS : BOOT_DONE);
            }
            break;
        case BOOT_START_ACTIONS:
            incrementBootProgress(s);
            if(getBootProgress(s) == getStartNumber(s))
            {
                setBootProgress(s, 0);
                setBootPhase(s, BOOT_DONE);
            }
            break;
        case BOOT_DONE:
            break;
        }

        if(getBootPhase(s) == BOOT_DONE)
        {
            s.tiaFlags.set(FLAG_ALE_STARTED);
        }
    }
    else
    {
        increment(s);
    }
}

}; // end class environment

} // end namespace atari
} // end namespace cule
