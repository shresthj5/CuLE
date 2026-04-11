#pragma once

#include <cule/config.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/state.hpp>

namespace cule
{
namespace atari
{
namespace games
{
namespace qbert
{

enum : uint8_t
{
    QBERT_LIVES_ACTIVE = 0,
    QBERT_LIVES_PREV_NEGATIVE_ONE = 1,
};

template<typename State>
CULE_ANNOTATION
 void reset(State& s)
{
    s.m_reward     = 0;
    s.m_score      = 0;
    s.tiaFlags.clear(FLAG_ALE_TERMINAL);
    s.m_lives 	   = 4;
    s.m_last_lives = 2;
}

template<typename State>
CULE_ANNOTATION
 void step(State& s)
{
    using cule::atari::games::getDecimalScore;
    using cule::atari::ram::read;

    // update terminal status
    int lives_value = ram::read(s, 0x88);
    // Lives start at 2 (4 lives, 3 displayed) and go down to 0xFE (death)
    // Alternatively we can die and reset within one frame; we catch this case
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>((lives_value == 0xFE) || ((lives_value == 0x02) && (s.m_last_lives == -1)));

    // Convert char into a signed integer
    int livesAsChar = static_cast<char>(lives_value);

    if ((s.m_last_lives - 1) == livesAsChar) s.m_lives--;
    s.m_last_lives = livesAsChar;

    // update the reward
    // Ignore reward if reset the game via the fire button; otherwise the agent
    //  gets a big negative reward on its last step
    if (!s.tiaFlags[FLAG_ALE_TERMINAL])
    {
        int score = getDecimalScore(s, 0xDB, 0xDA, 0xD9);
        int reward = score - s.m_score;
        s.m_reward = reward;
        s.m_score = score;
    }
    else
    {
        s.m_reward = 0;
    }
}

CULE_ANNOTATION
 bool isMinimal(const Action &a)
{
    switch (a)
    {
    case ACTION_NOOP:
    case ACTION_FIRE:
    case ACTION_UP:
    case ACTION_RIGHT:
    case ACTION_LEFT:
    case ACTION_DOWN:
        return true;
    default:
        return false;
    }
}

template<typename State>
CULE_ANNOTATION
int32_t lives(State& s)
{
    const int32_t lives_value = cule::atari::ram::read(s.ram, 0x88);
    if(lives_value == 0xFE)
    {
        return 0;
    }

    // ALE's internal QBert lives counter stays at 4 during the initial
    // reset boundary even while the RAM byte is still zero.
    const int32_t frame_number = SELECT_FIELD(s.frameData, FIELD_FRAME_NUMBER);
    if((lives_value == 0x00) &&
       (frame_number <= 1) &&
       (cule::atari::games::getDecimalScore(s, 0xDB, 0xDA, 0xD9) == 0))
    {
        return 4;
    }

    return static_cast<int8_t>(lives_value) + 2;
}

template<typename State>
CULE_ANNOTATION
 void setTerminal(State& s)
{
    // update terminal status
    const int lives_value = cule::atari::ram::read(s.ram, 0x88);
    // Lives start at 2 (4 lives, 3 displayed) and go down to 0xFE (death)
    // Alternatively we can die and reset within one frame; we catch this case
    const uint8_t aux_state = SELECT_FIELD(s.frameData, FIELD_ALE_AUX);
    s.tiaFlags.template change<FLAG_ALE_TERMINAL>(
        (lives_value == 0xFE) ||
        ((lives_value == 0x02) && (aux_state == QBERT_LIVES_PREV_NEGATIVE_ONE)));

    uint8_t next_aux_state = QBERT_LIVES_ACTIVE;
    const int livesAsChar = static_cast<int8_t>(lives_value);
    if(livesAsChar == -1)
    {
        next_aux_state = QBERT_LIVES_PREV_NEGATIVE_ONE;
    }

    UPDATE_FIELD(s.frameData, FIELD_ALE_AUX, next_aux_state);
}

template<typename State>
CULE_ANNOTATION
int32_t score(State& s)
{
    return cule::atari::games::getDecimalScore(s, 0xDB, 0xDA, 0xD9);
}

template<typename State>
CULE_ANNOTATION
int32_t reward(State& s)
{
    int32_t m_reward = 0;

    // update the reward
    // Ignore reward if reset the game via the fire button; otherwise the agent
    //  gets a big negative reward on its last step
    if (!s.tiaFlags[FLAG_ALE_TERMINAL])
    {
        m_reward = score(s) - s.score;
    }

    return m_reward;
}

} // end namespace qbert
} // end namespace games
} // end namespace atari
} // end namespace cule
