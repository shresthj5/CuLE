#pragma once

#include <cule/macros.hpp>

#include <cule/atari/flags.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/tables.hpp>

namespace cule
{
namespace atari
{

CULE_ANNOTATION
void refresh_frame_state_masks(frame_state& fs)
{
    const uint8_t nusiz0_mode = SELECT_FIELD(fs.PF, FIELD_NUSIZ0_MODE);
    const uint8_t nusiz0_size = SELECT_FIELD(fs.PF, FIELD_NUSIZ0_SIZE);
    const uint8_t nusiz1_mode = SELECT_FIELD(fs.PF, FIELD_NUSIZ1_MODE);
    const uint8_t nusiz1_size = SELECT_FIELD(fs.PF, FIELD_NUSIZ1_SIZE);
    const uint8_t ctrlpf = SELECT_FIELD(fs.PF, FIELD_CTRLPF);

    const uint8_t posp0 = SELECT_FIELD(fs.POS, FIELD_POSP0);
    const uint8_t posp1 = SELECT_FIELD(fs.POS, FIELD_POSP1);
    const uint8_t posm0 = SELECT_FIELD(fs.POS, FIELD_POSM0);
    const uint8_t posm1 = SELECT_FIELD(fs.POS, FIELD_POSM1);
    const uint8_t posbl = SELECT_FIELD(fs.HM, FIELD_POSBL);

    fs.CurrentPFMask = &playfield_accessor(fs.tiaFlags[FLAG_TIA_CTRLPF], 0);
    fs.CurrentP0Mask = &player_mask_accessor(posp0 & 0x03, 0, nusiz0_mode, 160 - (posp0 & 0xFC));
    fs.CurrentP1Mask = &player_mask_accessor(posp1 & 0x03, 0, nusiz1_mode, 160 - (posp1 & 0xFC));
    fs.CurrentM0Mask = &missle_accessor(posm0 & 0x03, nusiz0_mode, nusiz0_size, 160 - (posm0 & 0xFC));
    fs.CurrentM1Mask = &missle_accessor(posm1 & 0x03, nusiz1_mode, nusiz1_size, 160 - (posm1 & 0xFC));
    fs.CurrentBLMask = &ball_accessor(posbl & 0x03, ctrlpf, 160 - (posbl & 0xFC));
}

} // namespace atari
} // namespace cule
