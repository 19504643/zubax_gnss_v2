/*
 * Copyright (C) 2014-2015  Zubax Robotics  <info@zubax.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Pavel Kirienko <pavel.kirienko@zubax.com>
 */

#include <uavcan_stm32/bxcan.hpp>
#include "bootloader_interface.hpp"

namespace bootloader_interface
{
/**
 * This is the Brickproof Bootloader's app descriptor.
 * Details: https://github.com/PX4/Firmware/tree/nuttx_next/src/drivers/bootloaders/src/uavcan
 */
static const volatile struct __attribute__((packed))
{
    std::uint8_t signature[8] = {'A','P','D','e','s','c','0','0'};
    std::uint64_t image_crc = 0;
    std::uint32_t image_size = 0;
    std::uint32_t vcs_commit = GIT_HASH;
    std::uint8_t major_version = FW_VERSION_MAJOR;
    std::uint8_t minor_version = FW_VERSION_MINOR;
    std::uint8_t reserved[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
} _app_descriptor __attribute__((section(".app_descriptor")));


static constexpr auto BootloaderSignature = 0xB0A0424CU;
static constexpr auto AppSignature        = 0xB0A04150U;


struct SharedData
{
    std::uint32_t can_bus_bit_rate_bps = 0;
    uavcan::NodeID uavcan_node_id;
} static shared_data;


class CrcComputer : public uavcan::DataTypeSignatureCRC
{
public:
    void add(std::uint32_t value)
    {
        uavcan::DataTypeSignatureCRC::add(reinterpret_cast<const std::uint8_t*>(&value), 4);
    }
};


void init()
{
    const auto signature = uavcan_stm32::bxcan::Can[0]->FilterRegister[3].FR1;
    uavcan_stm32::bxcan::Can[0]->FilterRegister[3].FR1 = 0;                     // Invalidate to prevent deja vu

    const auto bus_speed = uavcan_stm32::bxcan::Can[0]->FilterRegister[3].FR2;
    const auto node_id   = uavcan_stm32::bxcan::Can[0]->FilterRegister[4].FR1;

    union
    {
        std::uint64_t u64;
        std::uint32_t u32[2];
    } crc;
    crc.u32[0] = uavcan_stm32::bxcan::Can[0]->FilterRegister[2].FR2;
    crc.u32[1] = uavcan_stm32::bxcan::Can[0]->FilterRegister[2].FR1;

    CrcComputer computer;
    computer.add(signature);
    computer.add(bus_speed);
    computer.add(node_id);

    const auto signature_match = signature == BootloaderSignature;
    const auto crc_match = crc.u64 == computer.get();
    const auto valid_params = (bus_speed > 0) && (node_id <= uavcan::NodeID::Max);

    if (signature_match && crc_match && valid_params)
    {
        shared_data.can_bus_bit_rate_bps = bus_speed;
        shared_data.uavcan_node_id = static_cast<std::uint8_t>(node_id);
    }
}

uavcan::protocol::SoftwareVersion makeUavcanSoftwareVersionStruct()
{
    uavcan::protocol::SoftwareVersion x;

    x.major = _app_descriptor.major_version;
    x.minor = _app_descriptor.minor_version;

    x.image_crc = _app_descriptor.image_crc;
    x.vcs_commit = _app_descriptor.vcs_commit;

    x.optional_field_flags = x.OPTIONAL_FIELD_FLAG_VCS_COMMIT | x.OPTIONAL_FIELD_FLAG_IMAGE_CRC;

    return x;
}

std::uint32_t getInheritedCanBusBitRate()
{
    return shared_data.can_bus_bit_rate_bps;
}

uavcan::NodeID getInheritedNodeID()
{
    return shared_data.uavcan_node_id;
}

void passParametersToBootloader(std::uint32_t can_bus_bit_rate, uavcan::NodeID node_id)
{
    uavcan_stm32::bxcan::Can[0]->FilterRegister[3].FR1 = AppSignature;
    uavcan_stm32::bxcan::Can[0]->FilterRegister[3].FR2 = can_bus_bit_rate;
    uavcan_stm32::bxcan::Can[0]->FilterRegister[4].FR1 = node_id.get();

    CrcComputer computer;
    computer.add(uavcan_stm32::bxcan::Can[0]->FilterRegister[3].FR1);
    computer.add(uavcan_stm32::bxcan::Can[0]->FilterRegister[3].FR2);
    computer.add(uavcan_stm32::bxcan::Can[0]->FilterRegister[4].FR1);

    union
    {
        std::uint64_t u64;
        std::uint32_t u32[2];
    } crc;
    crc.u64 = computer.get();
    uavcan_stm32::bxcan::Can[0]->FilterRegister[2].FR2 = crc.u32[0];
    uavcan_stm32::bxcan::Can[0]->FilterRegister[2].FR1 = crc.u32[1];
}

}
