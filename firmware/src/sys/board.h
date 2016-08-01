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

#pragma once

#define STM32_HSECLK            8000000

#define STM32F10X_HD

/*
 * GPIO
 */
// Misc
//#define GPIO_PORT_PERIPH_RESET  GPIOC
//#define GPIO_PIN_PERIPH_RESET   10

// LED
#define GPIO_PORT_LED_STATUS    GPIOC
#define GPIO_PIN_LED_STATUS     13

#define GPIO_PORT_LED_CAN1      GPIOC
#define GPIO_PIN_LED_CAN1       15

#define GPIO_PORT_LED_CAN2      GPIOC
#define GPIO_PIN_LED_CAN2       14

/*
 * I/O ports initial setup, this configuration is established soon after reset
 * in the initialization code.
 *
 * The digits have the following meaning:
 *   0 - Analog input.
 *   1 - Push Pull output 10MHz.
 *   2 - Push Pull output 2MHz.
 *   3 - Push Pull output 50MHz.
 *   4 - Digital input.
 *   5 - Open Drain output 10MHz.
 *   6 - Open Drain output 2MHz.
 *   7 - Open Drain output 50MHz.
 *   8 - Digital input with PullUp or PullDown resistor depending on ODR.
 *   9 - Alternate Push Pull output 10MHz.
 *   A - Alternate Push Pull output 2MHz.
 *   B - Alternate Push Pull output 50MHz.
 *   C - Reserved.
 *   D - Alternate Open Drain output 10MHz.
 *   E - Alternate Open Drain output 2MHz.
 *   F - Alternate Open Drain output 50MHz.
 * Please refer to the STM32 Reference Manual for details.
 */

#define VAL_GPIOACRL            0x88888A88                      // 7..0
#define VAL_GPIOACRH            0x88800888                      // 15..8
#define VAL_GPIOAODR            0x00000000

#define VAL_GPIOBCRL            0xEE222888
#define VAL_GPIOBCRH            0x88488A48
#define VAL_GPIOBODR            ((1 << 7) | (1 << 6))

#define VAL_GPIOCCRL            0x88888888
#define VAL_GPIOCCRH            0x33388288
#define VAL_GPIOCODR            ((1<<14)|(1<<15))

#define VAL_GPIODCRL            0x88888888
#define VAL_GPIODCRH            0x88888888
#define VAL_GPIODODR            0x00000000

#define VAL_GPIOECRL            0x88888888
#define VAL_GPIOECRH            0x88888888
#define VAL_GPIOEODR            0x00000000

#if !defined(_FROM_ASM_)
#ifdef __cplusplus
extern "C" {
#endif
    void boardInit(void);
#ifdef __cplusplus
}
#endif
#endif /* _FROM_ASM_ */
