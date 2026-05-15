#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "gf2x.h"

/**
 * @brief Modular reduction of a degree < 2*n polynomial mod (X^n - 1).
 *
 * Folds the high half of the full product back into the low half
 * and masks any excess bits in the last word.
 *
 * @param[out] o  Result buffer, size VEC_N_SIZE_64 words.
 * @param[in]  a  Input buffer, size 2*VEC_N_SIZE_64 words.
 */
static void reduce(uint64_t *o, const uint64_t *a) {
    for (size_t i = 0; i < VEC_N_SIZE_64; i++) {
        uint64_t r = a[i + VEC_N_SIZE_64 - 1] >> (PARAM_N & 0x3F);
        uint64_t carry = a[i + VEC_N_SIZE_64] << (64 - (PARAM_N & 0x3F));
        o[i] = a[i] ^ r ^ carry;
    }

    o[VEC_N_SIZE_64 - 1] &= BITMASK(PARAM_N, 64);
}

/*
 * Unreduced sparse x dense multiplication over GF(2).
 * NOT secure
 *
 * support: positions of 1 bits in the sparse polynomial
 * weight: number of entries in support
 * dense: dense polynomial as uint64_t words
 * result: unreduced output buffer
 */
static void sparse_dense_mult(
    uint64_t *result,
    const uint32_t *support,
    size_t weight,
    const uint64_t *dense
) {
    memset(result, 0, 2 * VEC_N_SIZE_64 * sizeof(uint64_t));

    for (size_t i = 0; i < weight; i++) {
        uint32_t shift = support[i];

        size_t word_shift = shift / 64;
        unsigned int bit_shift = shift % 64;

        for (size_t j = 0; j < VEC_N_SIZE_64; j++) {
            uint64_t x = dense[j];

            result[j + word_shift] ^= x << bit_shift;

            if (bit_shift != 0) {
                result[j + word_shift + 1] ^= x >> (64 - bit_shift);
            }
        }
    }
}

void vect_mult(
    uint64_t *o,
    const uint32_t *support,
    size_t weight,
    const uint64_t *dense
) {
    uint64_t unreduced[2 * VEC_N_SIZE_64];

    sparse_dense_mult(unreduced, support, weight, dense);

    reduce(o, unreduced);
}