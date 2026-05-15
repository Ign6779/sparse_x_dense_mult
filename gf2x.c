#include <stdint.h>
#include <stddef.h>
#include <string.h>

/*
 * Unreduced sparse x dense multiplication over GF(2).
 *
 * Does NOT reduce modulo X^n - 1.
 * Does NOT wrap around cyclically.
 * Is NOT secure
 *
 * support:      positions of 1 bits in the sparse polynomial
 * weight:       number of entries in support
 * dense:        dense polynomial as uint64_t words
 * dense_words:  number of words in dense
 * result:       output buffer
 * result_words: number of words in result
 */

 void sparse_dense_mult(
    uint64_t *result,
    size_t result_words,
    const uint32_t *support,
    size_t weight,
    const uint64_t *dense,
    size_t dense_words
) {
    memset(result, 0, result_words * sizeof(uint64_t));

    for (size_t i = 0; i < weight; i++) {
        uint32_t shift = support[i];

        size_t word_shift = shift / 64;
        unsigned int bit_shift = shift % 64;

        for (size_t j = 0; j < dense_words; j++) {
            uint64_t x = dense[j];

            result[j + word_shift] ^= x << bit_shift;

            if (bit_shift != 0) {
                result[j + word_shift + 1] ^= x >> (64 - bit_shift);
            }
        }
    }
}