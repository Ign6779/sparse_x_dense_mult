#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <stdint.h>

/*
 * HQC-1 parameters for now
 * polynomial length n = 17669 bits.
 */
#define PARAM_N 17669

/*
 * Number of 64-bit words needed to store PARAM_N bits.
 * ceil(PARAM_N / 64)
 */
#define VEC_N_SIZE_64 ((PARAM_N + 63) / 64)

/*
 * Mask for the last word.
 * Keeps only the valid bits of the final uint64_t.
 */
#define BITMASK(n, word_size) \
    (((n) % (word_size)) == 0 ? UINT64_MAX : ((1ULL << ((n) % (word_size))) - 1ULL))

#endif