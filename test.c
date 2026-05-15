#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "gf2x.h"

void vect_mult(
    uint64_t *o,
    const uint32_t *support,
    size_t weight,
    const uint64_t *dense
);

static void clear_vec(uint64_t *v) {
    memset(v, 0, VEC_N_SIZE_64 * sizeof(uint64_t));
}

static void toggle_bit(uint64_t *v, size_t bit_pos) {
    v[bit_pos / 64] ^= 1ULL << (bit_pos % 64);
}

static int get_bit(const uint64_t *v, size_t bit_pos) {
    return (v[bit_pos / 64] >> (bit_pos % 64)) & 1ULL;
}

static int check(const char *name, const uint64_t *got, const uint64_t *expected) {
    for (size_t i = 0; i < VEC_N_SIZE_64; i++) {
        if (got[i] != expected[i]) {
            printf("[FAIL] %s\n", name);
            printf("word %zu: got 0x%016lx, expected 0x%016lx\n",
                   i,
                   (unsigned long)got[i],
                   (unsigned long)expected[i]);
            return 0;
        }
    }

    printf("[PASS] %s\n", name);
    return 1;
}

/*
 * Shit reference implementation:
 *
 * for every dense bit j and every sparse support position s,
 * output bit is (j + s) mod PARAM_N.
 *
 * just for testing
 */
static void reference_sparse_dense_mod(
    uint64_t *expected,
    const uint32_t *support,
    size_t weight,
    const uint64_t *dense
) {
    clear_vec(expected);

    for (size_t i = 0; i < weight; i++) {
        uint32_t shift = support[i];

        for (size_t bit = 0; bit < PARAM_N; bit++) {
            if (get_bit(dense, bit)) {
                size_t out_bit = (bit + shift) % PARAM_N;
                toggle_bit(expected, out_bit);
            }
        }
    }
}

int main(void) {
    int ok = 1;

    /*
     * Test 1:
     * Normal shift inside the first word.
     *
     * dense = X^10
     * support = {7}
     *
     * Expected:
     * X^17
     */
    {
        uint64_t result[VEC_N_SIZE_64];
        uint64_t dense[VEC_N_SIZE_64];
        uint64_t expected[VEC_N_SIZE_64];
        uint32_t support[1] = {7};

        clear_vec(dense);
        clear_vec(expected);

        toggle_bit(dense, 10);
        toggle_bit(expected, 17);

        vect_mult(result, support, 1, dense);
        ok &= check("normal bit shift: X^10 * X^7 = X^17", result, expected);
    }

    /*
     * Test 2:
     * Cross 64-bit word boundary.
     *
     * dense = X^63
     * support = {1}
     *
     * Expected:
     * X^64
     */
    {
        uint64_t result[VEC_N_SIZE_64];
        uint64_t dense[VEC_N_SIZE_64];
        uint64_t expected[VEC_N_SIZE_64];
        uint32_t support[1] = {1};

        clear_vec(dense);
        clear_vec(expected);

        toggle_bit(dense, 63);
        toggle_bit(expected, 64);

        vect_mult(result, support, 1, dense);
        ok &= check("cross word boundary: X^63 * X = X^64", result, expected);
    }

    /*
     * Test 3:
     * Whole-word shift.
     *
     * dense = X^3
     * support = {64}
     *
     * Expected:
     * X^67
     */
    {
        uint64_t result[VEC_N_SIZE_64];
        uint64_t dense[VEC_N_SIZE_64];
        uint64_t expected[VEC_N_SIZE_64];
        uint32_t support[1] = {64};

        clear_vec(dense);
        clear_vec(expected);

        toggle_bit(dense, 3);
        toggle_bit(expected, 67);

        vect_mult(result, support, 1, dense);
        ok &= check("whole-word shift: X^3 * X^64 = X^67", result, expected);
    }

    /*
     * Test 4:
     * Exact wraparound modulo X^n - 1.
     *
     * dense = X^(n - 1)
     * support = {1}
     *
     * Unreduced:
     * X^n
     *
     * Reduced:
     * X^n = 1
     *
     * Expected:
     * X^0
     */
    {
        uint64_t result[VEC_N_SIZE_64];
        uint64_t dense[VEC_N_SIZE_64];
        uint64_t expected[VEC_N_SIZE_64];
        uint32_t support[1] = {1};

        clear_vec(dense);
        clear_vec(expected);

        toggle_bit(dense, PARAM_N - 1);
        toggle_bit(expected, 0);

        vect_mult(result, support, 1, dense);
        ok &= check("mod wrap: X^(n-1) * X = 1", result, expected);
    }

    /*
     * Test 5:
     * Larger wraparound.
     *
     * dense = X^(n - 3)
     * support = {5}
     *
     * Unreduced:
     * X^(n + 2)
     *
     * Reduced:
     * X^2
     */
    {
        uint64_t result[VEC_N_SIZE_64];
        uint64_t dense[VEC_N_SIZE_64];
        uint64_t expected[VEC_N_SIZE_64];
        uint32_t support[1] = {5};

        clear_vec(dense);
        clear_vec(expected);

        toggle_bit(dense, PARAM_N - 3);
        toggle_bit(expected, 2);

        vect_mult(result, support, 1, dense);
        ok &= check("mod wrap: X^(n-3) * X^5 = X^2", result, expected);
    }

    /*
     * Test 6:
     * XOR cancellation.
     *
     * dense = 1 + X
     * support = {0, 1}
     *
     * Product:
     * (1 + X) * (1 + X)
     *
     * Over GF(2):
     * 1 + X + X + X^2 = 1 + X^2
     *
     * Expected:
     * X^0 + X^2
     */
    {
        uint64_t result[VEC_N_SIZE_64];
        uint64_t dense[VEC_N_SIZE_64];
        uint64_t expected[VEC_N_SIZE_64];
        uint32_t support[2] = {0, 1};

        clear_vec(dense);
        clear_vec(expected);

        toggle_bit(dense, 0);
        toggle_bit(dense, 1);

        toggle_bit(expected, 0);
        toggle_bit(expected, 2);

        vect_mult(result, support, 2, dense);
        ok &= check("XOR cancellation: (1 + X)(1 + X) = 1 + X^2", result, expected);
    }

    /*
     * Test 7:
     * Actual sparse x dense multiplication with a much denser dense polynomial.
     *
     * dense has many bits set across the polynomial:
     * - regular low bits
     * - cross-word boundary bits
     * - bits in many different words
     * - bits close to PARAM_N to force wraparound
     *
     * sparse support has:
     * 0, 1, 7, 64, 1234
     *
     * Expected result is computed using the slow bit-level reference.
     */
    {
        uint64_t result[VEC_N_SIZE_64];
        uint64_t dense[VEC_N_SIZE_64];
        uint64_t expected[VEC_N_SIZE_64];

        uint32_t support[5] = {
            0,
            1,
            7,
            64,
            1234
        };

        clear_vec(dense);

        /*
         * Fill dense with many deterministic bits.
         * This is not random; it is reproducible.
         */
        for (size_t bit = 0; bit < PARAM_N; bit += 37) {
            toggle_bit(dense, bit);
        }

        for (size_t bit = 11; bit < PARAM_N; bit += 101) {
            toggle_bit(dense, bit);
        }

        for (size_t bit = 23; bit < PARAM_N; bit += 509) {
            toggle_bit(dense, bit);
        }

        /*
         * Add specific edge-case bits.
         */
        toggle_bit(dense, 0);
        toggle_bit(dense, 1);
        toggle_bit(dense, 63);
        toggle_bit(dense, 64);
        toggle_bit(dense, 65);
        toggle_bit(dense, 127);
        toggle_bit(dense, 128);
        toggle_bit(dense, 129);

        toggle_bit(dense, PARAM_N - 1);
        toggle_bit(dense, PARAM_N - 2);
        toggle_bit(dense, PARAM_N - 3);
        toggle_bit(dense, PARAM_N - 64);
        toggle_bit(dense, PARAM_N - 65);

        reference_sparse_dense_mod(expected, support, 5, dense);

        vect_mult(result, support, 5, dense);
        ok &= check("actual sparse x dense multiplication with dense input",
                    result,
                    expected);
    }

    if (ok) {
        printf("\nAll HQC-1 sparse x dense tests passed.\n");
        return 0;
    }

    printf("\nSome tests failed.\n");
    return 1;
}