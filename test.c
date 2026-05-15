#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

void sparse_dense_mult(
    uint64_t *result,
    size_t result_words,
    const uint32_t *support,
    size_t weight,
    const uint64_t *dense,
    size_t dense_words
);

static int check(const char *name, const uint64_t *got, const uint64_t *expected, size_t words) {
    for (size_t i = 0; i < words; i++) {
        if (got[i] != expected[i]) {
            printf("[FAIL] %s\n", name);
            printf("word %zu: got 0x%016lx, expected 0x%016lx\n", i, got[i], expected[i]);
            return 0;
        }
    }

    printf("[PASS] %s\n", name);
    return 1;
}

int main(void) {
    int ok = 1;

    /*
     * Test 1:
     * dense = 1
     * support = {3}
     *
     * Expected:
     * 1 << 3 = 8
     */
    {
        uint64_t result[2];
        uint64_t dense[1] = {1};
        uint32_t support[1] = {3};
        uint64_t expected[2] = {8, 0};

        sparse_dense_mult(result, 2, support, 1, dense, 1);
        ok &= check("1 shifted by 3", result, expected, 2);
    }

    /*
     * Test 2:
     * dense = 0b10001 = bits 0 and 4
     * support = {2}
     *
     * Expected:
     * bits 2 and 6 = 0b1000100 = 68
     */
    {
        uint64_t result[2];
        uint64_t dense[1] = {(1ULL << 0) | (1ULL << 4)};
        uint32_t support[1] = {2};
        uint64_t expected[2] = {(1ULL << 2) | (1ULL << 6), 0};

        sparse_dense_mult(result, 2, support, 1, dense, 1);
        ok &= check("bits 0 and 4 shifted by 2", result, expected, 2);
    }

    /*
     * Test 3:
     * dense = 1
     * support = {2, 5}
     *
     * Expected:
     * (1 << 2) XOR (1 << 5) = 4 XOR 32 = 36
     */
    {
        uint64_t result[2];
        uint64_t dense[1] = {1};
        uint32_t support[2] = {2, 5};
        uint64_t expected[2] = {(1ULL << 2) ^ (1ULL << 5), 0};

        sparse_dense_mult(result, 2, support, 2, dense, 1);
        ok &= check("support positions 2 and 5", result, expected, 2);
    }

    /*
     * Test 4:
     * dense = bit 63 set
     * support = {1}
     *
     * Expected:
     * bit 63 shifted by 1 becomes bit 64,
     * so result[0] = 0, result[1] = 1
     */
    {
        uint64_t result[2];
        uint64_t dense[1] = {1ULL << 63};
        uint32_t support[1] = {1};
        uint64_t expected[2] = {0, 1};

        sparse_dense_mult(result, 2, support, 1, dense, 1);
        ok &= check("cross word boundary", result, expected, 2);
    }

    /*
     * Test 5:
     * dense = 0x1234
     * support = {64}
     *
     * Expected:
     * shift by exactly one word:
     * result[0] = 0
     * result[1] = 0x1234
     */
    {
        uint64_t result[3];
        uint64_t dense[1] = {0x1234};
        uint32_t support[1] = {64};
        uint64_t expected[3] = {0, 0x1234, 0};

        sparse_dense_mult(result, 3, support, 1, dense, 1);
        ok &= check("whole word shift by 64", result, expected, 3);
    }

    /*
    * Test 6:
    * This test catches XOR vs OR.
    *
    * dense = bits 0 and 1
    * support = {0, 1}
    *
    * First shifted copy:
    *   dense << 0 = bits 0 and 1
    *
    * Second shifted copy:
    *   dense << 1 = bits 1 and 2
    *
    * XOR accumulation:
    *   bit 0 = 1
    *   bit 1 = 1 XOR 1 = 0
    *   bit 2 = 1
    *
    * Expected result = bits 0 and 2 = 0b101 = 5
    *
    * If you use OR instead of XOR:
    *   bits 0, 1, and 2 stay set = 0b111 = 7
    */
    {
        uint64_t result[2];
        uint64_t dense[1] = {(1ULL << 0) | (1ULL << 1)};
        uint32_t support[2] = {0, 1};
        uint64_t expected[2] = {(1ULL << 0) | (1ULL << 2), 0};

        sparse_dense_mult(result, 2, support, 2, dense, 1);
        ok &= check("XOR cancellation, not OR", result, expected, 2);
    }

    if (ok) {
        printf("\nAll simple tests passed.\n");
        return 0;
    }

    printf("\nSome tests failed.\n");
    return 1;
}