#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *model_path = "/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    printf("Loading tokenizer from %s...\n", model_path);
    tokenizer *tok = tokenizer_init(model_path);
    if (!tok) {
        fprintf(stderr, "Failed to initialize tokenizer from %s\n", model_path);
        return 1;
    }
    printf("Tokenizer initialized successfully. Vocab size: %d\n", tokenizer_vocab_size(tok));

    /* Test 1: Plain text string */
    const char *text1 = "The capital of France is";
    int32_t tokens1[32];
    int count1 = tokenizer_encode(tok, text1, tokens1, 32);

    printf("\nTest 1 Encoding: \"%s\"\n", text1);
    printf("Token count: %d\nToken IDs:", count1);
    for (int i = 0; i < count1; i++) {
        printf(" %d", tokens1[i]);
    }
    printf("\n");

    int32_t expected1[] = {760, 6511, 314, 9338, 369};
    int num_expected1 = sizeof(expected1) / sizeof(expected1[0]);

    assert(count1 == num_expected1);
    for (int i = 0; i < count1; i++) {
        assert(tokens1[i] == expected1[i]);
    }
    printf("Test 1 PASSED! Tokens match expected: [760, 6511, 314, 9338, 369]\n");

    /* Test 2: Chat template with special tokens */
    const char *text2 = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n";
    int32_t tokens2[64];
    int count2 = tokenizer_encode(tok, text2, tokens2, 64);

    printf("\nTest 2 Encoding: \"%s\"\n", text2);
    printf("Token count: %d\nToken IDs:", count2);
    for (int i = 0; i < count2; i++) {
        printf(" %d", tokens2[i]);
    }
    printf("\n");

    /* Verify special tokens mapped correctly */
    assert(count2 > 0);
    assert(tokens2[0] == 248045); /* <|im_start|> */
    /* Find <|im_end|> */
    int found_end = 0;
    for (int i = 0; i < count2; i++) {
        if (tokens2[i] == 248046) found_end = 1;
    }
    assert(found_end == 1);
    printf("Test 2 PASSED! Chat prompt tokenized with special tokens preserved.\n");

    tokenizer_free(tok);
    printf("\nALL TOKENIZER TESTS PASSED!\n");
    return 0;
}
