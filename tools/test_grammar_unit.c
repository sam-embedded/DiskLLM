#include "grammar.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>

int main() {
    printf("=== Testing Grammar Parser Unit Test ===\n");

    const char *gbnf = 
        "root ::= \"{\" ws \"\\\"bbox_2d\\\":\" ws \"[\" ws int \"]\" ws \"}\"\n"
        "int ::= [0-9]+\n"
        "ws ::= [ \\t\\n\\r]*\n";

    diskllm_grammar *g = diskllm_grammar_init_from_str(gbnf);
    assert(g != NULL);

    assert(diskllm_grammar_accept_str(g, "{"));
    diskllm_grammar_advance(g, "{");

    assert(diskllm_grammar_accept_str(g, "\"bbox_2d\":"));
    assert(!diskllm_grammar_accept_str(g, "invalid"));

    diskllm_grammar_advance(g, "\"bbox_2d\": [123] }");
    assert(diskllm_grammar_is_finished(g));

    diskllm_grammar_free(g);
    printf("Grammar Unit Test: PASSED!\n");
    return 0;
}
