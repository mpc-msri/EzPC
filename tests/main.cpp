#include "../random.h"
#include "cleartext.h"
#include "llama.h"
#include "../backend/minillama/prng.h"

int main() {
    rand_init();
    prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    // branching_test();
    bn_float();
    bn_int();
    pt_test_maxpooldouble();
    return 0;
}