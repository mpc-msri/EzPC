/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "spline.h"
#include "dcf.h"
#include "utils.h"
#include <assert.h>

std::pair<ReluKeyPack, ReluKeyPack> keyGenRelu(int Bin, int Bout,
                        GroupElement rin, GroupElement rout)
{
    // represents offset poly p(x-rin)'s coefficients, where p(x)=x
    GroupElement beta[2];
    beta[0] = GroupElement(1, Bin);
    beta[1] = -rin;
    ReluKeyPack k0, k1;
    
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    GroupElement gamma = rin - 1;
    auto dcfKeys = keyGenDCF(Bin, Bout, 2, gamma, beta);

    GroupElement p = GroupElement(0, Bin);
    GroupElement q = GroupElement((((uint64_t)1 << (Bin-1)) - 1), Bin);
    GroupElement q1 = q + 1, alpha_L = p + rin, alpha_R = q + rin, alpha_R1 = q + 1 + rin;
    GroupElement cr = GroupElement((alpha_L > alpha_R) - (alpha_L > p) + (alpha_R1 > q1) + (alpha_R == GroupElement(-1, Bin)), Bin);

    GroupElement val;
    val = beta[0] * cr;
    auto val_split = splitShare(val);
    k0.e_b0 = val_split.first;
    k1.e_b0 = val_split.second;

    val = beta[1] * cr;
    val_split = splitShare(val);
    k0.e_b1 = val_split.first;
    k1.e_b1 = val_split.second;

    auto beta_split = splitShare(beta[0]);
    k0.beta_b0 = beta_split.first;
    k1.beta_b0 = beta_split.second;

    beta_split = splitShare(beta[1]);
    k0.beta_b1 = beta_split.first;
    k1.beta_b1 = beta_split.second;


    auto rout_split = splitShare(rout);
    k0.r_b = rout_split.first; k1.r_b = rout_split.second;
    k0.k = dcfKeys.first.k;
    k0.g = dcfKeys.first.g;
    k0.v = dcfKeys.first.v;
    k1.k = dcfKeys.second.k;
    k1.g = dcfKeys.second.g;
    k1.v = dcfKeys.second.v;

    return std::make_pair(k0, k1);
}

GroupElement evalRelu(int party, GroupElement x, const ReluKeyPack &k)
{
    int Bout = k.Bout;
    int Bin = k.Bin;

    GroupElement p = GroupElement(0, Bin);
    GroupElement q = GroupElement((((uint64_t)1 << (Bin-1)) - 1), Bin);
    GroupElement q1 = q.value + 1, xL = x.value - 1, xR1 = x.value - 1 - q1.value;
    GroupElement share_L[2]; 
    evalDCF(Bin, Bout, 2, share_L, party, xL, k.k, k.g, k.v);
    GroupElement share_R1[2];
    evalDCF(Bin, Bout, 2, share_R1, party, xR1, k.k, k.g, k.v);

    GroupElement cx = GroupElement((x.value > 0) - (x.value > q1.value), k.Bin);
    GroupElement sum = GroupElement(0, Bout);
    
    GroupElement w_b = cx.value * k.beta_b0.value - share_L[0].value + share_R1[0].value + k.e_b0.value;
    sum.value = sum.value + (w_b.value * x.value);

    w_b.value = cx.value * k.beta_b1.value - share_L[1].value + share_R1[1].value + k.e_b1.value;
    sum.value = sum.value + w_b.value;

    GroupElement ub(k.r_b.value + sum.value, Bout);
    return ub;
}


std::pair<MaxpoolKeyPack, MaxpoolKeyPack> keyGenMaxpool(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement rout)
{
    // maxpool(x, y) = relu(x - y) + y
    // for correctness, ensure magnitude(x) + magnitude(y) in signed context < N/2
    MaxpoolKeyPack k0, k1;
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    auto reluKeys = keyGenRelu(Bin, Bout, rin1 - rin2, GroupElement(0, Bout));
    k0.reluKey = reluKeys.first; 
    k1.reluKey = reluKeys.second;

    auto rb_split = splitShare(-rin2 + rout);
    k0.rb = rb_split.first; k1.rb = rb_split.second;

    return std::make_pair(k0, k1);
}

GroupElement evalMaxpool(int party, GroupElement x, GroupElement y, const MaxpoolKeyPack &k)
{
    // maxpool(x, y) = relu(x - y) + y
    // for correctness, ensure magnitude(x) + magnitude(y) in signed context < N/2
    GroupElement res = evalRelu(party, x - y, k.reluKey).value + (party * y.value) + k.rb.value;
    return res;
}


std::pair<SplineKeyPack, SplineKeyPack> keyGenSigmoid(int Bin, int Bout, int numPoly, int degree, 
                    std::vector<std::vector<GroupElement>> polynomials,
                    std::vector<GroupElement> p,
                    GroupElement rin, GroupElement rout)
{
    bool print = false;
    /*
    Same as keyGenSpline but with the following change:
    octave poly coefs are such that sigmoid(x) = poly(x-p[i-1]) where p[i-1]<=x<=p[i]
    So evaluator has to find suitable p[i-1] based on his x and input that to spline
    Problem: evaluator knows not x but x + rin, and so the knot preceeding x + rin
    need not be the one that preceeds x, and hence incorrect knot p[i-1] is chosen

    Workaround: when generating offset poly for poly(x), generate coefs of poly(x - rin - p[i-1])
    where poly(x) is the spline poly for interval p[i-1], p[i]
    Now evaluator simply feeds x + rin to this, and FSS selects the interval p[i-1],p[i]
    in which (x + rin) - rin falls, and poly evaluated is poly((x + rin) - rin - p[i-1]) as required

    Summary: line changed is when generateOffsetPolynomial is called

    Update: the above thing is irrelevant now because in octave, I'm generating offset poly directly
    */
    SplineKeyPack k0, k1;
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;
    k0.numPoly = numPoly; k1.numPoly = numPoly;
    k0.degree = degree; k1.degree = degree;
    k0.p = p; k1.p = p;
    k0.beta_b.resize(numPoly * (degree+1));
    k1.beta_b.resize(numPoly * (degree+1));
    k0.e_b.resize(numPoly, std::vector<GroupElement>(degree + 1));
    k1.e_b.resize(numPoly, std::vector<GroupElement>(degree + 1));
    
    assert(polynomials.size() == numPoly);
    const int m = numPoly;
    // size of p: m + 1
    assert(p.size() == (m + 1));
    // p[0] = 0, p[m] = N-1
    assert((p[0] == GroupElement(0, Bin)) && (p[m] == GroupElement(-1, Bin)));

    // line 1 and 2
    std::vector<GroupElement> beta(m * (degree + 1));
    for (int i = 0; i < m; ++i)
    {
        // following line is different in keyGenSpline

        /* if i == 7 == m/2 and inp_bitlen = 64 case: this happens when -8 < x < -6.8
        8 and -8 are same in 16 bitlen and scale 12
        But -8 and 8 are not same in 64 bitlen (8 in fxd with 64 bitlen is 32000 something and -8 is very large)
        So with this inconsistency, x (which is negative value and hence very large in fxd) cant be subtracted simply with p[i]=8
        It should be subtracted with -8 in 64 bitlen
        Hence subtract with dummy -p[i] cast in 64 bits */

        GroupElement prev_knot = p[i];

        if (Bin == 64 && (i == m/2)) {
            prev_knot = GroupElement(-prev_knot.value, 64);
        }

        // std::cout << "rin " << rin << " prev_knot " << prev_knot << std::endl;
        // std::cout << "polynomials[" << i << "]:" << std::endl;
        // for (int j=0; j<degree+1; j++){
        //     std::cout << polynomials[i][j] << "\t";
        // }
        // std::cout << std::endl;

        // auto b = generateOffsetPolynomial(Bout, polynomials[i], GroupElement(rin.value + prev_knot.value, Bout));
        // auto b = generateOffsetPolynomial(Bout, polynomials[i], changeBitsize(rin + prev_knot, Bout));
        auto b = generateOffsetPolynomial(Bout, polynomials[i], GroupElement(rin.value, Bout));

        for (int j = 0; j < degree + 1; ++j)
        {
            beta[i * (degree + 1) + j] = b[j];
            if (i < 2)
            if(print) std::cout << "offset poly for i " << i <<  " j " << j << " is " << b[j] << std::endl;
        }
    }

    GroupElement gamma = rin - 1;

    // line 3
    block seed;
    int dcfGroupSize = beta.size();
#ifdef SIGMOID_TANH_37
    int newBitlen = 37;
    auto dcfKeys = keyGenDCF(newBitlen, Bout, beta.size(), gamma, beta.data());
#elif defined(SIGMOID_12_12) || defined(TANH_12_12)
    auto dcfKeys = keyGenDCF(16, Bout, beta.size(), gamma, beta.data());
    // auto dcfKeys = keyGenDCF(Bin, Bout, beta.size(), gamma, beta.data());
#else // all equal bit case
    // assert(Bin == Bout);
    // auto dcfKeys = keyGenDCF(Bin, Bout, beta.size(), gamma, beta.data());
    auto dcfKeys = keyGenDCF(16, Bout, beta.size(), gamma, beta.data());
#endif 
    
    k0.dcfKey = dcfKeys.first;
    k1.dcfKey = dcfKeys.second;

    std::vector<GroupElement> tmp(degree + 1);

    // if we make p[0] also as N-1, then uniformity in following convention:
    // for i = 1...m, when p[i-1] + 1 <= x <= p[i] then o/p poly[i-1](x)
    p[0] = p[0] - GroupElement(1, Bin);

    // after change, p[0] = p[m] = N - 1
    for (int i = 1; i <= m; ++i)
    {
        GroupElement alpha_L = p[i - 1] + GroupElement(1, Bin) + rin, alpha_R = p[i] + rin; 
        GroupElement alpha_R1 = p[i] + GroupElement(1, Bin) + rin;
        GroupElement cr = GroupElement((alpha_L > alpha_R) - (alpha_L > (p[i - 1] + 1)) + (alpha_R1 > (p[i] + 1)) + (alpha_R == GroupElement(-1, Bin)), Bout);

        if(print) std::cout <<  " i " << i << " cr " << cr << std::endl;

        for (uint32_t j = 0; j < degree + 1; j++)
        {
            // when p[i-1]+1 <= x <= p[i] o/p poly[i-1](x)

            tmp[j] = beta[(i - 1) * (degree + 1) + j] * cr;
            auto tmpj_split = splitShare(tmp[j]);
            k0.e_b[i-1][j] = tmpj_split.first;
            k1.e_b[i-1][j] = tmpj_split.second;
        }
    }

    for (int i = 0; i < beta.size(); i++) {
        auto betai_split = splitShare(beta[i]);
        k0.beta_b[i] = betai_split.first;
        k1.beta_b[i] = betai_split.second;
    }
    
    auto rout_split = splitShare(rout);
    k0.r_b = rout_split.first; k1.r_b = rout_split.second;
    if(print) std::cout << "k0rb " << k0.r_b << " k1rb " << k1.r_b << std::endl;
    return std::make_pair(k0, k1);
}



GroupElement evalSigmoid(int party, GroupElement x, SplineKeyPack &k)
{
    bool print = false;
    if(print) std::cout << "==== entered eval " << std::endl;

    /*
    this function is same as evalSpline, except:
    
    todo: if you make any "core" code changes here reflect the same in evalSpline
    */

    const int m = k.numPoly, degree = k.degree;
    // size of p: m + 1
    // do modifications similar to keygen (make p[0] = N-1)
    assert((k.p[0] == GroupElement(0, k.Bin)) && (k.p[m] == GroupElement(-1, k.Bin)));
    k.p[0] = k.p[0] - GroupElement(1, k.Bin);

    std::vector<std::vector<GroupElement>> s (m, std::vector<GroupElement>(m * (degree + 1)));

    int dcfGroupSize = m * (degree + 1);
    for (int i = 0; i <m ; ++i)
    {
        GroupElement xi = x + (GroupElement(-1, k.Bin) - (k.p[i] + GroupElement(1, k.Bin)));
        evalDCFPartial(party, s[i].data(), xi, k.dcfKey, (i - 1) * (degree + 1), 2 * (degree + 1));
    }

    GroupElement w_b[m][degree + 1];
    for (int i = 0; i < m; ++i)
    {
#ifdef SIGMOID_TANH_37
        int newBitlen = 37;
        GroupElement cx = GroupElement((changeBitsize(x, newBitlen) > (changeBitsize(k.p[i], newBitlen) + GroupElement(1, newBitlen))) - (changeBitsize(x, newBitlen) > (changeBitsize(k.p[i+1], newBitlen) + GroupElement(1, newBitlen))), k.Bout);
#elif defined(SIGMOID_12_12) || defined(TANH_12_12) 
        GroupElement cx = GroupElement((changeBitsize(x, 16) > (changeBitsize(k.p[i], 16) + GroupElement(1, 16))) - (changeBitsize(x, 16) > (changeBitsize(k.p[i+1], 16) + GroupElement(1, 16))), k.Bout);
        // GroupElement cx = GroupElement((x > k.p[i - 1] + 1) - (x > k.p[i] + 1), k.Bout);
#else // all equal bit case
        // assert(k.Bin == k.Bout);
        // GroupElement cx = GroupElement((x > k.p[i] + 1) - (x > k.p[i+1] + 1), k.Bout);
        GroupElement cx = GroupElement((changeBitsize(x, 16) > (changeBitsize(k.p[i], 16) + GroupElement(1, 16))) - (changeBitsize(x, 16) > (changeBitsize(k.p[i+1], 16) + GroupElement(1, 16))), k.Bout);
#endif 
        if(print) std::cout << " i " << i << " cx " << cx << std::endl;
        for (int j = 0; j < degree + 1; ++j)
        {
            w_b[i][j] = cx * k.beta_b[i * (degree + 1) + j] - s[i][i * (degree + 1) + j] + s[(i+1) % (m)][i * (degree + 1) + j] + k.e_b[i][j];
        }
    }

    std::vector<GroupElement> tb(degree + 1, GroupElement(0, k.Bout));
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < degree + 1; ++j)
        {
            tb[j] = tb[j] + w_b[i][j];
        }
    }

    GroupElement sum = tb[0];
    // GroupElement x_adjusted(x.value, k.Bout);
    GroupElement x_adjusted = changeBitsize(x, k.Bout);

    for (int i = 1; i < degree + 1; i++)
    {
        sum = sum * x_adjusted + tb[i];
        if(print) std::cout << "i " << i << " sum " << sum << " tb[i] " << tb[i] << std::endl;
        // // sum = sum + (tb[i] * pow(x_upscaled, degree - i));
        
        // GroupElement coef_scale_adjusted = GroupElement(tb[i].value << (uint64_t(i) * 12), 64);
        // sum = sum * x_adjusted + coef_scale_adjusted;
    }

    GroupElement ub = k.r_b + sum;
    if(print) std::cout << " ub " << ub << std::endl;
    return ub;
}




std::pair<SplineKeyPack, SplineKeyPack> keyGenSigmoid_main_wrapper(int Bin, int Bout, int scaleIn, int scaleOut,
                    GroupElement rin, GroupElement rout)
{
    // todo: add other scales
    assert((Bin == 64) && (Bout == 64));
#if defined(SIGMOID_12_12) || defined(SIGMOID_TANH_37)
    assert((scaleIn == 12) && (scaleOut == 12));
#elif defined(SIGMOID_9_14)
    assert((scaleIn == 9) && (scaleOut == 14));
#elif defined(SIGMOID_8_14)
    assert((scaleIn == 8) && (scaleOut == 14));
#elif defined(SIGMOID_11_14)
    assert((scaleIn == 11) && (scaleOut == 14));
#elif defined(SIGMOID_13_14)
    assert((scaleIn == 13) && (scaleOut == 14));
#else 
    throw std::invalid_argument("no scales selected for sigmoid");
#endif

    // imp: make sure different choices of i/p, o/p scales use same coef scale
    int scaleCoef = 20, coefBitsize = 64;

    int ib = Bin, cb = coefBitsize, ob = Bout, sin = scaleIn, scoef = scaleCoef, sout = scaleOut;
    
    // from octave:

    // best fit for spline with ulp 4: degree = 2 and 32 polys
    // bitlen = 16
    // scalein = 9, coefscale = 18, scaleout = 14

    // poly coefs: size 32x3
    //     10        115200     310902784
    //     19        196096     498335744
    //     35        337920     807403520
    //     65        578560    1293156352
    //     120        985600    2051014656
    //     219       1666048    3213361152
    //     400       2790912    4960550912
    //     728       4619776    7515930624
    //     1311       7516672   11114905600
    //     2325      11926528   15908470784
    //     4011      18210304   21762932736
    //     6580      26188800   27957395456
    //     9856      34328576   33013366784
    //     12528      39306752   35332030464
    //     11687      38262272   35008020480
    //     4966      34087424   34359476224

    // (marks start of x=0)

    //     -4967      34087424   34359738368
    // -11688      38262272   33711194112
    // -12529      39306752   33387184128
    //     -9857      34328576   35705847808
    //     -6581      26188800   40761819136
    //     -4012      18210304   46956281856
    //     -2326      11926528   52810743808
    //     -1312       7516672   57604308992
    //     -729       4619776   61203283968
    //     -401       2790912   63758663680
    //     -220       1666048   65505853440
    //     -121        985600   66668199936
    //     -66        578560   67426058240
    //     -36        337920   67911811072
    //     -20        196096   68220878848
    //     -11        115200   6840831180

    // breakpoints: size 33
    // -4969  -4659  -4348  -4038  -3727  -3417  -3106  -2796  -2485  -2174  -1864  -1553  -1243   -932   -622   -311
    // 0    310    621    931   1242   1552   1863   2173   2484   2795   3105   3416   3726   4037   4347   4658
    // 4969

    // task 1: rearrange breakpoints so that their fxd point values are in ascending order, i.e. positives values first then negative values
    // this means changing order of poly coefs is also needed
    // task 2: add polys y=1 after right breakp and y=0 before left breakp
    // in fxd, coefs of y=1 are (0, 0, 1<<(degree*sin + scoef))

    // rearranged breakpoints:
    // 0    310    621    931   1242   1552   1863   2173   2484   2795   3105   3416   3726   4037   4347   4658
    // 4969  (N/2 - 1) = 32767
    // -4969  -4659  -4348  -4038  -3727  -3417  -3106  -2796  -2485  -2174  -1864  -1553  -1243   -932   -622   -311

    // rearranged polys:
    // (from x=0 to N/2-1)
    //     -4967      34087424   34359738368
    //     -11688      38262272   33711194112
    //     -12529      39306752   33387184128
    //     -9857      34328576   35705847808
    //     -6581      26188800   40761819136
    //     -4012      18210304   46956281856
    //     -2326      11926528   52810743808
    //     -1312       7516672   57604308992
    //     -729       4619776   61203283968
    //     -401       2790912   63758663680
    //     -220       1666048   65505853440
    //     -121        985600   66668199936
    //     -66        578560   67426058240
    //     -36        337920   67911811072
    //     -20        196096   68220878848
    //     -11        115200   68408311808
    //      0         0        flt2fxd(1, degree*sin + scoef)

    // from x=N/2-1 to N-1

        // 0         0          0
        // 10        115200     310902784
        // 19        196096     498335744
        // 35        337920     807403520
        // 65        578560    1293156352
        // 120        985600    2051014656
        // 219       1666048    3213361152
        // 400       2790912    4960550912
        // 728       4619776    7515930624
        // 1311       7516672   11114905600
        // 2325      11926528   15908470784
        // 4011      18210304   21762932736
        // 6580      26188800   27957395456
        // 9856      34328576   33013366784
        // 12528      39306752   35332030464
        // 11687      38262272   35008020480
        // 4966      34087424   34359476224

    int degree = 2;

#ifdef SIGMOID_TANH_37

    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        {GroupElement(   -29928, cb),  GroupElement(     1114308608, cb),  GroupElement(   8796093022208, cb)},
        {GroupElement(   -52222, cb),  GroupElement(     1283092480, cb),  GroupElement(   8476621275136, cb)},
        {GroupElement(   -35115, cb),  GroupElement(     1024057344, cb),  GroupElement(   9457182441472, cb)},
        {GroupElement(   -17337, cb),  GroupElement(      620269568, cb),  GroupElement(  11749956780032, cb)},
        {GroupElement(    -7459, cb),  GroupElement(      321150976, cb),  GroupElement(  14014562172928, cb)},
        {GroupElement(    -3077, cb),  GroupElement(      155246592, cb),  GroupElement(  15584590823424, cb)},
        {GroupElement(    -1232, cb),  GroupElement(       71475200, cb),  GroupElement(  16535942856704, cb)},
        {GroupElement(     -493, cb),  GroupElement(       32301056, cb),  GroupElement(  17054962810880, cb)},
        {GroupElement(     -200, cb),  GroupElement(       14553088, cb),  GroupElement(  17323733811200, cb)},
        {GroupElement(        0, cb),  GroupElement(              0, cb),  flt2fxd(1, degree*sin + scoef, cb)},

        // after x = N/2

        {GroupElement(        0, cb),  GroupElement(              0, cb),  GroupElement(               0, cb)},
        {GroupElement(      199, cb),  GroupElement(       14553088, cb),  GroupElement(    268435456000, cb)},
        {GroupElement(      492, cb),  GroupElement(       32301056, cb),  GroupElement(    537206456320, cb)},
        {GroupElement(     1231, cb),  GroupElement(       71475200, cb),  GroupElement(   1056226410496, cb)},
        {GroupElement(     3076, cb),  GroupElement(      155246592, cb),  GroupElement(   2007578443776, cb)},
        {GroupElement(     7458, cb),  GroupElement(      321150976, cb),  GroupElement(   3577607094272, cb)},
        {GroupElement(    17336, cb),  GroupElement(      620269568, cb),  GroupElement(   5842212487168, cb)},
        {GroupElement(    35114, cb),  GroupElement(     1024057344, cb),  GroupElement(   8134986825728, cb)},
        {GroupElement(    52221, cb),  GroupElement(     1283092480, cb),  GroupElement(   9115547992064, cb)},
        {GroupElement(    29927, cb),  GroupElement(     1114308608, cb),  GroupElement(   8796093022208, cb)},
    };

    std::vector<GroupElement> fxd_p{
        GroupElement(0, cb), GroupElement( 3785, cb), GroupElement( 7570, cb), GroupElement(11356, cb), GroupElement(15141, cb), GroupElement( 18927, cb), GroupElement(22712, cb), GroupElement(26498, cb), GroupElement(30283, cb), GroupElement(34069, cb), /* important one */ GroupElement((1ULL << 36) - 1, cb),
        GroupElement(-34069, cb), GroupElement(-30284, cb), GroupElement(-26499, cb), GroupElement(-22713, cb), GroupElement(-18928, cb), GroupElement(-15142, cb), GroupElement(-11357, cb), GroupElement(-7571, cb), GroupElement(-3786, cb)
    };

#elif defined(SIGMOID_12_12)  // when bitlen = 16 and scale is 12

   // the above defined steps are a little different here because max allowed
   // value of x with bitlen 16 and scale 12 is [-8, 8] and turns out those are the
   // leftmost and rightmost breakpoints

    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        { GroupElement(   -28859 , cb) ,   GroupElement(    1111248896 , cb) , GroupElement(    8796093022208 , cb) },
        { GroupElement(   -52266 , cb) ,   GroupElement(    1281687552 , cb) , GroupElement(    8485815189504 , cb) },
        { GroupElement(   -37164 , cb) ,   GroupElement(    1061756928 , cb) , GroupElement(    9286558154752 , cb) },
        { GroupElement(   -19267 , cb) ,   GroupElement(     670793728 , cb) , GroupElement(   11421727326208 , cb) },
        { GroupElement(    -8685 , cb) ,   GroupElement(     362573824 , cb) , GroupElement(   13666132951040 , cb) },
        { GroupElement(    -3723 , cb) ,   GroupElement(     181891072 , cb) , GroupElement(   15310753103872 , cb) },
        { GroupElement(    -1552 , cb) ,   GroupElement(      87056384 , cb) , GroupElement(   16346595196928 , cb) },
        { GroupElement(     -643 , cb) ,   GroupElement(      40681472 , cb) , GroupElement(   16937522298880 , cb) },
        { GroupElement(     -271 , cb) ,   GroupElement(      19050496 , cb) , GroupElement(   17252564860928 , cb) },

        // marks end of x = 32768
        // add dummy polynomial for the interval 32768 to -32768 (because we are operating in 64 bitlen for breakp bitlen)
        // if we change break points to 16 bitlen need to remove/change this
        // currently breakpoints are represented in 64 bitlen as well (although dcf comparison happens in 16 bits)

        { GroupElement(     0 , cb) ,   GroupElement(      0 , cb) , GroupElement(   0 , cb) },

        // marks start of x = -32768

        {GroupElement(      270 , cb) ,    GroupElement(     19050496 , cb) , GroupElement(     339604406272 , cb) },
        {GroupElement(      642 , cb) ,    GroupElement(     40681472 , cb) , GroupElement(     654646968320 , cb) },
        {GroupElement(     1551 , cb) ,    GroupElement(     87056384 , cb) , GroupElement(    1245574070272 , cb) },
        {GroupElement(     3722 , cb) ,    GroupElement(    181891072 , cb) , GroupElement(    2281416163328 , cb) },
        {GroupElement(     8684 , cb) ,    GroupElement(    362573824 , cb) , GroupElement(    3926036316160 , cb) },
        {GroupElement(    19266 , cb) ,    GroupElement(    670793728 , cb) , GroupElement(    6170441940992 , cb) },
        {GroupElement(    37163 , cb) ,    GroupElement(   1061756928 , cb) , GroupElement(    8305611112448 , cb) },
        {GroupElement(    52265 , cb) ,    GroupElement(   1281687552 , cb) , GroupElement(    9106354077696 , cb) },
        {GroupElement(    28858 , cb) ,    GroupElement(   1111248896 , cb) , GroupElement(    8796093022208 , cb) },
    };

    std::vector<GroupElement> fxd_p{ GroupElement(0, ib),    GroupElement(3640, ib),    GroupElement(7281, ib),   GroupElement(10922, ib),   GroupElement(14563, ib),   GroupElement(18204, ib),   GroupElement(21845, ib),   GroupElement(25486, ib),   GroupElement(29127, ib),   GroupElement(32767, ib),
    GroupElement(-32768, ib),  GroupElement(-29128, ib),  GroupElement(-25487, ib),  GroupElement(-21846, ib),  GroupElement(-18205, ib),  GroupElement(-14564, ib),  GroupElement(-10923, ib),   GroupElement(-7282, ib),   GroupElement(-3641, ib)
    };

#elif defined(SIGMOID_9_14)  // when scale is not 12  (input scale 9, output scale 14)
    std::vector<std::vector<GroupElement>> fxd_polynomials
    {

        { GroupElement( -19865 , cb)  ,  GroupElement(    136350720 , cb) , GroupElement(   137438953472 , cb)},

        { GroupElement( -46751 , cb)  ,  GroupElement(    153050624 , cb) , GroupElement(   134845562880 , cb)},

        { GroupElement( -50113 , cb)  ,  GroupElement(    157227008 , cb) , GroupElement(   133548736512 , cb)},

        { GroupElement( -39428 , cb)  ,  GroupElement(    137315840 , cb) , GroupElement(   142824177664 , cb)},

        { GroupElement( -26323 , cb)  ,  GroupElement(    104756736 , cb) , GroupElement(   163047276544 , cb)},

        { GroupElement( -16046 , cb)  ,  GroupElement(     72842240 , cb) , GroupElement(   187825913856 , cb)},

        { GroupElement(  -9302 , cb)  ,  GroupElement(     47707136 , cb) , GroupElement(   211243761664 , cb)},

        { GroupElement(  -5245 , cb)  ,  GroupElement(     30067200 , cb) , GroupElement(   230417760256 , cb)},

        { GroupElement(  -2913 , cb)  ,  GroupElement(     18479104 , cb) , GroupElement(   244813398016 , cb)},

        { GroupElement(  -1604 , cb)  ,  GroupElement(     11164672 , cb) , GroupElement(   255035441152 , cb)},

        { GroupElement(   -880 , cb)  ,  GroupElement(      6664192 , cb) , GroupElement(   262023938048 , cb)},

        { GroupElement(   -481 , cb)  ,  GroupElement(      3942400 , cb) , GroupElement(   266673324032 , cb)},

        { GroupElement(   -263 , cb)  ,  GroupElement(      2315264 , cb) , GroupElement(   269705019392 , cb)},

        { GroupElement(   -144 , cb)  ,  GroupElement(      1353216 , cb) , GroupElement(   271647244288 , cb)},

        { GroupElement(    -78 , cb)  ,  GroupElement(       784384 , cb) , GroupElement(   272884039680 , cb)},

        { GroupElement(    -44 , cb)  ,  GroupElement(       462336 , cb) , GroupElement(   273633247232 , cb)},

        { GroupElement(      0 , cb)  ,  GroupElement(            0 , cb) , flt2fxd(1, degree*sin + scoef, cb)},

       

       // after x=N/2

 

        { GroupElement(      0 , cb)  , GroupElement(              0 , cb) , GroupElement(              0 , cb)},

        { GroupElement(     43 , cb) ,  GroupElement(         462336 , cb) , GroupElement(     1244397568 , cb)},

        { GroupElement(     77 , cb) ,  GroupElement(         784384 , cb) , GroupElement(     1993605120 , cb)},

        { GroupElement(    143 , cb) ,  GroupElement(        1353216 , cb) , GroupElement(     3230400512 , cb)},

        { GroupElement(    262 , cb) ,  GroupElement(        2315264 , cb) , GroupElement(     5172625408 , cb)},

        { GroupElement(    480 , cb) ,  GroupElement(        3942400 , cb) , GroupElement(     8204320768 , cb)},

        { GroupElement(    879 , cb) ,  GroupElement(        6664192 , cb) , GroupElement(    12853706752 , cb)},

        { GroupElement(   1603 , cb) ,  GroupElement(       11164672 , cb) , GroupElement(    19842203648 , cb)},

        { GroupElement(   2912 , cb) ,  GroupElement(       18479104 , cb) , GroupElement(    30064246784 , cb)},

        { GroupElement(   5244 , cb) ,  GroupElement(       30067200 , cb) , GroupElement(    44459884544 , cb)},

        { GroupElement(   9301 , cb) ,  GroupElement(       47707136 , cb) , GroupElement(    63633883136 , cb)},

        { GroupElement(  16045 , cb) ,  GroupElement(       72842240 , cb) , GroupElement(    87051730944 , cb)},

        { GroupElement(  26322 , cb) ,  GroupElement(      104756736 , cb) , GroupElement(   111830368256 , cb)},

        { GroupElement(  39427 , cb) ,  GroupElement(      137315840 , cb) , GroupElement(   132053467136 , cb)},

        { GroupElement(  50112 , cb) ,  GroupElement(      157227008 , cb) , GroupElement(   141328908288 , cb)},

        { GroupElement(  46750 , cb) ,  GroupElement(      153050624 , cb) , GroupElement(   140032081920 , cb)},

        { GroupElement(  19864 , cb) ,  GroupElement(      136350720 , cb) , GroupElement(   137438691328 , cb)},
    };

    std::vector<GroupElement> fxd_p{GroupElement(0, ib),    GroupElement(310, ib),    GroupElement(621, ib),    GroupElement(931, ib),   GroupElement(1242, ib),   GroupElement(1552, ib),   GroupElement(1863, ib),   GroupElement(2173, ib),   GroupElement(2484, ib),   GroupElement(2795, ib),   GroupElement(3105, ib),   GroupElement(3416, ib),   GroupElement(3726, ib),   GroupElement(4037, ib),   GroupElement(4347, ib),  GroupElement(4658, ib),
    GroupElement(4969, ib), /* important one */ GroupElement(32767, ib),
    GroupElement(-4969, ib),  GroupElement(-4659, ib),  GroupElement(-4348, ib),  GroupElement(-4038, ib),  GroupElement(-3727, ib),  GroupElement(-3417, ib),  GroupElement(-3106, ib),  GroupElement(-2796, ib),  GroupElement(-2485, ib),  GroupElement(-2174, ib),  GroupElement(-1864, ib),  GroupElement(-1553, ib),  GroupElement(-1243, ib),  GroupElement(-932, ib),   GroupElement(-622, ib),   GroupElement(-311, ib)
    };

#elif defined(SIGMOID_8_14)
    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        {GroupElement(-19869, cb),     GroupElement( 68175872, cb),  GroupElement(34359738368, cb)},
        {GroupElement(-46756, cb),     GroupElement( 76527616, cb),  GroupElement(33711128576, cb)},
        {GroupElement(-50110, cb),     GroupElement( 78611456, cb),  GroupElement(33387511808, cb)},
        {GroupElement(-39418, cb),     GroupElement( 68647680, cb),  GroupElement(35708731392, cb)},
        {GroupElement(-26312, cb),     GroupElement( 52363008, cb),  GroupElement(40767193088, cb)},
        {GroupElement(-16037, cb),     GroupElement( 36404992, cb),  GroupElement(46963359744, cb)},
        {GroupElement( -9295, cb),     GroupElement( 23839744, cb),  GroupElement(52818018304, cb)},
        {GroupElement( -5240, cb),     GroupElement( 15022848, cb),  GroupElement(57610797056, cb)},
        {GroupElement( -2910, cb),     GroupElement(  9231616, cb),  GroupElement(61208592384, cb)},
        {GroupElement( -1602, cb),     GroupElement(  5576960, cb),  GroupElement(63762923520, cb)},
        {GroupElement(  -879, cb),     GroupElement(  3328512, cb),  GroupElement(65508999168, cb)},
        {GroupElement(  -481, cb),     GroupElement(  1968640, cb),  GroupElement(66670493696, cb)},
        {GroupElement(  -263, cb),     GroupElement(  1156096, cb),  GroupElement(67427762176, cb)},
        {GroupElement(  -144, cb),     GroupElement(   675584, cb),  GroupElement(67912859648, cb)},
        {GroupElement(   -78, cb),     GroupElement(   391424, cb),  GroupElement(68221730816, cb)},
        {GroupElement(   -44, cb),     GroupElement(   230912, cb),  GroupElement(68408836096, cb)},
        {GroupElement(     0, cb),     GroupElement(        0, cb),  flt2fxd(1, degree*sin + scoef, cb)},
        // start of x=N/2
        {GroupElement(     0, cb),     GroupElement(        0, cb),  GroupElement(          0, cb)},
        {GroupElement(    43, cb),     GroupElement(   230912, cb),  GroupElement(  310575104, cb)},
        {GroupElement(    77, cb),     GroupElement(   391424, cb),  GroupElement(  497680384, cb)},
        {GroupElement(   143, cb),     GroupElement(   675584, cb),  GroupElement(  806551552, cb)},
        {GroupElement(   262, cb),     GroupElement(  1156096, cb),  GroupElement( 1291649024, cb)},
        {GroupElement(   480, cb),     GroupElement(  1968640, cb),  GroupElement( 2048917504, cb)},
        {GroupElement(   878, cb),     GroupElement(  3328512, cb),  GroupElement( 3210412032, cb)},
        {GroupElement(  1601, cb),     GroupElement(  5576960, cb),  GroupElement( 4956487680, cb)},
        {GroupElement(  2909, cb),     GroupElement(  9231616, cb),  GroupElement( 7510818816, cb)},
        {GroupElement(  5239, cb),     GroupElement( 15022848, cb),  GroupElement(11108614144, cb)},
        {GroupElement(  9294, cb),     GroupElement( 23839744, cb),  GroupElement(15901392896, cb)},
        {GroupElement( 16036, cb),     GroupElement( 36404992, cb),  GroupElement(21756051456, cb)},
        {GroupElement( 26311, cb),     GroupElement( 52363008, cb),  GroupElement(27952218112, cb)},
        {GroupElement( 39417, cb),     GroupElement( 68647680, cb),  GroupElement(33010679808, cb)},
        {GroupElement( 50109, cb),     GroupElement( 78611456, cb),  GroupElement(35331899392, cb)},
        {GroupElement( 46755, cb),     GroupElement( 76527616, cb),  GroupElement(35008282624, cb)},
        {GroupElement( 19868, cb),     GroupElement( 68175872, cb),  GroupElement(34359738368, cb)},
    };

    std::vector<GroupElement> fxd_p 
    {   GroupElement(0, ib),
        GroupElement(155, ib),
        GroupElement(310, ib),
        GroupElement(465, ib),
        GroupElement(621, ib),
        GroupElement(776, ib),
        GroupElement(931, ib),
        GroupElement(1087, ib),
        GroupElement(1242, ib),
        GroupElement(1397, ib),
        GroupElement(1553, ib),
        GroupElement(1708, ib),
        GroupElement(1863, ib),
        GroupElement(2019, ib),
        GroupElement(2174, ib),
        GroupElement(2329, ib),
        GroupElement(2485, ib),
        // x=N/2
        GroupElement(32767, ib),
        GroupElement(-2485, ib),
        GroupElement(-2330, ib),
        GroupElement(-2175, ib),
        GroupElement(-2020, ib),
        GroupElement(-1864, ib),
        GroupElement(-1709, ib),
        GroupElement(-1554, ib),
        GroupElement(-1398, ib),
        GroupElement(-1243, ib),
        GroupElement(-1088, ib),
        GroupElement( -932, ib),
        GroupElement( -777, ib),
        GroupElement( -622, ib),
        GroupElement( -466, ib),
        GroupElement( -311, ib),
        GroupElement( -156, ib)
    };
#elif defined(SIGMOID_11_14)
    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        {GroupElement(-19863, cb), GroupElement(  545402880, cb), GroupElement(  2199019061248, cb)},
        {GroupElement(-46749, cb), GroupElement(  612192256, cb), GroupElement(  2157541588992, cb)},
        {GroupElement(-50115, cb), GroupElement(  628916224, cb), GroupElement(  2136767201280, cb)},
        {GroupElement(-39432, cb), GroupElement(  549302272, cb), GroupElement(  2285102956544, cb)},
        {GroupElement(-26328, cb), GroupElement(  419088384, cb), GroupElement(  2608588652544, cb)},
        {GroupElement(-16051, cb), GroupElement(  291432448, cb), GroupElement(  3004996517888, cb)},
        {GroupElement( -9305, cb), GroupElement(  190885888, cb), GroupElement(  3379673694208, cb)},
        {GroupElement( -5247, cb), GroupElement(  120313856, cb), GroupElement(  3686482837504, cb)},
        {GroupElement( -2914, cb), GroupElement(   73947136, cb), GroupElement(  3916846596096, cb)},
        {GroupElement( -1605, cb), GroupElement(   44681216, cb), GroupElement(  4080437035008, cb)},
        {GroupElement(  -880, cb), GroupElement(   26671104, cb), GroupElement(  4192286539776, cb)},
        {GroupElement(  -482, cb), GroupElement(   15777792, cb), GroupElement(  4266706075648, cb)},
        {GroupElement(  -263, cb), GroupElement(    9267200, cb), GroupElement(  4315229978624, cb)},
        {GroupElement(  -144, cb), GroupElement(    5416960, cb), GroupElement(  4346322354176, cb)},
        {GroupElement(   -79, cb), GroupElement(    3139584, cb), GroupElement(  4366123663360, cb)},
        {GroupElement(   -44, cb), GroupElement(    1851392, cb), GroupElement(  4378115178496, cb)},
        {GroupElement(     0, cb), GroupElement(          0, cb), flt2fxd(1, degree*sin + scoef, cb)},
        // x=N/2
        {GroupElement(     0, cb), GroupElement(          0, cb), GroupElement(              0, cb)},
        {GroupElement(    43, cb), GroupElement(    1851392, cb), GroupElement(    19927138304, cb)},
        {GroupElement(    78, cb), GroupElement(    3139584, cb), GroupElement(    31918653440, cb)},
        {GroupElement(   143, cb), GroupElement(    5416960, cb), GroupElement(    51719962624, cb)},
        {GroupElement(   262, cb), GroupElement(    9267200, cb), GroupElement(    82812338176, cb)},
        {GroupElement(   481, cb), GroupElement(   15777792, cb), GroupElement(   131336241152, cb)},
        {GroupElement(   879, cb), GroupElement(   26671104, cb), GroupElement(   205755777024, cb)},
        {GroupElement(  1604, cb), GroupElement(   44681216, cb), GroupElement(   317605281792, cb)},
        {GroupElement(  2913, cb), GroupElement(   73947136, cb), GroupElement(   481195720704, cb)},
        {GroupElement(  5246, cb), GroupElement(  120313856, cb), GroupElement(   711559479296, cb)},
        {GroupElement(  9304, cb), GroupElement(  190885888, cb), GroupElement(  1018368622592, cb)},
        {GroupElement( 16050, cb), GroupElement(  291432448, cb), GroupElement(  1393045798912, cb)},
        {GroupElement( 26327, cb), GroupElement(  419088384, cb), GroupElement(  1789453664256, cb)},
        {GroupElement( 39431, cb), GroupElement(  549302272, cb), GroupElement(  2112939360256, cb)},
        {GroupElement( 50114, cb), GroupElement(  628916224, cb), GroupElement(  2261275115520, cb)},
        {GroupElement( 46748, cb), GroupElement(  612192256, cb), GroupElement(  2240500727808, cb)},
        {GroupElement( 19862, cb), GroupElement(  545402880, cb), GroupElement(  2199019061248, cb)},
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(     0, ib),
        GroupElement(  1242, ib),
        GroupElement(  2484, ib),
        GroupElement(  3726, ib),
        GroupElement(  4968, ib),
        GroupElement(  6210, ib),
        GroupElement(  7452, ib),
        GroupElement(  8694, ib),
        GroupElement(  9937, ib),
        GroupElement( 11179, ib),
        GroupElement( 12421, ib),
        GroupElement( 13663, ib),
        GroupElement( 14905, ib),
        GroupElement( 16147, ib),
        GroupElement( 17389, ib),
        GroupElement( 18631, ib),
        GroupElement( 19874, ib),
        // x=N/2
        GroupElement( 32767, ib),
        GroupElement(-19874, ib),
        GroupElement(-18632, ib),
        GroupElement(-17390, ib),
        GroupElement(-16148, ib),
        GroupElement(-14906, ib),
        GroupElement(-13664, ib),
        GroupElement(-12422, ib),
        GroupElement(-11180, ib),
        GroupElement( -9937, ib),
        GroupElement( -8695, ib),
        GroupElement( -7453, ib),
        GroupElement( -6211, ib),
        GroupElement( -4969, ib),
        GroupElement( -3727, ib),
        GroupElement( -2485, ib),
        GroupElement( -1243, ib)
    };
#elif defined(SIGMOID_13_14)
    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        {GroupElement( -18720, cb), GroupElement(  2177662976, cb), GroupElement( 35184304979968, cb)},
        {GroupElement( -45253, cb), GroupElement(  2426077184, cb), GroupElement( 34602940891136, cb)},
        {GroupElement( -50752, cb), GroupElement(  2529050624, cb), GroupElement( 34120897921024, cb)},
        {GroupElement( -42055, cb), GroupElement(  2284789760, cb), GroupElement( 35835999158272, cb)},
        {GroupElement( -29603, cb), GroupElement(  1818443776, cb), GroupElement( 40202034741248, cb)},
        {GroupElement( -18884, cb), GroupElement(  1316691968, cb), GroupElement( 46074060341248, cb)},
        {GroupElement( -11677, cb), GroupElement(   911826944, cb), GroupElement( 51759657517056, cb)},
        // dummy poly b/w 2^16-1 and -2^16, problem arising b/c we have ib=64 and actual bitlen=16}
        {GroupElement(      0, cb), GroupElement(           0, cb), GroupElement(              0, cb)},
        {GroupElement(  11676, cb), GroupElement(   911826944, cb), GroupElement( 18609019551744, cb)},
        {GroupElement(  18883, cb), GroupElement(  1316691968, cb), GroupElement( 24294616727552, cb)},
        {GroupElement(  29602, cb), GroupElement(  1818443776, cb), GroupElement( 30166642327552, cb)},
        {GroupElement(  42054, cb), GroupElement(  2284789760, cb), GroupElement( 34532677910528, cb)},
        {GroupElement(  50751, cb), GroupElement(  2529050624, cb), GroupElement( 36247779147776, cb)},
        {GroupElement(  45252, cb), GroupElement(  2426077184, cb), GroupElement( 35765736177664, cb)},
        {GroupElement(  18719, cb), GroupElement(  2177662976, cb), GroupElement( 35184304979968, cb)}
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(     0, ib),
        GroupElement(  4681, ib),
        GroupElement(  9362, ib),
        GroupElement( 14043, ib),
        GroupElement( 18724, ib),
        GroupElement( 23405, ib),
        GroupElement( 28086, ib),
        GroupElement( 32767, ib),
        // blank area that is due to ib=64 and actual i/p bitlen 16
        GroupElement(-32768, ib),
        GroupElement(-28087, ib),
        GroupElement(-23406, ib),
        GroupElement(-18725, ib),
        GroupElement(-14044, ib),
        GroupElement( -9363, ib),
        GroupElement( -4682, ib)
    };
#else 
    throw std::invalid_argument("no scales selected for sigmoid");    
#endif

    int numPoly = fxd_polynomials.size(), m = numPoly;
    fxd_p.push_back(GroupElement(-1, ib));

    return keyGenSigmoid(ib, ob, numPoly, degree, fxd_polynomials, fxd_p, rin, rout);
}

GroupElement evalSigmoid_main_wrapper(int party, GroupElement x, SplineKeyPack &k)
{
    return evalSigmoid(party, x, k);
}

std::pair<SplineKeyPack, SplineKeyPack> keyGenTanh_main_wrapper(int Bin, int Bout, int scaleIn, int scaleOut,
                    GroupElement rin, GroupElement rout)
{
    // todo: add other scales
    assert((Bin == 64) && (Bout == 64));
#if defined(TANH_12_12) || defined(SIGMOID_TANH_37)
    assert((scaleIn == 12) && (scaleOut == 12));
#elif defined(TANH_9_9)
    assert((scaleIn == 9) && (scaleOut == 9));
#elif defined(TANH_8_8)
    assert((scaleIn == 8) && (scaleOut == 8));
#elif defined(TANH_11_11)
    assert((scaleIn == 11) && (scaleOut == 11));
#elif defined(TANH_13_13)
    assert((scaleIn == 13) && (scaleOut == 13));
#else 
    throw std::invalid_argument("no scales selected for tanh");
#endif

    // imp: make sure different choices of i/p, o/p scales use same coef scale
    int scaleCoef = 18, coefBitsize = 64;

    int ib = Bin, cb = coefBitsize, ob = Bout, sin = scaleIn, scoef = scaleCoef, sout = scaleOut;
    // from octave:

    // best fit for spline with ulp 4: degree = 2 and 10 polys
    // bitlen = 16
    // scalein = 9, coefscale = 18, scaleout = 9

    // poly coefs: size 10x3
    // 1191       4930048     -63602425856
    // 8318       25172480    -49230381056
    // 25411      61579776    -29843259392
    // 85270      146579968   331612160
    // 82638      144710656             0

    // (marks start of x=0)


    // -82639     144710656             0
    // -85271     146579968    -331874304
    // -25412     61579776     29842997248
    // -8319      25172480     49230118912
    // -1192      4930048      63602163712

    

    // breakpoints: size 11
    // -1775  -1420  -1065   -710   -355      0    355    710   1065   1420   1775

    // task 1: rearrange breakpoints so that their fxd point values are in ascending order, i.e. positives values first then negative values
    // this means changing order of poly coefs is also needed
    // task 2: add polys y=1 after right breakp and y=-1 before left breakp
    // in fxd, coefs of y=1 are (0, 0, 1<<(degree*sin + scoef))
    // in fxd, coefs of y=-1 are (0, 0, flt2fxd(1, degree*sin + scoef))

    // rearranged breakpoints:
    // 0    355    710   1065   1420   1775  (N/2 - 1) = 32767
    // -1775  -1420  -1065   -710   -355

    // rearranged polys:
    // (from x=0 to N/2-1)
    // -82639     144710656             0
    // -85271     146579968    -331874304
    // -25412     61579776     29842997248
    // -8319      25172480     49230118912
    // -1192      4930048      63602163712
    // 0          0            fxd2flt(1, degree*sin + scoef)

    // from x=N/2-1 to N-1

        // 0         0          fxd2flt(-1, degree*sin + scoef)
        // 1191       4930048     -63602425856
        // 8318       25172480    -49230381056
        // 25411      61579776    -29843259392
        // 85270      146579968   331612160
        // 82638      144710656             0

    int degree = 2;

#if defined(TANH_12_12) || defined(SIGMOID_TANH_37)

// turns out coefs and breaks for bitlen 16 and 37 are same (scale being 12 in both)
// (but i'm still including this case)
// this is because flt cutoffs are -4.3 to 4.3 (can be fit in both bitlengths)

// this wont happen in sigmoid because for bitlen 16, flt cutoffs are forced to be 8
// (actual flt cutoffs go > 8 but can only do so for bitlen 37)
// so, intervals for spline will change and so will coefs

    std::vector<std::vector<GroupElement>> fxd_polynomials {
        {GroupElement(  -49070 , cb) ,   GroupElement(    1100283904 , cb) , GroupElement(               0 , cb)},
        {GroupElement( -101928 , cb) ,   GroupElement(    1262866432 , cb) , GroupElement(   -125023813632 , cb)},
        {GroupElement(  -89387 , cb) ,   GroupElement(    1185718272 , cb) , GroupElement(     -6375342080 , cb)},
        {GroupElement(  -56632 , cb) ,   GroupElement(     883470336 , cb) , GroupElement(    690868977664 , cb)},
        {GroupElement(  -30750 , cb) ,   GroupElement(     565043200 , cb) , GroupElement(   1670306070528 , cb)},
        {GroupElement(  -15516 , cb) ,   GroupElement(     330747904 , cb) , GroupElement(   2571108352000 , cb)},
        {GroupElement(   -7557 , cb) ,   GroupElement(     183861248 , cb) , GroupElement(   3248807215104 , cb)},
        {GroupElement(   -3621 , cb) ,   GroupElement(      99115008 , cb) , GroupElement(   3704962940928 , cb)},
        {GroupElement(   -1721 , cb) ,   GroupElement(      52363264 , cb) , GroupElement(   3992574754816 , cb)},
        {GroupElement(    -816 , cb) ,   GroupElement(      27312128 , cb) , GroupElement(   4165950504960 , cb)},
        {GroupElement(    -384 , cb) ,   GroupElement(      14049280 , cb) , GroupElement(   4267939201024 , cb)},
        {GroupElement(    -186 , cb) ,   GroupElement(       7348224 , cb) , GroupElement(   4324612636672 , cb)},
        {GroupElement(       0 , cb) ,   GroupElement(             0 , cb) , flt2fxd(1, degree*sin + scoef, cb)},
        {GroupElement(       0 , cb) ,   GroupElement(             0 , cb) , flt2fxd(-1, degree*sin + scoef, cb)}, 
        {GroupElement(     185 , cb) ,   GroupElement(       7348224 , cb) , GroupElement(  -4324629413888 , cb)},
        {GroupElement(     383 , cb) ,   GroupElement(      14049280 , cb) , GroupElement(  -4267955978240 , cb)},
        {GroupElement(     815 , cb) ,   GroupElement(      27312128 , cb) , GroupElement(  -4165967282176 , cb)},
        {GroupElement(    1720 , cb) ,   GroupElement(      52363264 , cb) , GroupElement(  -3992591532032 , cb)},
        {GroupElement(    3620 , cb) ,   GroupElement(      99115008 , cb) , GroupElement(  -3704979718144 , cb)},
        {GroupElement(    7556 , cb) ,   GroupElement(     183861248 , cb) , GroupElement(  -3248823992320 , cb)},
        {GroupElement(   15515 , cb) ,   GroupElement(     330747904 , cb) , GroupElement(  -2571125129216 , cb)},
        {GroupElement(   30749 , cb) ,   GroupElement(     565043200 , cb) , GroupElement(  -1670322847744 , cb)},
        {GroupElement(   56631 , cb) ,   GroupElement(     883470336 , cb) , GroupElement(   -690885754880 , cb)},
        {GroupElement(   89386 , cb) ,   GroupElement(    1185718272 , cb) , GroupElement(      6358564864 , cb)},
        {GroupElement(  101927 , cb) ,   GroupElement(    1262866432 , cb) , GroupElement(    125007036416 , cb)},
        {GroupElement(   49069 , cb) ,   GroupElement(    1100283904 , cb) , GroupElement(               0 , cb)},   
    };

    std::vector<GroupElement> fxd_p {GroupElement(0, ib),    GroupElement(1537, ib),    GroupElement(3075, ib),    GroupElement(4613, ib),    GroupElement(6151, ib),    GroupElement(7689, ib),    GroupElement(9227, ib),   GroupElement(10765, ib),   GroupElement(12303, ib),   GroupElement(13841, ib),   GroupElement(15379, ib),   GroupElement(16917, ib),   GroupElement(18455, ib),    GroupElement(32767, ib),
    GroupElement(-18455, ib),  GroupElement(-16918, ib),  GroupElement(-15380, ib),  GroupElement(-13842, ib),  GroupElement(-12304, ib),  GroupElement(-10766, ib),   GroupElement(-9228, ib),   GroupElement(-7690, ib),   GroupElement(-6152, ib),   GroupElement(-4614, ib),   GroupElement(-3076, ib),   GroupElement(-1538, ib)
    };

#elif defined(TANH_9_9)  // scale is not 12 (input scale 9, output scale 9)
    std::vector<std::vector<GroupElement>> fxd_polynomials 
    {
       { GroupElement(-82639 , cb) ,   GroupElement(144710656 , cb) ,  GroupElement(         0 , cb)},
       { GroupElement(-85271 , cb) ,   GroupElement(146579968 , cb) ,  GroupElement(-331874304, cb)},
       { GroupElement(-25412 , cb) ,   GroupElement(61579776  , cb) ,  GroupElement(29842997248, cb)},
       { GroupElement(-8319  , cb) ,   GroupElement(25172480  , cb) ,  GroupElement(49230118912, cb)},
       { GroupElement(-1192  , cb) ,   GroupElement(4930048   , cb) ,  GroupElement(63602163712, cb)},
       { GroupElement( 0    , cb)  ,   GroupElement(0     , cb) ,      flt2fxd(1, degree*sin + scoef, cb)     },
       
       // after x=N/2

       { GroupElement( 0    , cb)  ,   GroupElement(0       , cb) ,    flt2fxd(-1, degree*sin + scoef, cb)     },
       { GroupElement(1191  , cb)  ,   GroupElement(4930048  , cb),   GroupElement(-63602425856 , cb)},
       { GroupElement(8318  , cb)  ,   GroupElement(25172480 , cb),   GroupElement(-49230381056 , cb)},
       { GroupElement(25411 , cb)  ,   GroupElement(61579776 , cb),   GroupElement(-29843259392 , cb)},
       { GroupElement(85270 , cb)  ,   GroupElement(146579968, cb),   GroupElement(331612160    , cb)},
       { GroupElement(82638 , cb)  ,   GroupElement(144710656, cb),   GroupElement(          0  , cb)}
       
    };

    std::vector<GroupElement> fxd_p{GroupElement(0, ib),    GroupElement(355, ib),    GroupElement(710, ib),   GroupElement(1065, ib),   GroupElement(1420, ib),   GroupElement(1775, ib),  GroupElement(32767, ib),
    GroupElement(-1775, ib),  GroupElement(-1420, ib),  GroupElement(-1065, ib),   GroupElement(-710, ib),   GroupElement(-355, ib)
    };
#elif defined(TANH_8_8)
    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
       {GroupElement( -87883, cb),   GroupElement( 73233920, cb),  GroupElement(      -65536, cb)},
       {GroupElement( -74280, cb),   GroupElement( 67799296, cb),  GroupElement(   542769152, cb)},
       {GroupElement( -15013, cb),   GroupElement( 20444928, cb),  GroupElement( 10001776640, cb)},
       {GroupElement(  -6420, cb),   GroupElement( 10146304, cb),  GroupElement( 13087539200, cb)},
       {GroupElement(      0, cb),   GroupElement(        0, cb),  flt2fxd(1, degree*sin + scoef, cb)}, 
       // after x=N/2
       {GroupElement(      0, cb),   GroupElement(        0, cb),  flt2fxd(-1, degree*sin + scoef, cb)}, 
       {GroupElement(   6419, cb),   GroupElement( 10146304, cb),  GroupElement(-13087604736, cb)},
       {GroupElement(  15012, cb),   GroupElement( 20444928, cb),  GroupElement(-10001842176, cb)},
       {GroupElement(  74279, cb),   GroupElement( 67799296, cb),  GroupElement(  -542834688, cb)},
       {GroupElement(  87882, cb),   GroupElement( 73233920, cb),  GroupElement(      -65536, cb)}
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(     0, ib),
        GroupElement(   199, ib),
        GroupElement(   399, ib),
        GroupElement(   599, ib),
        GroupElement(   799, ib),
        GroupElement( 32767, ib),
        // after x=N/2
        GroupElement(  -799, ib),
        GroupElement(  -600, ib),
        GroupElement(  -400, ib),
        GroupElement(  -200, ib)
    };
#elif defined(TANH_11_11)
    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        {GroupElement( -59861, cb), GroupElement( 557158400, cb), GroupElement(       -4194304, cb)},
        {GroupElement(-104442, cb), GroupElement( 641546240, cb), GroupElement(   -39938162688, cb)},
        {GroupElement( -70219, cb), GroupElement( 511983616, cb), GroupElement(    82686509056, cb)},
        {GroupElement( -34664, cb), GroupElement( 310079488, cb), GroupElement(   369325244416, cb)},
        {GroupElement( -14913, cb), GroupElement( 160530432, cb), GroupElement(   652403015680, cb)},
        {GroupElement(  -6150, cb), GroupElement(  77594624, cb), GroupElement(   848637722624, cb)},
        {GroupElement(  -2463, cb), GroupElement(  35721216, cb), GroupElement(   967529463808, cb)},
        {GroupElement(   -985, cb), GroupElement(  16142336, cb), GroupElement(  1032385986560, cb)},
        {GroupElement(   -399, cb), GroupElement(   7270400, cb), GroupElement(  1065973972992, cb)},
        {GroupElement(      0, cb), GroupElement(         0, cb), flt2fxd(1, degree*sin + scoef, cb)},
        // after x=N/2
        {GroupElement(      0, cb), GroupElement(         0, cb), flt2fxd(-1, degree*sin + scoef, cb)},
        {GroupElement(    398, cb), GroupElement(   7270400, cb), GroupElement( -1065978167296, cb)},
        {GroupElement(    984, cb), GroupElement(  16142336, cb), GroupElement( -1032390180864, cb)},
        {GroupElement(   2462, cb), GroupElement(  35721216, cb), GroupElement(  -967533658112, cb)},
        {GroupElement(   6149, cb), GroupElement(  77594624, cb), GroupElement(  -848641916928, cb)},
        {GroupElement(  14912, cb), GroupElement( 160530432, cb), GroupElement(  -652407209984, cb)},
        {GroupElement(  34663, cb), GroupElement( 310079488, cb), GroupElement(  -369329438720, cb)},
        {GroupElement(  70218, cb), GroupElement( 511983616, cb), GroupElement(   -82690703360, cb)},
        {GroupElement( 104441, cb), GroupElement( 641546240, cb), GroupElement(    39933968384, cb)},
        {GroupElement(  59860, cb), GroupElement( 557158400, cb), GroupElement(       -4194304, cb)},
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(    0, ib),
        GroupElement(  946, ib),
        GroupElement( 1892, ib),
        GroupElement( 2839, ib),
        GroupElement( 3785, ib),
        GroupElement( 4732, ib),
        GroupElement( 5678, ib),
        GroupElement( 6625, ib),
        GroupElement( 7571, ib),
        GroupElement( 8518, ib),
        GroupElement(32767, ib),
        // after x=N/2
        GroupElement(-8518, ib),
        GroupElement(-7572, ib),
        GroupElement(-6626, ib),
        GroupElement(-5679, ib),
        GroupElement(-4733, ib),
        GroupElement(-3786, ib),
        GroupElement(-2840, ib),
        GroupElement(-1893, ib),
        GroupElement( -947, ib)
    };
#elif defined(TANH_13_13)
    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        {GroupElement( -37436, cb),  GroupElement(2177662976, cb), GroupElement(       -67108864, cb)},
        {GroupElement( -90509, cb),  GroupElement(2426101760, cb), GroupElement(   -290782707712, cb)},
        {GroupElement(-101496, cb),  GroupElement(2528968704, cb), GroupElement(   -531569311744, cb)},
        {GroupElement( -84129, cb),  GroupElement(2285068288, cb), GroupElement(    324739792896, cb)},
        {GroupElement( -59163, cb),  GroupElement(1817600000, cb), GroupElement(   2513025630208, cb)},
        {GroupElement( -37872, cb),  GroupElement(1319256064, cb), GroupElement(   5429039988736, cb)},
        {GroupElement( -22968, cb),  GroupElement( 900677632, cb), GroupElement(   8368206905344, cb)},
        {GroupElement( -13507, cb),  GroupElement( 590635008, cb), GroupElement(  10908076081152, cb)},
        {GroupElement(  -7805, cb),  GroupElement( 377094144, cb), GroupElement(  12907316248576, cb)},
        {GroupElement(  -4465, cb),  GroupElement( 236412928, cb), GroupElement(  14389079965696, cb)},
        {GroupElement(  -2540, cb),  GroupElement( 146276352, cb), GroupElement(  15443829981184, cb)},
        {GroupElement(  -1442, cb),  GroupElement(  89726976, cb), GroupElement(  16171894046720, cb)},
        {GroupElement(   -814, cb),  GroupElement(  54444032, cb), GroupElement(  16667358789632, cb)},
        {GroupElement(   -470, cb),  GroupElement(  33513472, cb), GroupElement(  16985790349312, cb)},
        // blank area b/w 2^16-1 and -2^16
        {GroupElement(      0, cb),  GroupElement(         0, cb), GroupElement(               0, cb)},
        {GroupElement(    469, cb),  GroupElement(  33513472, cb), GroupElement( -16985857458176, cb)},
        {GroupElement(    813, cb),  GroupElement(  54444032, cb), GroupElement( -16667425898496, cb)},
        {GroupElement(   1441, cb),  GroupElement(  89726976, cb), GroupElement( -16171961155584, cb)},
        {GroupElement(   2539, cb),  GroupElement( 146276352, cb), GroupElement( -15443897090048, cb)},
        {GroupElement(   4464, cb),  GroupElement( 236412928, cb), GroupElement( -14389147074560, cb)},
        {GroupElement(   7804, cb),  GroupElement( 377094144, cb), GroupElement( -12907383357440, cb)},
        {GroupElement(  13506, cb),  GroupElement( 590635008, cb), GroupElement( -10908143190016, cb)},
        {GroupElement(  22967, cb),  GroupElement( 900677632, cb), GroupElement(  -8368274014208, cb)},
        {GroupElement(  37871, cb),  GroupElement(1319256064, cb), GroupElement(  -5429107097600, cb)},
        {GroupElement(  59162, cb),  GroupElement(1817600000, cb), GroupElement(  -2513092739072, cb)},
        {GroupElement(  84128, cb),  GroupElement(2285068288, cb), GroupElement(   -324806901760, cb)},
        {GroupElement( 101495, cb),  GroupElement(2528968704, cb), GroupElement(    531502202880, cb)},
        {GroupElement(  90508, cb),  GroupElement(2426101760, cb), GroupElement(    290715598848, cb)},
        {GroupElement(  37435, cb),  GroupElement(2177662976, cb), GroupElement(       -67108864, cb)},
    };

    std::vector<GroupElement> fxd_p 
    {
        GroupElement(     0, ib),
        GroupElement(  2340, ib),
        GroupElement(  4681, ib),
        GroupElement(  7021, ib),
        GroupElement(  9362, ib),
        GroupElement( 11702, ib),
        GroupElement( 14043, ib),
        GroupElement( 16384, ib),
        GroupElement( 18724, ib),
        GroupElement( 21065, ib),
        GroupElement( 23405, ib),
        GroupElement( 25746, ib),
        GroupElement( 28086, ib),
        GroupElement( 30427, ib),
        GroupElement( 32767, ib),
        // blank area b/w 2^16-1 and -2^16 because ib=64 and actual inp bitlen = 16
        GroupElement(-32768, ib),
        GroupElement(-30428, ib),
        GroupElement(-28087, ib),
        GroupElement(-25747, ib),
        GroupElement(-23406, ib),
        GroupElement(-21066, ib),
        GroupElement(-18725, ib),
        GroupElement(-16384, ib),
        GroupElement(-14044, ib),
        GroupElement(-11703, ib),
        GroupElement( -9363, ib),
        GroupElement( -7022, ib),
        GroupElement( -4682, ib),
        GroupElement( -2341, ib)
    };
#else 
    throw std::invalid_argument("no scales selected for tanh");
#endif

    int numPoly = fxd_polynomials.size(), m = numPoly;
    fxd_p.push_back(GroupElement(-1, ib));

    return keyGenSigmoid(ib, ob, numPoly, degree, fxd_polynomials, fxd_p, rin, rout);
}                    

GroupElement evalTanh_main_wrapper(int party, GroupElement x, SplineKeyPack &k)
{
    return evalSigmoid(party, x, k);
}

std::pair<SplineKeyPack, SplineKeyPack> keyGenInvsqrt_main_wrapper(int Bin, int Bout, int scaleIn, int scaleOut,
                    GroupElement rin, GroupElement rout)
{
    // todo: add other scales
    assert((Bin == 64) && (Bout == 64));

#ifdef INVSQRT_10_9
    assert((scaleIn == 10) && (scaleOut == 9));
#elif defined(INVSQRT_12_11)
    assert((scaleIn == 12) && (scaleOut == 11));
#else 
    throw std::invalid_argument("no scales selected for invsqrt");
#endif

    int scaleCoef = 13, coefBitsize = 64;

    int ib = Bin, cb = coefBitsize, ob = Bout, sin = scaleIn, scoef = scaleCoef, sout = scaleOut;
    int degree = 2;

#if defined(INVSQRT_10_9)  // when input scale = 10, output scale = 9

    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        // for x=0 to fxd2flt(epsilon=0.1), technically input shouldn't fall here because we have condition x >= epsilon,
        // but worst case, just to be safe, use same poly as first interval of octave spline
        { GroupElement(116573, cb),  GroupElement(-99579904, cb),    GroupElement(35036069888, cb)},
        // octave spline from epsilon to end (2^16)
        { GroupElement(116573, cb),  GroupElement(-99579904, cb),    GroupElement(35036069888, cb)},
        { GroupElement( 15140, cb),  GroupElement(-27114496, cb),    GroupElement(22093496320, cb)},
        { GroupElement(  5288, cb),  GroupElement(-15048704, cb),    GroupElement(18398314496, cb)},
        { GroupElement(   957, cb),  GroupElement( -5320704, cb),    GroupElement(12937330688, cb)},
        { GroupElement(   190, cb),  GroupElement( -2032640, cb),    GroupElement( 9413066752, cb)},
        { GroupElement(    34, cb),  GroupElement(  -731136, cb),    GroupElement( 6689914880, cb)},
        { GroupElement(     6, cb),  GroupElement(  -260096, cb),    GroupElement( 4739563520, cb)},
        { GroupElement(     1, cb),  GroupElement(   -97280, cb),    GroupElement( 3407872000, cb)},
        // for negative x (input doesn't fall here, so use some dummy)
        { GroupElement(0, cb),  GroupElement(0, cb),    GroupElement(0, cb)}
    };

    std::vector<GroupElement> fxd_p{/* dummy knot x=0 for consistency */  GroupElement(0, ib),   /* actual spline starts here */  GroupElement(102, ib),     GroupElement(357, ib),     GroupElement(612, ib),    GroupElement(1122, ib),    GroupElement(2143, ib),    GroupElement(4185, ib),    GroupElement(8268, ib),   GroupElement(16435, ib),   GroupElement(32767, ib)  /* actual spline ends here  */
    };

#elif defined(INVSQRT_12_11)  //  (input scale 12, output scale 11)

    std::vector<std::vector<GroupElement>> fxd_polynomials
    {
        // for x=0 to fxd2flt(epsilon=0.1), technically input shouldn't fall here because we have condition x >= epsilon,
        // but worst case, just to be safe, use same poly as first interval of octave spline
        { GroupElement( 454375, cb),   GroupElement(-850591744, cb),   GroupElement(705582596096, cb)},
        // octave spline from epsilon to end (2^16)
        { GroupElement( 454375, cb),   GroupElement(-850591744, cb),   GroupElement(705582596096, cb)},
        { GroupElement( 191957, cb),   GroupElement(-503250944, cb),   GroupElement(590641889280, cb)},
        { GroupElement(  75342, cb),   GroupElement(-289939456, cb),   GroupElement(493099155456, cb)},
        { GroupElement(  21226, cb),   GroupElement(-136224768, cb),   GroupElement(383946588160, cb)},
        { GroupElement(   4919, cb),   GroupElement( -56926208, cb),   GroupElement(287527927808, cb)},
        { GroupElement(   1016, cb),   GroupElement( -22155264, cb),   GroupElement(210101075968, cb)},
        { GroupElement(    189, cb),   GroupElement(  -8101888, cb),   GroupElement(150374187008, cb)},
        { GroupElement(     38, cb),   GroupElement(  -3104768, cb),   GroupElement(108917686272, cb)},
        // for negative x (input doesn't fall here, so use some dummy)
        { GroupElement(0, cb),  GroupElement(0, cb),    GroupElement(0, cb)}
    };

    std::vector<GroupElement> fxd_p{/* dummy knot x=0 for consistency */  GroupElement(0, ib),   /* actual spline starts here */   GroupElement(409, ib),     GroupElement(661, ib),     GroupElement(914, ib),    GroupElement(1420, ib),    GroupElement(2431, ib),    GroupElement(4453, ib),    GroupElement(8498, ib),   GroupElement(16588, ib),   GroupElement(32768, ib) /* actual spline ends here  */
    };
  
#else 
    throw std::invalid_argument("no scales selected for invsqrt");
#endif

    int numPoly = fxd_polynomials.size(), m = numPoly;
    fxd_p.push_back(GroupElement(-1, ib));

    return keyGenSigmoid(ib, ob, numPoly, degree, fxd_polynomials, fxd_p, rin, rout);
}

GroupElement evalInvsqrt_main_wrapper(int party, GroupElement x, SplineKeyPack &k)
{
    return evalSigmoid(party, x, k);
}
