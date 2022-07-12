
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include "secfloat.h"

using namespace std ;
using namespace sci ;


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

FPArray alpha(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input alpha:") << endl ;

}
float *__tmp_in_alpha = new float[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_alpha[0];
}
alpha = __fp_op->input(ALICE, 1, __tmp_in_alpha) ;

delete[] __tmp_in_alpha ;

FPArray gamma(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input gamma:") << endl ;

}
float *__tmp_in_gamma = new float[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_gamma[0];
}
gamma = __fp_op->input(ALICE, 1, __tmp_in_gamma) ;

delete[] __tmp_in_gamma ;

FPArray zeta(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input zeta:") << endl ;

}
float *__tmp_in_zeta = new float[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_zeta[0];
}
zeta = __fp_op->input(ALICE, 1, __tmp_in_zeta) ;

delete[] __tmp_in_zeta ;

FPArray theta(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input theta:") << endl ;

}
float *__tmp_in_theta = new float[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_theta[0];
}
theta = __fp_op->input(ALICE, 1, __tmp_in_theta) ;

delete[] __tmp_in_theta ;

FPArray mu(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input mu:") << endl ;

}
float *__tmp_in_mu = new float[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_mu[0];
}
mu = __fp_op->input(ALICE, 1, __tmp_in_mu) ;

delete[] __tmp_in_mu ;

FPArray beta(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input beta:") << endl ;

}
float *__tmp_in_beta = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_beta[0];
}
beta = __fp_op->input(BOB, 1, __tmp_in_beta) ;

delete[] __tmp_in_beta ;

FPArray delta(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input delta:") << endl ;

}
float *__tmp_in_delta = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_delta[0];
}
delta = __fp_op->input(BOB, 1, __tmp_in_delta) ;

delete[] __tmp_in_delta ;

FPArray eta(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input eta:") << endl ;

}
float *__tmp_in_eta = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_eta[0];
}
eta = __fp_op->input(BOB, 1, __tmp_in_eta) ;

delete[] __tmp_in_eta ;

FPArray lambda(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input lambda:") << endl ;

}
float *__tmp_in_lambda = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_lambda[0];
}
lambda = __fp_op->input(BOB, 1, __tmp_in_lambda) ;

delete[] __tmp_in_lambda ;

FPArray nu(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input nu:") << endl ;

}
float *__tmp_in_nu = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_nu[0];
}
nu = __fp_op->input(BOB, 1, __tmp_in_nu) ;

delete[] __tmp_in_nu ;

float RHS ;

cout << ("Input RHS:") << endl ;

float *__tmp_in_RHS = new float[1] ;

cin >> __tmp_in_RHS[0];
RHS = __tmp_in_RHS[0] ;

delete[] __tmp_in_RHS ;

FPArray alphaSq = __fp_op->mul(alpha, alpha) ;

FPArray twoAlphaGamma = __fp_op->mul(__fp_op->mul(__public_float_to_baba(2.), alpha), gamma) ;

FPArray gammaSq = __fp_op->mul(gamma, gamma) ;

FPArray zetaThetaSq = __fp_op->mul(__fp_op->mul(zeta, theta), theta) ;

FPArray twoZetaThetaMu = __fp_op->mul(__fp_op->mul(__fp_op->mul(__public_float_to_baba(2.), zeta), theta), mu) ;

FPArray zetaMuSq = __fp_op->mul(__fp_op->mul(zeta, mu), mu) ;

FPArray betaSq = __fp_op->mul(beta, beta) ;

FPArray betaDelta = __fp_op->mul(beta, delta) ;

FPArray deltaSq = __fp_op->mul(delta, delta) ;

FPArray etaLambdaSq = __fp_op->mul(__fp_op->mul(eta, lambda), lambda) ;

FPArray etaLambdaNu = __fp_op->mul(__fp_op->mul(eta, lambda), nu) ;

FPArray etaNuSq = __fp_op->mul(__fp_op->mul(eta, nu), nu) ;

FPArray LHS = __fp_op->mul(alphaSq, betaSq) ;

LHS = __fp_op->sub(LHS, __fp_op->mul(twoAlphaGamma, betaDelta)) ;

LHS = __fp_op->add(LHS, __fp_op->mul(gammaSq, deltaSq)) ;

LHS = __fp_op->add(LHS, __fp_op->mul(zetaThetaSq, etaLambdaSq)) ;

LHS = __fp_op->sub(LHS, __fp_op->mul(twoZetaThetaMu, etaLambdaNu)) ;

LHS = __fp_op->add(LHS, __fp_op->mul(zetaMuSq, etaNuSq)) ;

cout << "Value of ((LHS) <_baba (<public ~> baba> (RHS))) ?_baba (<public ~> baba> (true)) : (<public ~> baba> (false)) : " ;

__bool_pub = __bool_op->output(PUBLIC, __bool_op->if_else(__fp_op->LT(LHS, __public_float_to_baba(RHS)), __public_bool_to_boolean(1), __public_bool_to_boolean(0))) ;

cout << ((bool)__bool_pub.data[0]) << endl ;

return 0;
}

