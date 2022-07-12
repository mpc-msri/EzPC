/*
This is an autogenerated file, generated using the EzPC compiler.
*/

#include "emp-sh2pc/emp-sh2pc.h" 
using namespace emp;
using namespace std;
int bitlen = 32;
int party,port;
char *ip = "127.0.0.1"; 
template<typename T> 
vector<T> make_vector(size_t size) { 
return std::vector<T>(size); 
} 

template <typename T, typename... Args> 
auto make_vector(size_t first, Args... sizes) 
{ 
auto inner = make_vector<T>(sizes...); 
return vector<decltype(inner)>(first, inner); 
} 


int main(int argc, char** argv) {
parse_party_and_port(argv, &party, &port);
if(argc>3){
  ip=argv[3];
}
cout<<"Ip Address: "<<ip<<endl;
cout<<"Port: "<<port<<endl;
cout<<"Party: "<<(party==1? "CLIENT" : "SERVER")<<endl;
NetIO * io = new NetIO(party==ALICE ? nullptr : ip, port);
setup_semi_honest(io, party);


Float alpha;
if ((party == BOB)) {
cout << ("Input alpha:") << endl;
}
/* Variable to read the clear value corresponding to the input variable alpha at (2,4-2,34) */
float __tmp_in_alpha;
if ((party == BOB)) {
cin >> __tmp_in_alpha;
}
alpha = Float(__tmp_in_alpha, BOB);

Float gamma;
if ((party == BOB)) {
cout << ("Input gamma:") << endl;
}
/* Variable to read the clear value corresponding to the input variable gamma at (3,4-3,34) */
float __tmp_in_gamma;
if ((party == BOB)) {
cin >> __tmp_in_gamma;
}
gamma = Float(__tmp_in_gamma, BOB);

Float zeta;
if ((party == BOB)) {
cout << ("Input zeta:") << endl;
}
/* Variable to read the clear value corresponding to the input variable zeta at (4,4-4,33) */
float __tmp_in_zeta;
if ((party == BOB)) {
cin >> __tmp_in_zeta;
}
zeta = Float(__tmp_in_zeta, BOB);

Float theta;
if ((party == BOB)) {
cout << ("Input theta:") << endl;
}
/* Variable to read the clear value corresponding to the input variable theta at (5,4-5,34) */
float __tmp_in_theta;
if ((party == BOB)) {
cin >> __tmp_in_theta;
}
theta = Float(__tmp_in_theta, BOB);

Float mu;
if ((party == BOB)) {
cout << ("Input mu:") << endl;
}
/* Variable to read the clear value corresponding to the input variable mu at (6,4-6,31) */
float __tmp_in_mu;
if ((party == BOB)) {
cin >> __tmp_in_mu;
}
mu = Float(__tmp_in_mu, BOB);

Float beta;
if ((party == ALICE)) {
cout << ("Input beta:") << endl;
}
/* Variable to read the clear value corresponding to the input variable beta at (7,4-7,33) */
float __tmp_in_beta;
if ((party == ALICE)) {
cin >> __tmp_in_beta;
}
beta = Float(__tmp_in_beta, ALICE);

Float delta;
if ((party == ALICE)) {
cout << ("Input delta:") << endl;
}
/* Variable to read the clear value corresponding to the input variable delta at (8,4-8,34) */
float __tmp_in_delta;
if ((party == ALICE)) {
cin >> __tmp_in_delta;
}
delta = Float(__tmp_in_delta, ALICE);

Float eta;
if ((party == ALICE)) {
cout << ("Input eta:") << endl;
}
/* Variable to read the clear value corresponding to the input variable eta at (9,4-9,32) */
float __tmp_in_eta;
if ((party == ALICE)) {
cin >> __tmp_in_eta;
}
eta = Float(__tmp_in_eta, ALICE);

Float lambda;
if ((party == ALICE)) {
cout << ("Input lambda:") << endl;
}
/* Variable to read the clear value corresponding to the input variable lambda at (10,4-10,35) */
float __tmp_in_lambda;
if ((party == ALICE)) {
cin >> __tmp_in_lambda;
}
lambda = Float(__tmp_in_lambda, ALICE);

Float nu;
if ((party == ALICE)) {
cout << ("Input nu:") << endl;
}
/* Variable to read the clear value corresponding to the input variable nu at (11,4-11,31) */
float __tmp_in_nu;
if ((party == ALICE)) {
cin >> __tmp_in_nu;
}
nu = Float(__tmp_in_nu, ALICE);

float RHS;
cout << ("Input RHS:") << endl;
/* Variable to read the clear value corresponding to the input variable RHS at (12,4-12,29) */
float __tmp_in_RHS;
cin >> __tmp_in_RHS;
RHS = __tmp_in_RHS;

Float alphaSq = alpha.operator*(alpha);

Float twoAlphaGamma = Float(2., PUBLIC).operator*(alpha).operator*(gamma);

Float gammaSq = gamma.operator*(gamma);

Float zetaThetaSq = zeta.operator*(theta).operator*(theta);

Float twoZetaThetaMu = Float(2., PUBLIC).operator*(zeta).operator*(theta).operator*(mu);

Float zetaMuSq = zeta.operator*(mu).operator*(mu);

Float betaSq = beta.operator*(beta);

Float betaDelta = beta.operator*(delta);

Float deltaSq = delta.operator*(delta);

Float etaLambdaSq = eta.operator*(lambda).operator*(lambda);

Float etaLambdaNu = eta.operator*(lambda).operator*(nu);

Float etaNuSq = eta.operator*(nu).operator*(nu);

Float LHS = alphaSq.operator*(betaSq);
LHS = LHS.operator-(twoAlphaGamma.operator*(betaDelta));
LHS = LHS.operator+(gammaSq.operator*(deltaSq));
LHS = LHS.operator+(zetaThetaSq.operator*(etaLambdaSq));
LHS = LHS.operator-(twoZetaThetaMu.operator*(etaLambdaNu));
LHS = LHS.operator+(zetaMuSq.operator*(etaNuSq));
cout << ("Value of ((LHS) <_baba (<public ~> baba> (RHS))) ?_baba (<public ~> baba> (true)) : (<public ~> baba> (false)):") << endl;
cout << ( If(LHS.less_than(Float(RHS, PUBLIC)), Bit(1, PUBLIC), Bit(0, PUBLIC)).reveal<bool>(PUBLIC)) << endl;


finalize_semi_honest();
delete io; 
 
return 0;
}

