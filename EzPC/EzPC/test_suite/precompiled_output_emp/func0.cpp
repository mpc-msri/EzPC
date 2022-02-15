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

Integer hello(Integer a){
/* Temporary variable for sub-expression on source location: (36,10-36,12) */
Integer __tac_var1 = Integer(bitlen,  (uint32_t)5, PUBLIC);
/* Temporary variable for sub-expression on source location: (36,8-36,12) */
Integer __tac_var2 = a.operator+(__tac_var1);
return __tac_var2;
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


Integer b = Integer(bitlen,  (uint32_t)10, PUBLIC);

Integer c = hello(b);
cout << ("Value of c:") << endl;
cout << (c.reveal<uint32_t>(BOB)) << endl;


finalize_semi_honest();
delete io; 
 
return 0;
}

