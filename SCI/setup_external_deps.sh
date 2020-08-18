ROOT=$(pwd)
wget https://github.com/microsoft/SEAL/archive/3.3.2.zip
unzip 3.3.2.zip
rm 3.3.2.zip
rm -r extern/SEAL
mv SEAL-3.3.2/ extern/SEAL
cd extern/SEAL/native/src
cmake .
make -j

cd $ROOT
wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz
tar -xvf 3.3.7.tar.gz
rm 3.3.7.tar.gz
rm -r extern/eigen
mv eigen-eigen-323c052e1731 extern/eigen
cd extern/eigen
mkdir build && cd build
cmake ../
make -j
