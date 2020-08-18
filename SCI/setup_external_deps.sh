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
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar -xvf eigen-3.3.7.tar.gz
rm eigen-3.3.7.tar.gz
rm -r extern/eigen
mv eigen-3.3.7 extern/eigen
cd extern/eigen
mkdir build && cd build
cmake ../
make -j
