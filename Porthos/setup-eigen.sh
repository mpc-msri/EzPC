#!/bin/sh

#Get stable Eigen 3.3.7
wget http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz

#untar it
tar -xvf 3.3.7.tar.gz
rm 3.3.7.tar.gz

#rename the eigen directory
mv eigen-eigen-323c052e1731 lib_eigen

echo "Done Eigen setup."


