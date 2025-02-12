mkdir -p data/shd
cd data/shd

# spiking heidelberg digits
wget https://zenkelab.org/datasets/shd_test.h5.zip
wget https://zenkelab.org/datasets/shd_train.h5.zip
unzip shd_test.h5.zip
unzip shd_train.h5.zip

rm *.zip
cd -

