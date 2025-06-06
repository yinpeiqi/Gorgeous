bigann_10K="/data/wsl/SSD_Based_Plan/bigann/BIGANN_10K_R48_L128_B0.0003_disk.index"
bigann_1M="/data/wsl/SSD_Based_Plan/bigann/BIGANN1M_R48_L128_B0.03_disk.index"
bigann_10M="/data/wsl/SSD_Based_Plan/bigann/BIGANN10M_R48_L128_B0.03_M32_disk.index"
bigann_10M_R48="/data/wsl/SSD_Based_Plan/bigann/BIGANN_10M_R48_L128_B0.3_disk.index"
bigann_10M_R32="/data/wsl/SSD_Based_Plan/bigann/BIGANN_10M_R32_L96_B0.3_disk.index"
deep_10M="/data/wsl/diskann_re/build/tests/index/DEEP_10M_R48_L128_B0.225_disk.index_cp"
ssnpp_10M="/data/wsl/SSD_Based_Plan/bigann/SSNPP_ORG_R48_L128_B0.6_M32_disk.index"
sift_10M_R32="/data/wsl/SSD_Based_Plan/bigann/SIFT_10M_R32_L96_B0.3_disk.index"

index=$deep_10M
dim=96
data_type=float
nd=10000000
mkdir -p ./gpfile
gp_file=./gpfile/deep_10m_r48_rb1
lock_nums=0
./build/partitioner --data_type $data_type --index_file $index --gp_file $gp_file -N $nd -D $dim -L 16 --lock_nums $lock_nums --visual true --cut 3
#>> ${gp_file}.log

