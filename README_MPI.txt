HIP - HCC
MPI version

CentOS:

// Install ROCm relative package
usermod -a -G video $LOGNAME
yum --enablerepo=extras install -y   epel-release
yum -y install \sudo \git \cmake \cmake3 \dkms \gcc-c++ \libgcc \glibc.i686 \libcxx-devel \libssh \llvm \llvm-libs \make \pciutils \pciutils-devel \pciutils-libs \rpm \rpm-build \wget \fftw \fftw-devel
yum --enablerepo=extras install -y   fakeroot
yum clean all
yum install -y   centos-release-scl
yum install -y   devtoolset-7
yum install -y   devtoolset-7-libatomic-devel devtoolset-7-elfutils-libelf-devel
yum clean all
sh -c 'echo -e "[ROCm]\nname=ROCm\nbaseurl=http://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/rocm.repo'
yum install -y   hsakmt-roct hsakmt-roct-dev hsa-rocr-dev hsa-ext-rocr-dev rocm-opencl rocm-opencl-devel rocm-smi rocm-utils rocminfo hcc atmi hip_base hip_doc hip_hc hip_samples hsa-amd-aqlprofile rocm-clang-oclcomgr
yum install -y   miopen-hip cxlactivitylogger miopengemm rocblas rocrand rocfft hipblas
sh -c 'echo -e "gfx803\ngfx900\ngfx906" >> /opt/rocm/bin/target.lst'

//install OpenMPI, UCX and Gdrcopy
yum install -y autoconf automake libtool libnuma-devel numactl-devel
yum install -y flex
INSTALL_DIR=$HOME/mpi_install
UCX_DIR=$INSTALL_DIR/ucx
OMPI_DIR=$INSTALL_DIR/ompi
GDR_DIR=$INSTALL_DIR/gdr
LD_LIBRARY_PATH=$GDR_DIR/lib64:$LD_LIBRARY_PATH
git clone https://github.com/NVIDIA/gdrcopy.git -b v1.3
cd gdrcopy
mkdir -p $GDR_DIR/lib64 $GDR_DIR/include
make PREFIX=$GDR_DIR lib install
cd ~
git clone https://github.com/openucx/ucx.git -b v1.6.0
cd ucx
./autogen.sh
mkdir build
cd build
../contrib/configure-release --prefix=$UCX_DIR --with-rocm=/opt/rocm --with-gdrcopy=$GDR_DIR
make -j$(nproc)
make -j$(nproc) install
cd ~
git clone https://github.com/open-mpi/ompi.git -b v4.0.3
cd ompi
./autogen.pl
mkdir build
cd build
../configure --prefix=$OMPI_DIR --with-ucx=$UCX_DIR
make -j$(nproc)
make -j$(nproc) install
unset UCX_DIR

//PATH setup
PATH=/opt/rh/devtoolset-7/root/usr/bin:/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:/opt/rocm/hcc/bin:${PATH:+:${PATH}}
MANPATH=/opt/rh/devtoolset-7/root/usr/share/man:${MANPATH}
INFOPATH=/opt/rh/devtoolset-7/root/usr/share/info${INFOPATH:+:${INFOPATH}}
PCP_DIR=/opt/rh/devtoolset-7/root
PERL5LIB=/opt/rh/devtoolset-7/root//usr/lib64/perl5/vendor_perl:/opt/rh/devtoolset-7/root/usr/lib/perl5:/opt/rh/devtoolset-7/root//usr/share/perl5/
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:/opt/rh/devtoolset-7/root$rpmlibdir$rpmlibdir32${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
PYTHONPATH=/opt/rh/devtoolset-7/root/usr/lib64/python$pythonvers/site-packages:/opt/rh/devtoolset-7/root/usr/lib/python$pythonvers/
LDFLAGS="-Wl,-rpath=/opt/rh/devtoolset-7/root/usr/lib64 -Wl,-rpath=/opt/rh/devtoolset-7/root/usr/lib"
PATH=$OMPI_DIR/bin:/opt/rocm/bin:${PATH:+:${PATH}}
CPLUS_INCLUDE_PATH=$OMPI_DIR/include:${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}

//note:  librdmacm-devel in centos doesn't have corresponding version for mellanox mlx5 card. so can't install successfully
//note:  need to unset UCX_DIR, or else regression test will fail.

//Build GROMACS 
git clone https://github.com/ROCmSoftwarePlatform/Gromacs.git
cd Gromacs
cd build
#make MPI version
//Please make sure mpicc and mpicxx are from the source build of openmpi which mentioned in 1.1 instead openmpi installed by "yum install"
cmake3 -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DGMX_MPI=on -DGMX_GPU=on -DGMX_GPU_USE_AMD=on -DGMX_OPENMP=on -DGMX_GPU_DETECTION_DONE=on  -DGMX_SIMD=AVX2_256  -DREGRESSIONTEST_DOWNLOAD=OFF -DCMAKE_PREFIX_PATH=/opt/rocm ..
make -j$(nproc)
make -j$(nproc) install
source /usr/local/gromacs/bin/GMXRC







UBUNTU:
// Install ROCm relative package
usermod -a -G video $LOGNAME
apt update
apt dist-upgrade
apt install libnuma-dev
reboot
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
apt update
apt install rocm-dkms rocfft
reboot

//install OpenMPI, UCX and Gdrcopy
apt update
apt install ssh kmod
apt install pkg-config
apt install libfftw3-dev rocfft
apt install -y autoconf automake libtool libnuma-dev
apt install -y flex
INSTALL_DIR=$HOME/mpi_install
INSTALL_DIR=$HOME/mpi_install
UCX_DIR=$INSTALL_DIR/ucx
OMPI_DIR=$INSTALL_DIR/ompi
GDR_DIR=$INSTALL_DIR/gdr
LD_LIBRARY_PATH=$GDR_DIR/lib64:$LD_LIBRARY_PATH

git clone https://github.com/NVIDIA/gdrcopy.git -b v1.3
cd gdrcopy
mkdir -p $GDR_DIR/lib64 $GDR_DIR/include
make PREFIX=$GDR_DIR lib install
cd $INSTALL_DIR
git clone https://github.com/openucx/ucx.git -b v1.6.0
cd ucx
./autogen.sh
mkdir build
cd build
../contrib/configure-release --prefix=$UCX_DIR --with-rocm=/opt/rocm --with-gdrcopy=$GDR_DIR
make -j$(nproc)
make -j$(nproc) install
cd $INSTALL_DIR
git clone https://github.com/open-mpi/ompi.git -b v4.0.3
cd ompi
./autogen.pl
mkdir build
cd build
../configure --prefix=$OMPI_DIR --with-ucx=$UCX_DIR
make -j$(nproc)
make -j$(nproc) install
unset UCX_DIR

//PATH setup
PATH=/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:/opt/rocm/hcc/bin:${PATH:+:${PATH}}
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH}
PATH=$OMPI_DIR/bin:/opt/rocm/bin:${PATH}
CPLUS_INCLUDE_PATH=$OMPI_DIR/include:${CPLUS_INCLUDE_PATH}

//note:  librdmacm-devel in centos doesn't have corresponding version for mellanox mlx5 card. so can't install successfully
//note:  need to unset UCX_DIR, or else regression test will fail.

//Build GROMACS 
git clone https://github.com/ROCmSoftwarePlatform/Gromacs.git
cd Gromacs
cd build
#make MPI version
//Please make sure mpicc and mpicxx are from the source build of openmpi which mentioned in 1.1 instead openmpi installed by "yum install"
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DGMX_MPI=on -DGMX_GPU=on -DGMX_GPU_USE_AMD=on -DGMX_OPENMP=on -DGMX_GPU_DETECTION_DONE=on  -DGMX_SIMD=AVX2_256  -DREGRESSIONTEST_DOWNLOAD=OFF -DCMAKE_PREFIX_PATH=/opt/rocm ..
make -j$(nproc)
make -j$(nproc) install
source /usr/local/gromacs/bin/GMXRC
