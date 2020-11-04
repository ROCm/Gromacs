HIP - HCC
MPI version

CentOS:

// Install ROCm relative package
sudo usermod -a -G video $LOGNAME
sudo yum --enablerepo=extras install -y   epel-release
sudo yum -y install \sudo \git \cmake \cmake3 \dkms \gcc-c++ \libgcc \glibc.i686 \libcxx-devel \libssh \llvm \llvm-libs \make \pciutils \pciutils-devel \pciutils-libs \rpm \rpm-build \wget \fftw \fftw-devel
sudo yum --enablerepo=extras install -y   fakeroot
sudo yum clean all
sudo yum install -y   centos-release-scl
sudo yum install -y   devtoolset-7
sudo yum install -y   devtoolset-7-libatomic-devel devtoolset-7-elfutils-libelf-devel
sudo yum clean all
sudo sh -c 'echo -e "[ROCm]\nname=ROCm\nbaseurl=http://repo.radeon.com/rocm/yum/rpm\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/rocm.repo'
sudo yum install -y   hsakmt-roct hsakmt-roct-dev hsa-rocr-dev hsa-ext-rocr-dev rocm-opencl rocm-opencl-devel rocm-smi rocm-utils rocminfo hcc atmi hip_base hip_doc hip_hc hip_samples hsa-amd-aqlprofile rocm-clang-oclcomgr rocfft
sudo yum install -y   miopen-hip cxlactivitylogger miopengemm rocblas rocrand rocfft hipblas
sudo sh -c 'echo -e "gfx803\ngfx900\ngfx906\ngfx908" >> /opt/rocm/bin/target.lst'

//PATH setup
PATH=/opt/rh/devtoolset-7/root/usr/bin:/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:/opt/rocm/hcc/bin:${PATH:+:${PATH}}
MANPATH=/opt/rh/devtoolset-7/root/usr/share/man:${MANPATH}
INFOPATH=/opt/rh/devtoolset-7/root/usr/share/info${INFOPATH:+:${INFOPATH}}
PCP_DIR=/opt/rh/devtoolset-7/root
PERL5LIB=/opt/rh/devtoolset-7/root//usr/lib64/perl5/vendor_perl:/opt/rh/devtoolset-7/root/usr/lib/perl5:/opt/rh/devtoolset-7/root//usr/share/perl5/
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:/opt/rh/devtoolset-7/root$rpmlibdir$rpmlibdir32${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
PYTHONPATH=/opt/rh/devtoolset-7/root/usr/lib64/python$pythonvers/site-packages:/opt/rh/devtoolset-7/root/usr/lib/python$pythonvers/
LDFLAGS="-Wl,-rpath=/opt/rh/devtoolset-7/root/usr/lib64 -Wl,-rpath=/opt/rh/devtoolset-7/root/usr/lib"
PATH=/opt/rocm/bin:${PATH:+:${PATH}}

//Build GROMACS 
git clone https://github.com/ROCmSoftwarePlatform/Gromacs.git
cd Gromacs
cd build
#make MPI version
cmake3 -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DGMX_MPI=off -DGMX_GPU=on -DGMX_GPU_USE_AMD=on -DGMX_OPENMP=on -DGMX_GPU_DETECTION_DONE=on   -DGMX_SIMD=AVX2_256   -DREGRESSIONTEST_DOWNLOAD=OFF -DCMAKE_PREFIX_PATH=/opt/rocm ..
make -j$(nproc)
sudo make -j$(nproc) install
source /usr/local/gromacs/bin/GMXRC








UBUNTU:
// Install ROCm relative package
sudo usermod -a -G video $LOGNAME
sudo apt update
sudo apt dist-upgrade
sudo apt install libnuma-dev
sudo reboot
wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms rocfft
sudo reboot

//PATH setup
PATH=/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/bin:/opt/rocm/hcc/bin:${PATH}
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH}
PATH=/opt/rocm/bin:${PATH}

//Build GROMACS 
git clone https://github.com/ROCmSoftwarePlatform/Gromacs.git
cd Gromacs
cd build
#make MPI version
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DGMX_MPI=off -DGMX_GPU=on -DGMX_GPU_USE_AMD=on -DGMX_OPENMP=on -DGMX_GPU_DETECTION_DONE=on   -DGMX_SIMD=AVX2_256   -DREGRESSIONTEST_DOWNLOAD=OFF -DCMAKE_PREFIX_PATH=/opt/rocm ..
make -j$(nproc)
sudo make -j$(nproc) install
source /usr/local/gromacs/bin/GMXRC

