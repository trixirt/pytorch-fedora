%global debug_package %{nil}
%global toolchain clang

Summary:        An AI/ML python package
Name:           pytorch
License:        TBD
Version:        2.0.1
Release:        3%{?dist}

URL:            https://github.com/pytorch/pytorch
Source0:        %{url}/releases/download/v%{version}/%{name}-v%{version}.tar.gz
Patch0:         0001-Include-stdexcept.patch
Patch1:         0001-Include-stdint.h.patch

%bcond_with python

%if 0%{?fedora}
BuildRequires:  blas-static
%endif
BuildRequires:  clang-devel
BuildRequires:  cmake
BuildRequires:  gcc-c++
BuildRequires:  lapack-static
BuildRequires:  make
%if 0%{?rhel}
BuildRequires:  openblas-static
%endif
BuildRequires:  protobuf-devel
%if %{with python}
BuildRequires:  python3-devel
%endif
BuildRequires:  python3-pybind11
BuildRequires:  python3-pyyaml
BuildRequires:  python3-typing-extensions

# TBD : add more

%description
PyTorch is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy,
and Cython to extend PyTorch when needed.

%package devel
Summary:        Headers and libraries for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description devel
This package contains the developement libraries and headers
for %{name}.

%prep
%autosetup -p1 -n %{name}-v%{version}

%build
%cmake \
        -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON \
        -DBUILD_CUSTOM_PROTOBUF=OFF \
%if %{without python}
        -DBUILD_PYTHON=OFF \
%endif
        -DBUILD_SHARED_LIBS=ON \
	-DONNX_ML=OFF \
        -DUSE_FBGEMM=OFF \
        -DUSE_KINETO=OFF \
        -DUSE_MKLDNN=OFF \
	-DUSE_NNPACK=OFF \
	-DUSE_TENSORPIPE=OFF \
        -DBUILD_CUSTOM_PROTOBUF=OFF \
        -DCAFFE2_LINK_LOCAL_PROTOBUF=OFF \
        -DUSE_SYSTEM_PYBIND11=ON \
	-DUSE_XNNPACK=OFF

%cmake_build

%install
%cmake_install

%files
%{_datadir}/ATen
%{_datadir}/cmake/ATen
%{_datadir}/cmake/Caffe2
%{_datadir}/cmake/Torch

/usr/lib/libc10.so
/usr/lib/libtorch.so
/usr/lib/libtorch_cpu.so
/usr/lib/libtorch_global_deps.so
%{_libdir}/libCaffe2_perfkernels_avx.a
%{_libdir}/libCaffe2_perfkernels_avx2.a
%{_libdir}/libCaffe2_perfkernels_avx512.a

# FIXME: coming from third_party 
# cpuinfo
%{_libdir}/libclog.a
%{_libdir}/libcpuinfo.a
%{_libdir}/pkgconfig/libcpuinfo.pc
%{_datadir}/cpuinfo

# pthreadpool
%{_libdir}/libpthreadpool.a

# QNNPACK
%{_libdir}/libpytorch_qnnpack.a
%{_libdir}/libqnnpack.a

# sleef
%{_libdir}/libsleef.a
%{_libdir}/pkgconfig/sleef.pc

%files devel
%{_includedir}/ATen
%{_includedir}/c10
%{_includedir}/torch
%{_includedir}/caffe2

# FIXME: coming from third_party
# cpuinfo
%{_includedir}/clog.h
%{_includedir}/cpuinfo.h

# FP16
%{_includedir}/fp16.h
%{_includedir}/fp16

# FXdiv
%{_includedir}/fxdiv.h

# psmid
%{_includedir}/psimd.h

# pthreadpool
%{_includedir}/pthreadpool.h

# QNNPACK
%{_includedir}/qnnpack.h
%{_includedir}/qnnpack_func.h

# sleef
%{_includedir}/sleef.h

%changelog
* Thu Aug 3 2023 Tom Rix <trix@redhat.com> - 2.0.1-3
- Add condition with python

* Mon Jul 31 2023 Jason Montleon <jmontleo@redhat.com> - 2.0.1-2
- Improvements to get building in mock

* Sat Jul 29 2023 Tom Rix <trix@redhat.com> - 2.0.1-1
- Stub something together
