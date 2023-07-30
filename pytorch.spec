%global toolchain clang

Summary:        An AI/ML python package
Name:           pytorch
License:        TBD
Version:        2.0.1
Release:        1%{?dist}

URL:            https://github.com/pytorch/pytorch
Source0:        %{url}/releases/download/v%{version}/%{name}-v%{version}.tar.gz
Patch0:         0001-Include-stdexcept.patch
Patch1:         0001-Include-stdint.h.patch

ExclusiveArch:  x86_64

BuildRequires:  blas-static
BuildRequires:  cmake
BuildRequires:  gcc-c++
BuildRequires:  lapack-static
BuildRequires:  make
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
	-DBUILD_PYTHON=OFF \
        -DBUILD_SHARED_LIBS=ON \
	-DONNX_ML=OFF \
        -DUSE_FBGEMM=OFF \
        -DUSE_KINETO=OFF \
        -DUSE_MKLDNN=OFF \
	-DUSE_NNPACK=OFF \
	-DUSE_TENSORPIPE=OFF \
	-DUSE_XNNPACK=OFF

%cmake_build

%install
%cmake_install

%files

%files devel
%{_includedir}/torch


%changelog
* Sat Jul 29 2023 Tom Rix <trix@redhat.com> - 2.0.1-1
- Stub something together

