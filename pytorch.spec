%global debug_package %{nil}

%bcond_with check
%bcond_with fxdiv
%bcond_with gloo
%bcond_with new
%bcond_with pthreadpool

%bcond_with rocm
%bcond_with toolchain_clang

%if %{with toolchain_clang}
%global toolchain clang
%global _clang_lto_cflags -flto=thin
%else
%global toolchain gcc
%endif

Summary:        An AI/ML python package
Name:           pytorch
License:        TBD
URL:            https://github.com/pytorch/pytorch

%if %{with new}
%global commit0 1841d54370d167365d15f0ac78efc2c56cdf43ab
%global shortcommit0 %(c=%{commit0}; echo ${c:0:7})
Version:        2.1.0
Release:        rc5%{?dist}
Source0:        %{url}/archive/%{commit0}/%{name}-%{shortcommit0}.tar.gz
%bcond_without python
%else
Version:        2.0.1
Release:        12%{?dist}
Source0:        %{url}/releases/download/v%{version}/%{name}-v%{version}.tar.gz
Patch0:         0001-Include-stdexcept.patch
Patch1:         0001-Include-stdint.h.patch
Patch2:         fix-cpuinfo-implicit-syscall.patch
Patch3:         do-not-force-Werror-on-Pooling.patch
Patch4:         fallback-to-cpu_kernel-for-VSX.patch
%bcond_with python
%endif

%if 0%{?fedora}
BuildRequires:  blas-static
%else
BuildRequires:  openblas-static
%endif
BuildRequires:  clang-devel
BuildRequires:  cmake
BuildRequires:  cpuinfo-devel
BuildRequires:  fmt-devel
BuildRequires:  FP16-devel
%if %{with fxdiv}
BuildRequires:  FXdiv-devel
%endif
%if %{with gloo}
BuildRequires:  gloo-devel
%endif
BuildRequires:  gcc-c++
BuildRequires:  lapack-static
BuildRequires:  make
BuildRequires:  onnx-devel
BuildRequires:  protobuf-devel
BuildRequires:  protobuf-devel
BuildRequires:  psimd-devel
%if %{with pthreadpool}
BuildRequires:  pthreadpool-devel
%endif
%if %{with python}
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools
%endif
BuildRequires:  python3-pybind11
BuildRequires:  python3-pyyaml
BuildRequires:  python3-typing-extensions
BuildRequires:  sleef-devel

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
%if %{with new}
%autosetup -p1 -n %{name}-%{commit0}
%else
%autosetup -p1 -n %{name}-v%{version}
%endif


%build
%if 0%{?rhel}
ulimit -n 2048
%endif

%if %{with rocm}
# Radeon RX 7600
export PYTORCH_ROCM_ARCH=gfx1102
%endif

%cmake \
        -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON \
        -DBUILD_CUSTOM_PROTOBUF=OFF \
%if %{without python}
        -DBUILD_PYTHON=OFF \
%endif
        -DBUILD_SHARED_LIBS=ON \
%if %{with check}
	-DBUILD_TEST=ON \
%endif
        -DCAFFE2_LINK_LOCAL_PROTOBUF=OFF \
        -DHAVE_SOVERSION=ON \
        -DUSE_CUDA=OFF \
	-DUSE_DISTRIBUTED=ON \
        -DUSE_FBGEMM=OFF \
	-DUSE_ITT=OFF \
        -DUSE_KINETO=OFF \
        -DUSE_MKLDNN=OFF \
	-DUSE_NNPACK=OFF \
	-DUSE_OPENMP=OFF \
	-DUSE_PYTORCH_QNNPACK=OFF \
	-DUSE_QNNPACK=OFF \
%if %{with rocm}
        -DUSE_ROCM=ON \
%else
        -DUSE_ROCM=OFF \
%endif
        -DUSE_SYSTEM_CPUINFO=ON \
        -DUSE_SYSTEM_FP16=ON \
%if %{with fxdiv}
        -DUSE_SYSTEM_FXDIV=ON \
%endif
%if %{with gloo}
        -DUSE_SYSTEM_GLOO=ON \
%endif
        -DUSE_SYSTEM_ONNX=ON \
        -DUSE_SYSTEM_PSIMD=ON \
%if %{with pthreadpool}
        -DUSE_SYSTEM_PTHREADPOOL=ON \
%endif
        -DUSE_SYSTEM_PYBIND11=ON \
        -DUSE_SYSTEM_SLEEF=ON \
        -DUSE_SYSTEM_ZSTD=ON \
	-DUSE_TENSORPIPE=OFF \
	-DUSE_XNNPACK=OFF

%cmake_build

%if %{with check}
%check
%ctest
%endif

%install
%cmake_install

%files
%{_datadir}/ATen
%{_datadir}/cmake/ATen
%{_datadir}/cmake/Caffe2
%{_datadir}/cmake/Torch

/usr/lib/libc10.so.*
/usr/lib/libtorch.so.*
/usr/lib/libtorch_cpu.so.*
/usr/lib/libtorch_global_deps.so.*
%ifarch x86_64
%{_libdir}/libCaffe2_perfkernels_avx.a
%{_libdir}/libCaffe2_perfkernels_avx2.a
%{_libdir}/libCaffe2_perfkernels_avx512.a
%endif

%files devel
%{_includedir}/ATen
%{_includedir}/c10
%{_includedir}/torch
%{_includedir}/caffe2
/usr/lib/libc10.so
/usr/lib/libtorch.so
/usr/lib/libtorch_cpu.so
/usr/lib/libtorch_global_deps.so

%ifarch x86_64 aarch64

%if %{without fxdiv}
%{_includedir}/fxdiv.h
%endif

%if %{without pthreadpool}
%{_libdir}/libpthreadpool.a
%{_includedir}/pthreadpool.h
%endif

%endif

%changelog
* Fri Sep 22 2023 Tom Rix <trix@redhat.com> - 2.0.1-13
- Try rawhide bound gloo

* Thu Sep 21 2023 Tom Rix <trix@redhat.com> - 2.0.1-12
- Use so version
- remove option to not use system cpuinfo, fp16, psimd
- exclude some things that will never be packaged

* Sat Sep 16 2023 Tom Rix <trix@redhat.com> - 2.0.1-11
- Try rawhide bound fxdiv package
- Try rawhide bound pthreadpool package
- Use fp16 package
- Add a check option

* Sat Sep 09 2023 Tom Rix <trix@redhat.com> - 2.0.1-10
- Try rawhide bound fp16 package
- Use psimd package

* Thu Aug 17 2023 Tom Rix <trix@redhat.com> - 2.0.1-9
- Try rawhide bound psimd package

* Wed Aug 9 2023 Tom Rix <trix@redhat.com> - 2.0.1-8
- Fix clod.h error
- Stub in rocm to work out integration

* Sat Aug 05 2023 Jason Montleon <jmontleo@redhat.com> - 2.0.1-7
- Fix ppc64le builds
- Add option to build with gcc or clang, default to gcc for sake of ppc64le

* Sat Aug 05 2023 Tom Rix <trix@redhat.com> - 2.0.1-6
- Use cpuinfo, sleef

* Fri Aug 04 2023 Jason Montleon <jmontleo@redhat.com> - 2.0.1-5
- Adjust architecture specific packaging
- Fix cpuinfo build failures with clang 16 on s390x and ppc64le

* Thu Aug 03 2023 Jason Montleon <jmontleo@redhat.com> - 2.0.1-4
- Conditionalize file list to fix aarch64 builds
- Set _clang_lto_cflags to drastically improve older EL9 build times.
- Increase open files to work around older EL9 ld.

* Thu Aug 3 2023 Tom Rix <trix@redhat.com> - 2.0.1-3
- Add condition with python

* Mon Jul 31 2023 Jason Montleon <jmontleo@redhat.com> - 2.0.1-2
- Improvements to get building in mock

* Sat Jul 29 2023 Tom Rix <trix@redhat.com> - 2.0.1-1
- Stub something together
