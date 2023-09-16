%global debug_package %{nil}

%bcond_with check
%bcond_with fxdiv
%bcond_with pthreadpool
%bcond_with python
%bcond_with rocm
%bcond_with toolchain_clang
%bcond_without cpuinfo
%bcond_without fp16
%bcond_without psimd

%if %{with toolchain_clang}
%global toolchain clang
%global _clang_lto_cflags -flto=thin
%else
%global toolchain gcc
%endif

Summary:        An AI/ML python package
Name:           pytorch
License:        TBD
Version:        2.0.1
Release:        11%{?dist}

URL:            https://github.com/pytorch/pytorch
Source0:        %{url}/releases/download/v%{version}/%{name}-v%{version}.tar.gz
Patch0:         0001-Include-stdexcept.patch
Patch1:         0001-Include-stdint.h.patch
Patch2:         fix-cpuinfo-implicit-syscall.patch
Patch3:         do-not-force-Werror-on-Pooling.patch
Patch4:         fallback-to-cpu_kernel-for-VSX.patch

%if 0%{?fedora}
BuildRequires:  blas-static
%endif
BuildRequires:  clang-devel
BuildRequires:  cmake
%if %{with cpuinfo}
BuildRequires:  cpuinfo-devel
%endif
%if %{with fp16}
BuildRequires:  FP16-devel
%endif
%if %{with fxdiv}
BuildRequires:  FXdiv-devel
%endif
BuildRequires:  gcc-c++
BuildRequires:  lapack-static
BuildRequires:  make
%if 0%{?rhel}
BuildRequires:  openblas-static
%endif
BuildRequires:  protobuf-devel
%if %{with psimd}
BuildRequires:  psimd-devel
%endif
%if %{with pthreadpool}
BuildRequires:  pthreadpool-devel
%endif
%if %{with python}
BuildRequires:  python3.10-devel
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
%autosetup -p1 -n %{name}-v%{version}

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
        -DONNX_ML=OFF \
        -DUSE_CUDA=OFF \
        -DUSE_FBGEMM=OFF \
        -DUSE_KINETO=OFF \
        -DUSE_MKLDNN=OFF \
	-DUSE_NNPACK=OFF \
%if %{with rocm}
        -DUSE_ROCM=ON \
%else
        -DUSE_ROCM=OFF \
%endif
%if %{with cpuinfo}
        -DUSE_SYSTEM_CPUINFO=ON \
%endif
%if %{with fp16}
        -DUSE_SYSTEM_FP16=ON \
%endif
%if %{with fxdiv}
        -DUSE_SYSTEM_FXDIV=ON \
%endif
%if %{with psimd}
        -DUSE_SYSTEM_PSIMD=ON \
%endif
%if %{with pthreadpool}
        -DUSE_SYSTEM_PTHREADPOOL=ON \
%endif
        -DUSE_SYSTEM_PYBIND11=ON \
        -DUSE_SYSTEM_SLEEF=ON \
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

/usr/lib/libc10.so
/usr/lib/libtorch.so
/usr/lib/libtorch_cpu.so
/usr/lib/libtorch_global_deps.so
%ifarch x86_64
%{_libdir}/libCaffe2_perfkernels_avx.a
%{_libdir}/libCaffe2_perfkernels_avx2.a
%{_libdir}/libCaffe2_perfkernels_avx512.a
%endif

# FIXME: coming from third_party 
# cpuinfo
%if %{without cpuinfo}
%{_libdir}/libcpuinfo.a
%{_libdir}/pkgconfig/libcpuinfo.pc
%{_datadir}/cpuinfo
%{_includedir}/cpuinfo.h
%endif

%if %{without cpuinfo}
%{_libdir}/libclog.a
%else
%ifarch x86_64 aarch64
%{_libdir}/libclog.a
%endif
%endif

%ifarch x86_64 aarch64
%if %{without pthreadpool}
# pthreadpool
%{_libdir}/libpthreadpool.a
%{_includedir}/pthreadpool.h
%endif

# QNNPACK
%{_libdir}/libpytorch_qnnpack.a
%{_libdir}/libqnnpack.a
%endif

%files devel
%{_includedir}/ATen
%{_includedir}/c10
%{_includedir}/torch
%{_includedir}/caffe2

%if %{without cpuinfo}
%{_includedir}/clog.h
%else
%ifarch x86_64 aarch64
%{_includedir}/clog.h
%endif
%endif

# FP16
%if %{without fp16}
%{_includedir}/fp16.h
%{_includedir}/fp16
%else
%exclude %{_includedir}/fp16.h
%exclude %{_includedir}/fp16
%endif

%ifarch x86_64 aarch64
# FXdiv
%if %{without fxdiv}
%{_includedir}/fxdiv.h
%else
%exclude %{_includedir}/fxdiv.h
%endif
%endif

%if %{without psimd}
# psmid
%{_includedir}/psimd.h
%else
%exclude %{_includedir}/psimd.h
%endif

%ifarch x86_64 aarch64

# QNNPACK
%{_includedir}/qnnpack.h
%{_includedir}/qnnpack_func.h
%endif

%changelog
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
