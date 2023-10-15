%global pypi_name torch
%global pypi_version 2.1.0

# Where the src comes from
%global forgeurl https://github.com/pytorch/pytorch

# To check between pyproject_ and py3_ building
# current pyproject problem with mock
# + /usr/bin/python3 -Bs /usr/lib/rpm/redhat/pyproject_wheel.py /builddir/build/BUILD/pytorch-v2.1.0/pyproject-wheeldir
# /usr/bin/python3: No module named pip
# Adding pip to build requires does not fix
%bcond_with pyproject

# So pre releases can be tried
%bcond_with gitcommit
%if %{with gitcommit}
# The top of the 2.1.0 branch - update to whatever..
%global commit0 1841d54370d167365d15f0ac78efc2c56cdf43ab
%global shortcommit0 %(c=%{commit0}; echo ${c:0:7})
%endif

# Check if static blas,lapack are really needed
# And this is the error :
# CMake Error at /usr/lib64/cmake/lapack-3.11.0/lapack-targets.cmake:103 (message):
#  The imported target "blas" references the file
#
#     "/usr/lib64/libblas.a"
#
#  but this file does not exist.  Possible reasons include:
# ...
#
# See BZ 2243823
%bcond_without static_blas

Name:           python-%{pypi_name}
Version:        2.1.0
Release:        4%{?dist}
Summary:        An AI/ML python package
License:        BSD-3-Clause and MIT

URL:            https://pytorch.org/
%if %{with gitcommit}
Source0:        %{forgeurl}/archive/%{commit0}/pytorch-%{shortcommit0}.tar.gz
Source1:        pyproject.toml
%else
Source0:        %{forgeurl}/releases/download/v%{version}/pytorch-v%{version}.tar.gz
%endif

# Misc cmake changes that would be difficult to upstream
# * Use the system fmt
# * Remove foxi use
# * Remove warnings/errors for clang 17
# * fxdiv is not a library on Fedora
Patch0:         0001-Prepare-pytorch-cmake-for-fedora.patch
# Use Fedora's fmt
Patch1:         0002-Regenerate-flatbuffer-header.patch
# https://github.com/pytorch/pytorch/pull/111048
Patch2:         0003-Stub-in-kineto-ActivityType.patch
# PyTorch has not fully baked 3.12 support because 3.12 is so new
Patch3:         0004-torch-python-3.12-changes.patch
# Short circuit looking for things that can not be downloade by mock
Patch4:         0005-disable-submodule-search.patch
# A Fedora libblas.a problem of undefined symbol
# libtorch_cpu.so: undefined symbol: _gfortran_stop_string
Patch5:         0001-torch-unresolved-syms-need-gfortran.patch
# Fedora requires versioned so's
Patch6:         0001-pytorch-use-SO-version-by-default.patch

# Limit to these because they are well behaved with clang
ExclusiveArch:  x86_64 aarch64
%global toolchain clang

%if 0%{?fedora}
%if %{with static_blas}
BuildRequires:  blas-static
%else
BuildRequires:  blas-devel
%endif
%else
BuildRequires:  openblas-static
%endif
BuildRequires:  clang-devel
BuildRequires:  cmake
BuildRequires:  cpuinfo-devel
BuildRequires:  fmt-devel
BuildRequires:  flatbuffers-devel
BuildRequires:  FP16-devel
BuildRequires:  fxdiv-devel
BuildRequires:  gcc-c++
BuildRequires:  gcc-gfortran
BuildRequires:  gloo-devel
%if %{with static_blas}
BuildRequires:  lapack-static
%else
BuildRequires:  lapack-devel
%endif
BuildRequires:  ninja-build
BuildRequires:  onnx-devel
BuildRequires:  pocketfft-devel
BuildRequires:  protobuf-devel
BuildRequires:  pthreadpool-devel
BuildRequires:  psimd-devel
BuildRequires:  python3-numpy
BuildRequires:  python3-pybind11
BuildRequires:  python3-pyyaml
BuildRequires:  python3-typing-extensions
BuildRequires:  sleef-devel
BuildRequires:  valgrind-devel
BuildRequires:  xnnpack-devel

BuildRequires:  python3-devel
BuildRequires:  python3dist(filelock)
BuildRequires:  python3dist(fsspec)
BuildRequires:  python3dist(jinja2)
BuildRequires:  python3dist(networkx)
BuildRequires:  python3dist(setuptools)
BuildRequires:  python3dist(sympy)
BuildRequires:  python3dist(typing-extensions)
BuildRequires:  python3dist(sphinx)

Provides:       bundled(miniz) = 2.1.0

%description
PyTorch is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy,
and Cython to extend PyTorch when needed.

%package -n     python3-%{pypi_name}
Summary:        %{summary}

Requires:       python3dist(filelock)
Requires:       python3dist(fsspec)
Requires:       python3dist(jinja2)
Requires:       python3dist(networkx)
Requires:       python3dist(setuptools)
Requires:       python3dist(sympy)
Requires:       python3dist(typing-extensions)
%description -n python3-%{pypi_name}
PyTorch is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy,
and Cython to extend PyTorch when needed.

%prep
%if %{with gitcommit}
%autosetup -p1 -n pytorch-%{commit0}

# Remove bundled egg-info
rm -rf %{pypi_name}.egg-info
# Overwrite with a git checkout of the pyproject.toml
cp %{SOURCE1} .
%else
%autosetup -p1 -n pytorch-v%{version}

# Release comes fully loaded with third party src
# Remove what we can
#
# For 2.1 this is all but miniz-2.1.0
# Instead of building as a library, caffe2 reaches into
# the third_party dir to compile the file.
# mimiz is licensed MIT
# https://github.com/richgel999/miniz/blob/master/LICENSE
mv third_party/miniz-2.1.0 .
#
# setup.py depends on this script
mv third_party/build_bundled.py .
# Remove everything
rm -rf third_party/*
# Put stuff back
mv build_bundled.py third_party
mv miniz-2.1.0 third_party

#
# Fake out pocketfft, and system header will be used
mkdir third_party/pocketfft
#
# Use the system valgrind headers
mkdir third_party/valgrind-headers
cp %{_includedir}/valgrind/* third_party/valgrind-headers

%endif

# Broken, looking for cmake,ninja
# %generate_buildrequires
# %pyproject_buildrequires

%build

# For debugging setup.py
# export SETUPTOOLS_SCM_DEBUG=1

export CMAKE_FIND_PACKAGE_PREFER_CONFIG=ON
export BUILD_CUSTOM_PROTOBUF=OFF
export BUILD_SHARED_LIBS=ON
export BUILD_TEST=OFF
export CAFFE2_LINK_LOCAL_PROTOBUF=OFF
export USE_CUDA=OFF
export USE_DISTRIBUTED=OFF
export USE_FBGEMM=OFF
export USE_ITT=OFF
export USE_KINETO=OFF
export USE_LITE_INTERPRETER_PROFILER=OFF
export USE_MKLDNN=OFF
export USE_NNPACK=OFF
export USE_NUMPY=ON
export USE_OPENMP=OFF
export USE_PYTORCH_QNNPACK=OFF
export USE_QNNPACK=OFF
export USE_ROCM=OFF
export USE_SYSTEM_LIBS=ON
export USE_TENSORPIPE=OFF
export USE_XNNPACK=ON

%if %{with pyproject}
%pyproject_wheel
%else
%py3_build
%endif

%install
%if %{with pyproject}
%pyproject_install
%else
%py3_install
%endif

%files -n python3-%{pypi_name}
%license LICENSE
%doc README.md
%{_bindir}/convert-caffe2-to-onnx
%{_bindir}/convert-onnx-to-caffe2
%{_bindir}/torchrun
%{python3_sitearch}/functorch/
%{python3_sitearch}/torch/
%{python3_sitearch}/torchgen/
%if %{without pyproject}
%{python3_sitearch}/torch*.egg-info/
%endif

%changelog
* Sat Oct 14 2023 Tom Rix <trix@redhat.com> - 2.1.0-4
- Use gloo, xnnpack
- Find missing build_bundled.py
- Add pyproject option

* Thu Oct 12 2023 Tom Rix <trix@redhat.com> - 2.1.0-3
- Address review comments
- Force so versioning on

* Mon Oct 9 2023 Tom Rix <trix@redhat.com> - 2.1.0-2
- Use the 2.1 release
- Reduce USE_SYSTEM_LIBS to parts
- Remove almost all of third_party/
- Remove py2rpm generated noise

* Sat Sep 30 2023 Tom Rix <trix@redhat.com> - 2.1.0-1
- Initial package.

