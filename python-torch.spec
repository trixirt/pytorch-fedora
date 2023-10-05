%global pypi_name torch
%global pypi_version 2.1.0

# The top of the 2.1.0 branch
%global commit0 1841d54370d167365d15f0ac78efc2c56cdf43ab
%global shortcommit0 %(c=%{commit0}; echo ${c:0:7})

# So build lto takes forever, disable
%global _lto_cflags %{nil}

Name:           python-%{pypi_name}
Version:        2.1.0
Release:        1%{?dist}
Summary:        An AI/ML python package

License:        BSD-3-Clause

URL:            https://github.com/pytorch/pytorch
Source0:        %{url}/archive/%{commit0}/pytorch-%{shortcommit0}.tar.gz
Source1:        pyproject.toml

Patch0:         0001-Prepare-pytorch-cmake-for-fedora.patch
Patch1:         0002-Regenerate-flatbuffer-header.patch
Patch2:         0003-Stub-in-kineto-ActivityType.patch
Patch3:         0004-torch-python-3.12-changes.patch
Patch4:         0005-disable-submodule-search.patch
Patch5:         0007-use-system-xnn-pain.patch

# Limit to these because they are well behaved with clang
ExclusiveArch:  x86_64 aarch64
%global toolchain clang

%if 0%{?fedora}
BuildRequires:  blas-static
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
BuildRequires:  lapack-static
BuildRequires:  ninja-build
BuildRequires:  onnx-devel
BuildRequires:  pocketfft-devel
BuildRequires:  protobuf-devel
BuildRequires:  psimd-devel
BuildRequires:  pthreadpool-devel
BuildRequires:  python3-pybind11
BuildRequires:  python3-pyyaml
BuildRequires:  python3-typing-extensions
BuildRequires:  sleef-devel

BuildRequires:  python3-devel
BuildRequires:  python3dist(filelock)
BuildRequires:  python3dist(fsspec)
BuildRequires:  python3dist(jinja2)
BuildRequires:  python3dist(networkx)
BuildRequires:  python3dist(setuptools)
BuildRequires:  python3dist(sympy)
BuildRequires:  python3dist(typing-extensions)
BuildRequires:  python3dist(sphinx)

%description
PyTorch is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy,
and Cython to extend PyTorch when needed.

%package -n     python3-%{pypi_name}
Summary:        %{summary}
%{?python_provide:%python_provide python3-%{pypi_name}}

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

%package -n python-%{pypi_name}-doc
Summary:        torch documentation
%description -n python-%{pypi_name}-doc
Documentation for torch

%prep
%autosetup -p1 -n pytorch-%{commit0}

# Remove bundled egg-info
rm -rf %{pypi_name}.egg-info
# Overwrite with a git checkout of the pyproject.toml
cp %{SOURCE1} .

%build

export SETUPTOOLS_SCM_DEBUG=1

export CC=clang
export CXX=clang++

export CMAKE_FIND_PACKAGE_PREFER_CONFIG=ON
export BUILD_CUSTOM_PROTOBUF=OFF
export BUILD_SHARED_LIBS=ON
export BUILD_TEST=OFF
export CAFFE2_LINK_LOCAL_PROTOBUF=OFF
export HAVE_SOVERSION=ON
export USE_CUDA=OFF
export USE_DISTRIBUTED=OFF
export USE_FBGEMM=OFF
export USE_ITT=OFF
export USE_KINETO=OFF
export USE_LITE_INTERPRETER_PROFILER=OFF
export USE_MKLDNN=OFF
export USE_NNPACK=OFF
export USE_OPENMP=OFF
export USE_PYTORCH_QNNPACK=OFF
export USE_QNNPACK=OFF
export USE_ROCM=OFF
export USE_SYSTEM_LIBS=ON
export USE_TENSORPIPE=OFF
export USE_XNNPACK=OFF

%py3_build

%install
%py3_install

%check
%tox

%files -n python3-%{pypi_name}
%license LICENSE aten/src/ATen/native/quantized/cpu/qnnpack/LICENSE aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/LICENSE caffe2/mobile/contrib/libopencl-stub/LICENSE test/distributed/pipeline/sync/LICENSE test/test_license.py third_party/LICENSES_BUNDLED.txt third_party/miniz-2.1.0/LICENSE torch/distributed/pipeline/sync/LICENSE torch/fx/experimental/unification/LICENSE.txt
%doc .ci/caffe2/README.md .ci/docker/README.md .ci/onnx/README.md .ci/pytorch/README.md .circleci/README.md .circleci/scripts/README.md .github/requirements/README.md .github/scripts/README.md README.md android/README.md aten/src/ATen/core/README.md aten/src/ATen/core/dispatch/README.md aten/src/ATen/core/op_registration/README.md aten/src/ATen/cudnn/README.md aten/src/ATen/mkl/README.md aten/src/ATen/native/README.md aten/src/ATen/native/cpu/README.md aten/src/ATen/native/nested/README.md aten/src/ATen/native/quantized/README.md aten/src/ATen/native/quantized/cpu/kernels/README.md aten/src/ATen/native/quantized/cpu/qnnpack/README.md aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/README.md aten/src/README.md benchmarks/README.md benchmarks/distributed/ddp/README.md benchmarks/distributed/rpc/parameter_server/README.md benchmarks/distributed/rpc/rl/README.md benchmarks/dynamo/README.md benchmarks/dynamo/_onnx/README.md benchmarks/fastrnns/README.md benchmarks/functional_autograd_benchmark/README.md benchmarks/instruction_counts/README.md benchmarks/operator_benchmark/README.md benchmarks/overrides_benchmark/README.md benchmarks/sparse/README.md benchmarks/sparse/dlmc/README.md c10/core/impl/README.md c10/core/impl/cow/README.md c10/cuda/README.md caffe2/README.md caffe2/contrib/aten/README.md caffe2/contrib/fakelowp/test/README.md caffe2/contrib/playground/README.md caffe2/contrib/tensorrt/README.md caffe2/core/nomnigraph/README.md caffe2/mobile/contrib/libopencl-stub/README.md caffe2/mobile/contrib/snpe/README.md caffe2/observers/README.md caffe2/python/onnx/README.md caffe2/python/serialized_test/README.md caffe2/quantization/server/README.md cmake/Modules/README.md cmake/Modules_CUDA_fix/README.md cmake/Modules_CUDA_fix/upstream/README.md docs/README.md docs/caffe2/README.md functorch/COMPILE_README.md functorch/README.md functorch/dim/README.md functorch/docs/README.md functorch/examples/compilation/README.md functorch/examples/dp_cifar10/README.md functorch/examples/maml_omniglot/README.md ios/README.md ios/TestApp/README.md scripts/README.md scripts/release/README.md scripts/release_notes/README.md test/cpp/api/README.md test/cpp/jit/README.md test/cpp/tensorexpr/README.md test/distributed/_tensor/README.md test/mobile/model_test/README.md third_party/README.md third_party/miniz-2.1.0/readme.md third_party/nvfuser/csrc/python_frontend/README.md third_party/nvfuser/examples/sinh_extension/README.md third_party/nvfuser/examples/sinh_libtorch/README.md third_party/valgrind-headers/README.md tools/README.md tools/autograd/README.md tools/code_coverage/README.md tools/coverage_plugins_package/README.md tools/linter/adapters/README.md tools/stats/README.md torch/README.txt torch/_numpy/README.md torch/ao/pruning/_experimental/activation_sparsifier/README.md torch/ao/pruning/_experimental/data_scheduler/README.md torch/ao/pruning/_experimental/data_sparsifier/README.md torch/ao/pruning/_experimental/data_sparsifier/benchmarks/README.md torch/ao/pruning/_experimental/data_sparsifier/lightning/callbacks/README.md torch/ao/pruning/_experimental/pruner/README.md torch/ao/quantization/backend_config/README.md torch/ao/quantization/fx/README.md torch/ao/quantization/fx/_model_report/README.md torch/csrc/README.md torch/csrc/autograd/README.md torch/csrc/deploy/README.md torch/csrc/jit/README.md torch/csrc/jit/codegen/cuda/README.md torch/csrc/jit/codegen/fuser/README.md torch/csrc/jit/codegen/onednn/README.md torch/csrc/jit/operator_upgraders/README.md torch/csrc/jit/passes/onnx/README.md torch/csrc/jit/runtime/static/README.md torch/csrc/lazy/generated/README.md torch/csrc/lazy/python/README.md torch/distributed/_tensor/README.md torch/distributed/benchmarks/README.md torch/fx/README.md torch/fx/passes/README.md torch/legacy/README.md torch/onnx/README.md torch/utils/benchmark/README.md torch/utils/data/datapipes/README.md torchgen/packaged/autograd/README.md
%{_bindir}/convert-caffe2-to-onnx
%{_bindir}/convert-onnx-to-caffe2
%{_bindir}/torchrun
%{python3_sitearch}/


%files -n python-%{pypi_name}-doc
%license LICENSE aten/src/ATen/native/quantized/cpu/qnnpack/LICENSE aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/LICENSE caffe2/mobile/contrib/libopencl-stub/LICENSE test/distributed/pipeline/sync/LICENSE test/test_license.py third_party/LICENSES_BUNDLED.txt third_party/miniz-2.1.0/LICENSE torch/distributed/pipeline/sync/LICENSE torch/fx/experimental/unification/LICENSE.txt

%changelog
* Sat Sep 30 2023 Tom Rix <trix@redhat.com> - 2.1.0-1
- Initial package.

