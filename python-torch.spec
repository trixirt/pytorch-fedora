%global pypi_name torch
%global pypi_version 2.1.0

# Where the src comes from
%global forgeurl https://github.com/pytorch/pytorch

# So pre releases can be tried
%bcond_with gitcommit
%if %{with gitcommit}
# The top of the 2.1.0 branch - update to whatever..
%global commit0 1841d54370d167365d15f0ac78efc2c56cdf43ab
%global shortcommit0 %(c=%{commit0}; echo ${c:0:7})
%endif

# For -test subpackage
# suitable only for local testing
# Install and do something like
#   export LD_LIBRARY_PATH=/usr/lib64/python3.12/site-packages/torch/lib
#   /usr/lib64/python3.12/site-packages/torch/bin/test_api, test_lazy
%bcond_with test

Name:           python-%{pypi_name}
Version:        2.1.0
Release:        10%{?dist}
Summary:        PyTorch AI/ML framework
# See below for details
License:        BSD-3-Clause AND BSD-2-Clause AND 0BSD AND Apache-2.0 AND MIT AND BSL-1.0 AND GPL-3.0-or-later AND Zlib

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
# Fedora requires versioned so's
Patch5:         0001-pytorch-use-SO-version-by-default.patch
# libtorch_python.so: undefined symbols: Py*
Patch6:         0001-python-torch-link-with-python.patch
# E: unused-direct-shlib-dependency libshm.so.2.1.0 libtorch.so.2.1
# turn on as-needed globally
Patch7:         0001-python-torch-remove-ubuntu-specific-linking.patch
# Tries to use git and is confused by tarball
Patch8:         0001-torch-sane-version.patch
# libtorch is a wrapper so turn off as-needed locally
# resolves this rpmlint
# E: shared-library-without-dependency-information libtorch.so.2.1.0
# causes these
# E: unused-direct-shlib-dependency libtorch.so.2.1.0 libtorch_cpu.so.2.1
# etc.
# As a wrapper library, this should be the expected behavior.
Patch9:         0001-disable-as-needed-for-libtorch.patch

# Limit to these because they are well behaved with clang
ExclusiveArch:  x86_64 aarch64
%global toolchain clang

BuildRequires:  clang-devel
BuildRequires:  cmake
BuildRequires:  cpuinfo-devel
BuildRequires:  eigen3-devel
BuildRequires:  fmt-devel
BuildRequires:  flatbuffers-devel
BuildRequires:  FP16-devel
BuildRequires:  fxdiv-devel
BuildRequires:  gcc-c++
BuildRequires:  gcc-gfortran
BuildRequires:  gloo-devel
BuildRequires:  ninja-build
BuildRequires:  onnx-devel
BuildRequires:  openblas-devel
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

%description -n python3-%{pypi_name}
PyTorch is a Python package that provides two high-level features:

 * Tensor computation (like NumPy) with strong GPU acceleration
 * Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy,
and Cython to extend PyTorch when needed.

%package -n python3-%{pypi_name}-devel
Summary:        Libraries and headers for %{name}
Requires:       python3-%{pypi_name}%{?_isa} = %{version}-%{release}

%description -n python3-%{pypi_name}-devel
%{summary}

%if %{with test}
%package -n python3-%{pypi_name}-test
Summary:        Tests for %{name}
Requires:       python3-%{pypi_name}%{?_isa} = %{version}-%{release}

%description -n python3-%{pypi_name}-test
%{summary}
%endif


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

%if %{with test}
mv third_party/googletest .
%endif

# Remove everything
rm -rf third_party/*
# Put stuff back
mv build_bundled.py third_party
mv miniz-2.1.0 third_party
%if %{with test}
mv googletest third_party
%endif
#
# Fake out pocketfft, and system header will be used
mkdir third_party/pocketfft
#
# Use the system valgrind headers
mkdir third_party/valgrind-headers
cp %{_includedir}/valgrind/* third_party/valgrind-headers

# Remove unneeded OpenCL files that confuse the lincense scanner
rm caffe2/contrib/opencl/OpenCL/cl.hpp
rm caffe2/mobile/contrib/libopencl-stub/include/CL/cl.h
rm caffe2/mobile/contrib/libopencl-stub/include/CL/cl.hpp
rm caffe2/mobile/contrib/libopencl-stub/include/CL/cl_ext.h
rm caffe2/mobile/contrib/libopencl-stub/include/CL/cl_gl.h
rm caffe2/mobile/contrib/libopencl-stub/include/CL/cl_gl_ext.h
rm caffe2/mobile/contrib/libopencl-stub/include/CL/cl_platform.h
rm caffe2/mobile/contrib/libopencl-stub/include/CL/opencl.h

%endif

%build

# For debugging setup.py
# export SETUPTOOLS_SCM_DEBUG=1

# For verbose cmake output
# export VERBOSE=ON
# For verbose linking
# export CMAKE_SHARED_LINKER_FLAGS=-Wl,--verbose

# Manually set this hardening flag
export CMAKE_EXE_LINKER_FLAGS=-pie

export BUILD_CUSTOM_PROTOBUF=OFF
export BUILD_SHARED_LIBS=ON
%if %{with test}
export BUILD_TEST=ON
%else
export BUILD_TEST=OFF
%endif
export CMAKE_BUILD_TYPE=RelWithDebInfo
export CMAKE_FIND_PACKAGE_PREFER_CONFIG=ON
export CAFFE2_LINK_LOCAL_PROTOBUF=OFF
export USE_CUDA=OFF
export USE_DISTRIBUTED=OFF
export USE_FBGEMM=OFF
export USE_GOLD_LINKER=OFF
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

# Why we are using py3_ vs pyproject_
#
# current pyproject problem with mock
# + /usr/bin/python3 -Bs /usr/lib/rpm/redhat/pyproject_wheel.py /builddir/build/BUILD/pytorch-v2.1.0/pyproject-wheeldir
# /usr/bin/python3: No module named pip
# Adding pip to build requires does not fix
#
# See BZ 2244862
%py3_build

%install
%py3_install

# empty files
rm %{buildroot}%{python3_sitearch}/torch/py.typed
rm %{buildroot}%{python3_sitearch}/torch/ao/quantization/backend_config/observation_type.py
rm %{buildroot}%{python3_sitearch}/torch/ao/quantization/backend_config/__pycache__/observation_type.*.pyc
rm %{buildroot}%{python3_sitearch}/torch/cuda/error.py
rm %{buildroot}%{python3_sitearch}/torch/cuda/__pycache__/error.*.pyc
rm %{buildroot}%{python3_sitearch}/torch/include/ATen/cudnn/Exceptions.h

# exec permission
for f in `find %{buildroot}%{python3_sitearch} -name '*.py'`; do
    if [ ! -x $f ]; then
        sed -i '1{\@^#!/usr/bin@d}' $f
    fi
done

# shebangs
%py3_shebang_fix %{buildroot}%{python3_sitearch}

# Remove copies of the *.so's
#
# *.so should be symlinks, not copies
# there is a copy of *.so.x.y.z to *.so and *.so.x.y
# remove the copies and do the link
#
%global lib_list c10 shm torch torch_cpu torch_global_deps torch_python
{
    cd %{buildroot}%{python3_sitearch}/torch/lib
    for l in %{lib_list}
    do
        long_name=lib${l}.so.%{version}
        short_name=${long_name%.*}
        devel_name=lib${l}.so
        rm ${short_name}
        rm ${devel_name}
        ln -s ${long_name} ${short_name}
        ln -s ${long_name} ${devel_name}
    done
    cd -
}

# Programatically create the list of dirs
echo "s|%{buildroot}%{python3_sitearch}|%%dir %%{python3_sitearch}|g" > br.sed
find %{buildroot}%{python3_sitearch} -mindepth 1 -type d  > dirs.files
sed -i -f br.sed dirs.files 
cat dirs.files > main.files

# Similar for the python files
find %{buildroot}%{python3_sitearch} -type f -name "*.py" -o -name "*.pyc" -o -name "*.pyi"  > py.files
echo "s|%{buildroot}%{python3_sitearch}|%%{python3_sitearch}|g" > br.sed
sed -i -f br.sed py.files
cat py.files >> main.files

# devel files, headers and such
find %{buildroot}%{python3_sitearch} -type f -name "*.h" -o -name "*.hpp" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.cu" > devel.files
sed -i -f br.sed devel.files

#
# Main package
##% dir % {python3_sitearch}/torch*.egg-info

%files -n python3-%{pypi_name} -f main.files

%license LICENSE
%doc README.md

# bins
%{_bindir}/convert-caffe2-to-onnx
%{_bindir}/convert-onnx-to-caffe2
%{_bindir}/torchrun
%{python3_sitearch}/torch/bin/torch_shm_manager

# libs
%{python3_sitearch}/functorch/_C.cpython*.so
%{python3_sitearch}/torch/_C.cpython*.so
%{python3_sitearch}/torch/lib/libc10.so.*
%{python3_sitearch}/torch/lib/libshm.so.*
%{python3_sitearch}/torch/lib/libtorch.so.*
%{python3_sitearch}/torch/lib/libtorch_cpu.so.*
%{python3_sitearch}/torch/lib/libtorch_global_deps.so.*
%{python3_sitearch}/torch/lib/libtorch_python.so.*

# misc
%{python3_sitearch}/torch/utils/model_dump/{*.js,*.mjs,*.html}
%{python3_sitearch}/torchgen/packaged/ATen/native/*.yaml
%{python3_sitearch}/torchgen/packaged/autograd/{*.md,*.yaml}

# egg
%{python3_sitearch}/torch*.egg-info/*

# excludes
# bazel build cruft
%exclude %{python3_sitearch}/torchgen/packaged/autograd/{BUILD.bazel,build.bzl}

#
# devel package
#
%files -n python3-%{pypi_name}-devel -f devel.files

# devel libs
%{python3_sitearch}/torch/lib/libc10.so
%{python3_sitearch}/torch/lib/libshm.so
%{python3_sitearch}/torch/lib/libtorch.so
%{python3_sitearch}/torch/lib/libtorch_cpu.so
%{python3_sitearch}/torch/lib/libtorch_global_deps.so
%{python3_sitearch}/torch/lib/libtorch_python.so

# devel cmake
%{python3_sitearch}/torch/share/cmake/{ATen,Caffe2,Torch}/*.cmake
%{python3_sitearch}/torch/share/cmake/Caffe2/public/*.cmake
%{python3_sitearch}/torch/share/cmake/Caffe2/Modules_CUDA_fix/*.cmake
%{python3_sitearch}/torch/share/cmake/Caffe2/Modules_CUDA_fix/upstream/*.cmake
%{python3_sitearch}/torch/share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/*.cmake

# devel misc
%{python3_sitearch}/torchgen/packaged/ATen/templates/RegisterDispatchDefinitions.ini
%{python3_sitearch}/torchgen/packaged/autograd/templates/annotated_fn_args.py.in

%if %{with test}
%files -n python3-%{pypi_name}-test

# test bins
%{python3_sitearch}/torch/bin/test_api
%{python3_sitearch}/torch/bin/test_edge_op_registration
%{python3_sitearch}/torch/bin/test_jit
%{python3_sitearch}/torch/bin/test_lazy
%{python3_sitearch}/torch/bin/test_tensorexpr
%{python3_sitearch}/torch/bin/tutorial_tensorexpr

# test libs
# Unversioned - not ment for release
%{python3_sitearch}/torch/lib/libbackend_with_compiler.so
%{python3_sitearch}/torch/lib/libjitbackend_test.so
%{python3_sitearch}/torch/lib/libtorchbind_test.so
   
# tests
%{python3_sitearch}/torch/test/*

%endif

#
# License Details
# Main license BSD 3-Clause
#
# Apache-2.0
# android/libs/fbjni/LICENSE
# android/libs/fbjni/CMakeLists.txt
# android/libs/fbjni/build.gradle
# android/libs/fbjni/cxx/fbjni/ByteBuffer.cpp
# android/libs/fbjni/cxx/fbjni/ByteBuffer.h
# android/libs/fbjni/cxx/fbjni/Context.h
# android/libs/fbjni/cxx/fbjni/File.h
# android/libs/fbjni/cxx/fbjni/JThread.h
# android/libs/fbjni/cxx/fbjni/NativeRunnable.h
# android/libs/fbjni/cxx/fbjni/OnLoad.cpp
# android/libs/fbjni/cxx/fbjni/ReadableByteChannel.cpp
# android/libs/fbjni/cxx/fbjni/ReadableByteChannel.h
# android/libs/fbjni/cxx/fbjni/detail/Boxed.h
# android/libs/fbjni/cxx/fbjni/detail/Common.h
# android/libs/fbjni/cxx/fbjni/detail/CoreClasses-inl.h
# android/libs/fbjni/cxx/fbjni/detail/CoreClasses.h
# android/libs/fbjni/cxx/fbjni/detail/Environment.cpp
# android/libs/fbjni/cxx/fbjni/detail/Environment.h
# android/libs/fbjni/cxx/fbjni/detail/Exceptions.cpp
# android/libs/fbjni/cxx/fbjni/detail/Exceptions.h
# android/libs/fbjni/cxx/fbjni/detail/FbjniApi.h
# android/libs/fbjni/cxx/fbjni/detail/Hybrid.cpp
# android/libs/fbjni/cxx/fbjni/detail/Hybrid.h
# android/libs/fbjni/cxx/fbjni/detail/Iterator-inl.h
# android/libs/fbjni/cxx/fbjni/detail/Iterator.h
# android/libs/fbjni/cxx/fbjni/detail/JWeakReference.h
# android/libs/fbjni/cxx/fbjni/detail/Log.h
# android/libs/fbjni/cxx/fbjni/detail/Meta-forward.h
# android/libs/fbjni/cxx/fbjni/detail/Meta-inl.h
# android/libs/fbjni/cxx/fbjni/detail/Meta.cpp
# android/libs/fbjni/cxx/fbjni/detail/Meta.h
# android/libs/fbjni/cxx/fbjni/detail/MetaConvert.h
# android/libs/fbjni/cxx/fbjni/detail/ReferenceAllocators-inl.h
# android/libs/fbjni/cxx/fbjni/detail/ReferenceAllocators.h
# android/libs/fbjni/cxx/fbjni/detail/References-forward.h
# android/libs/fbjni/cxx/fbjni/detail/References-inl.h
# android/libs/fbjni/cxx/fbjni/detail/References.cpp
# android/libs/fbjni/cxx/fbjni/detail/References.h
# android/libs/fbjni/cxx/fbjni/detail/Registration-inl.h
# android/libs/fbjni/cxx/fbjni/detail/Registration.h
# android/libs/fbjni/cxx/fbjni/detail/SimpleFixedString.h
# android/libs/fbjni/cxx/fbjni/detail/TypeTraits.h
# android/libs/fbjni/cxx/fbjni/detail/utf8.cpp
# android/libs/fbjni/cxx/fbjni/detail/utf8.h
# android/libs/fbjni/cxx/fbjni/fbjni.cpp
# android/libs/fbjni/cxx/fbjni/fbjni.h
# android/libs/fbjni/cxx/lyra/cxa_throw.cpp
# android/libs/fbjni/cxx/lyra/lyra.cpp
# android/libs/fbjni/cxx/lyra/lyra.h
# android/libs/fbjni/cxx/lyra/lyra_breakpad.cpp
# android/libs/fbjni/cxx/lyra/lyra_exceptions.cpp
# android/libs/fbjni/cxx/lyra/lyra_exceptions.h
# android/libs/fbjni/gradle.properties
# android/libs/fbjni/gradle/android-tasks.gradle
# android/libs/fbjni/gradle/release.gradle
# android/libs/fbjni/gradlew
# android/libs/fbjni/gradlew.bat
# android/libs/fbjni/host.gradle
# android/libs/fbjni/java/com/facebook/jni/CppException.java
# android/libs/fbjni/java/com/facebook/jni/CppSystemErrorException.java
# android/libs/fbjni/java/com/facebook/jni/DestructorThread.java
# android/libs/fbjni/java/com/facebook/jni/HybridClassBase.java
# android/libs/fbjni/java/com/facebook/jni/HybridData.java
# android/libs/fbjni/java/com/facebook/jni/IteratorHelper.java
# android/libs/fbjni/java/com/facebook/jni/MapIteratorHelper.java
# android/libs/fbjni/java/com/facebook/jni/NativeRunnable.java
# android/libs/fbjni/java/com/facebook/jni/ThreadScopeSupport.java
# android/libs/fbjni/java/com/facebook/jni/UnknownCppException.java
# android/libs/fbjni/java/com/facebook/jni/annotations/DoNotStrip.java
# android/libs/fbjni/scripts/android-setup.sh
# android/libs/fbjni/scripts/run-host-tests.sh
# android/libs/fbjni/settings.gradle
# android/libs/fbjni/test/BaseFBJniTests.java
# android/libs/fbjni/test/ByteBufferTests.java
# android/libs/fbjni/test/DocTests.java
# android/libs/fbjni/test/FBJniTests.java
# android/libs/fbjni/test/HybridTests.java
# android/libs/fbjni/test/IteratorTests.java
# android/libs/fbjni/test/PrimitiveArrayTests.java
# android/libs/fbjni/test/ReadableByteChannelTests.java
# android/libs/fbjni/test/jni/CMakeLists.txt
# android/libs/fbjni/test/jni/byte_buffer_tests.cpp
# android/libs/fbjni/test/jni/doc_tests.cpp
# android/libs/fbjni/test/jni/expect.h
# android/libs/fbjni/test/jni/fbjni_onload.cpp
# android/libs/fbjni/test/jni/fbjni_tests.cpp
# android/libs/fbjni/test/jni/hybrid_tests.cpp
# android/libs/fbjni/test/jni/inter_dso_exception_test_1/Test.cpp
# android/libs/fbjni/test/jni/inter_dso_exception_test_1/Test.h
# android/libs/fbjni/test/jni/inter_dso_exception_test_2/Test.cpp
# android/libs/fbjni/test/jni/inter_dso_exception_test_2/Test.h
# android/libs/fbjni/test/jni/iterator_tests.cpp
# android/libs/fbjni/test/jni/modified_utf8_test.cpp
# android/libs/fbjni/test/jni/no_rtti.cpp
# android/libs/fbjni/test/jni/no_rtti.h
# android/libs/fbjni/test/jni/primitive_array_tests.cpp
# android/libs/fbjni/test/jni/readable_byte_channel_tests.cpp
# android/libs/fbjni/test/jni/simple_fixed_string_tests.cpp
# android/libs/fbjni/test/jni/utf16toUTF8_test.cpp
# android/pytorch_android/host/build.gradle
# aten/src/ATen/cuda/llvm_basic.cpp
# aten/src/ATen/cuda/llvm_complex.cpp
# aten/src/ATen/native/quantized/cpu/qnnpack/confu.yaml
# aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-neon.c
# aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-scalar.h
# aten/src/ATen/native/quantized/cpu/qnnpack/src/requantization/gemmlowp-sse.h
# aten/src/ATen/nnapi/codegen.py
# aten/src/ATen/nnapi/NeuralNetworks.h
# aten/src/ATen/nnapi/nnapi_wrapper.cpp
# aten/src/ATen/nnapi/nnapi_wrapper.h
# binaries/benchmark_args.h
# binaries/benchmark_helper.cc
# binaries/benchmark_helper.h
# binaries/compare_models_torch.cc
# binaries/convert_and_benchmark.cc
# binaries/convert_caffe_image_db.cc
# binaries/convert_db.cc
# binaries/convert_encoded_to_raw_leveldb.cc
# binaries/convert_image_to_tensor.cc
# binaries/core_overhead_benchmark.cc
# binaries/core_overhead_benchmark_gpu.cc
# binaries/db_throughput.cc
# binaries/dump_operator_names.cc
# binaries/inspect_gpu.cc
# binaries/load_benchmark_torch.cc
# binaries/make_cifar_db.cc
# binaries/make_image_db.cc
# binaries/make_mnist_db.cc
# binaries/optimize_for_mobile.cc
# binaries/parallel_info.cc
# binaries/predictor_verifier.cc
# binaries/print_core_object_sizes_gpu.cc
# binaries/print_registered_core_operators.cc
# binaries/run_plan.cc
# binaries/run_plan_mpi.cc
# binaries/speed_benchmark.cc
# binaries/speed_benchmark_torch.cc
# binaries/split_db.cc
# binaries/tsv_2_proto.cc
# binaries/tutorial_blob.cc
# binaries/zmq_feeder.cc
# c10/test/util/small_vector_test.cpp
# c10/util/FunctionRef.h
# c10/util/SmallVector.cpp
# c10/util/SmallVector.h
# c10/util/llvmMathExtras.h
# c10/util/sparse_bitset.h
# caffe2/contrib/aten/gen_op.py
# caffe2/contrib/fakelowp/fp16_fc_acc_op.cc
# caffe2/contrib/fakelowp/fp16_fc_acc_op.h
# caffe2/contrib/gloo/allgather_ops.cc
# caffe2/contrib/gloo/allgather_ops.h
# caffe2/contrib/gloo/reduce_scatter_ops.cc
# caffe2/contrib/gloo/reduce_scatter_ops.h
# caffe2/core/hip/common_miopen.h
# caffe2/core/hip/common_miopen.hip
# caffe2/core/net_async_tracing.cc
# caffe2/core/net_async_tracing.h
# caffe2/core/net_async_tracing_test.cc
# caffe2/experiments/operators/fully_connected_op_decomposition.cc
# caffe2/experiments/operators/fully_connected_op_decomposition.h
# caffe2/experiments/operators/fully_connected_op_decomposition_gpu.cc
# caffe2/experiments/operators/fully_connected_op_prune.cc
# caffe2/experiments/operators/fully_connected_op_prune.h
# caffe2/experiments/operators/fully_connected_op_sparse.cc
# caffe2/experiments/operators/fully_connected_op_sparse.h
# caffe2/experiments/operators/funhash_op.cc
# caffe2/experiments/operators/funhash_op.h
# caffe2/experiments/operators/sparse_funhash_op.cc
# caffe2/experiments/operators/sparse_funhash_op.h
# caffe2/experiments/operators/sparse_matrix_reshape_op.cc
# caffe2/experiments/operators/sparse_matrix_reshape_op.h
# caffe2/experiments/operators/tt_contraction_op.cc
# caffe2/experiments/operators/tt_contraction_op.h
# caffe2/experiments/operators/tt_contraction_op_gpu.cc
# caffe2/experiments/operators/tt_pad_op.cc
# caffe2/experiments/operators/tt_pad_op.h
# caffe2/experiments/python/SparseTransformer.py
# caffe2/experiments/python/convnet_benchmarks.py
# caffe2/experiments/python/device_reduce_sum_bench.py
# caffe2/experiments/python/funhash_op_test.py
# caffe2/experiments/python/net_construct_bench.py
# caffe2/experiments/python/sparse_funhash_op_test.py
# caffe2/experiments/python/sparse_reshape_op_test.py
# caffe2/experiments/python/tt_contraction_op_test.py
# caffe2/experiments/python/tt_pad_op_test.py
# caffe2/mobile/contrib/libvulkan-stub/include/vulkan/vk_platform.h
# caffe2/mobile/contrib/libvulkan-stub/include/vulkan/vulkan.h
# caffe2/mobile/contrib/nnapi/NeuralNetworks.h
# caffe2/mobile/contrib/nnapi/dlnnapi.c
# caffe2/mobile/contrib/nnapi/nnapi_benchmark.cc
# caffe2/observers/profile_observer.cc
# caffe2/observers/profile_observer.h
# caffe2/operators/hip/conv_op_miopen.hip
# caffe2/operators/hip/local_response_normalization_op_miopen.hip
# caffe2/operators/hip/pool_op_miopen.hip
# caffe2/operators/hip/spatial_batch_norm_op_miopen.hip
# caffe2/operators/quantized/int8_utils.h
# caffe2/operators/stump_func_op.cc
# caffe2/operators/stump_func_op.cu
# caffe2/operators/stump_func_op.h
# caffe2/operators/unique_ops.cc
# caffe2/operators/unique_ops.cu
# caffe2/operators/unique_ops.h
# caffe2/operators/upsample_op.cc
# caffe2/operators/upsample_op.h
# caffe2/opt/fusion.h
# caffe2/python/layers/label_smooth.py
# caffe2/python/mint/static/css/simple-sidebar.css
# caffe2/python/modeling/get_entry_from_blobs.py
# caffe2/python/modeling/get_entry_from_blobs_test.py
# caffe2/python/modeling/gradient_clipping_test.py
# caffe2/python/operator_test/unique_ops_test.py
# caffe2/python/operator_test/upsample_op_test.py
# caffe2/python/operator_test/weight_scale_test.py
# caffe2/python/pybind_state_int8.cc
# caffe2/python/transformations.py
# caffe2/python/transformations_test.py
# caffe2/quantization/server/batch_matmul_dnnlowp_op.cc
# caffe2/quantization/server/batch_matmul_dnnlowp_op.h
# caffe2/quantization/server/compute_equalization_scale_test.py
# caffe2/quantization/server/elementwise_linear_dnnlowp_op.cc
# caffe2/quantization/server/elementwise_linear_dnnlowp_op.h
# caffe2/quantization/server/elementwise_sum_relu_op.cc
# caffe2/quantization/server/fb_fc_packed_op.cc
# caffe2/quantization/server/fb_fc_packed_op.h
# caffe2/quantization/server/fbgemm_fp16_pack_op.cc
# caffe2/quantization/server/fbgemm_fp16_pack_op.h
# caffe2/quantization/server/fully_connected_fake_lowp_op.cc
# caffe2/quantization/server/fully_connected_fake_lowp_op.h
# caffe2/quantization/server/int8_gen_quant_params_min_max_test.py
# caffe2/quantization/server/int8_gen_quant_params_test.py
# caffe2/quantization/server/int8_quant_scheme_blob_fill_test.py
# caffe2/quantization/server/spatial_batch_norm_relu_op.cc
# caffe2/sgd/weight_scale_op.cc
# caffe2/sgd/weight_scale_op.h
# caffe2/utils/bench_utils.h
# functorch/examples/maml_omniglot/maml-omniglot-higher.py
# functorch/examples/maml_omniglot/maml-omniglot-ptonly.py
# functorch/examples/maml_omniglot/maml-omniglot-transforms.py
# functorch/examples/maml_omniglot/support/omniglot_loaders.py
# modules/detectron/group_spatial_softmax_op.cc
# modules/detectron/group_spatial_softmax_op.cu
# modules/detectron/group_spatial_softmax_op.h
# modules/detectron/ps_roi_pool_op.cc
# modules/detectron/ps_roi_pool_op.h
# modules/detectron/roi_pool_f_op.cc
# modules/detectron/roi_pool_f_op.cu
# modules/detectron/roi_pool_f_op.h
# modules/detectron/sample_as_op.cc
# modules/detectron/sample_as_op.cu
# modules/detectron/sample_as_op.h
# modules/detectron/select_smooth_l1_loss_op.cc
# modules/detectron/select_smooth_l1_loss_op.cu
# modules/detectron/select_smooth_l1_loss_op.h
# modules/detectron/sigmoid_cross_entropy_loss_op.cc
# modules/detectron/sigmoid_cross_entropy_loss_op.cu
# modules/detectron/sigmoid_cross_entropy_loss_op.h
# modules/detectron/sigmoid_focal_loss_op.cc
# modules/detectron/sigmoid_focal_loss_op.cu
# modules/detectron/sigmoid_focal_loss_op.h
# modules/detectron/smooth_l1_loss_op.cc
# modules/detectron/smooth_l1_loss_op.cu
# modules/detectron/smooth_l1_loss_op.h
# modules/detectron/softmax_focal_loss_op.cc
# modules/detectron/softmax_focal_loss_op.cu
# modules/detectron/softmax_focal_loss_op.h
# modules/detectron/spatial_narrow_as_op.cc
# modules/detectron/spatial_narrow_as_op.cu
# modules/detectron/spatial_narrow_as_op.h
# modules/detectron/upsample_nearest_op.cc
# modules/detectron/upsample_nearest_op.h
# modules/module_test/module_test_dynamic.cc
# modules/rocksdb/rocksdb.cc
# scripts/apache_header.txt
# scripts/apache_python.txt
# torch/distributions/lkj_cholesky.py
#
# Apache 2.0 AND BSD 2-Clause
# caffe2/operators/deform_conv_op.cu
#
# Apache 2.0 AND BSD 2-Clause AND MIT
# modules/detectron/ps_roi_pool_op.cu
#
# Apache 2.0 AND BSD 2-Clause
# modules/detectron/upsample_nearest_op.cu
#
# BSD 0-Clause
# torch/csrc/utils/pythoncapi_compat.h
#
# BSD 2-Clause
# aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/LICENSE
# caffe2/image/transform_gpu.cu
# caffe2/image/transform_gpu.h
#
# BSL-1.0
# c10/util/flat_hash_map.h
# c10/util/hash.h
# c10/util/Optional.h
# c10/util/order_preserving_flat_hash_map.h
# c10/util/strong_type.h
# c10/util/variant.h
#
# GPL-3.0-or-later AND MIT
# c10/util/reverse_iterator.h
#
# Khronos
# These files are for OpenCL, an unused option
# Replace them later, as-needed with the opencl-headers.rpm
#
# caffe2/contrib/opencl/OpenCL/cl.hpp
# caffe2/mobile/contrib/libopencl-stub/include/CL/cl.h
# caffe2/mobile/contrib/libopencl-stub/include/CL/cl.hpp
# caffe2/mobile/contrib/libopencl-stub/include/CL/cl_ext.h
# caffe2/mobile/contrib/libopencl-stub/include/CL/cl_gl.h
# caffe2/mobile/contrib/libopencl-stub/include/CL/cl_gl_ext.h
# caffe2/mobile/contrib/libopencl-stub/include/CL/cl_platform.h
# caffe2/mobile/contrib/libopencl-stub/include/CL/opencl.h
#
# MIT
# android/libs/fbjni/googletest-CMakeLists.txt.in
# c10/util/BFloat16-math.h
# caffe2/mobile/contrib/libvulkan-stub/include/libvulkan-stub.h
# caffe2/mobile/contrib/libvulkan-stub/src/libvulkan-stub.c
# caffe2/onnx/torch_ops/defs.cc
# cmake/Modules_CUDA_fix/upstream/FindCUDA/make2cmake.cmake
# cmake/Modules_CUDA_fix/upstream/FindCUDA/parse_cubin.cmake
# cmake/Modules_CUDA_fix/upstream/FindCUDA/run_nvcc.cmake
# functorch/einops/_parsing.py
# test/functorch/test_parsing.py
# test/functorch/test_rearrange.py
# third_party/miniz-2.1.0/LICENSE
# third_party/miniz-2.1.0/miniz.c
# tools/coverage_plugins_package/setup.py
# torch/_appdirs.py
# torch/utils/hipify/hipify_python.py
#
# Public Domain
# caffe2/mobile/contrib/libopencl-stub/LICENSE
# caffe2/utils/murmur_hash3.cc
# caffe2/utils/murmur_hash3.h
#
# Zlib
# aten/src/ATen/native/cpu/avx_mathfun.h

%changelog
* Fri Dec 1 2023 Tom Rix <trix@redhat.com> - 2.1.0-10
- Disable gold linker
- Remove python requires
- Change to openblas, remove -lgfortran fixes
- Manually add -pie to linking options

* Fri Nov 24 2023 Tom Rix <trix@redhat.com> - 2.1.0-9
- Enable debug build
- Remove Khronos licensed files from source
- Use 0BSD license identifier
- Generate lists directories, python and header files
- Add a -test subpackage

* Wed Nov 15 2023 Tom Rix <trix@redhat.com> - 2.1.0-8
- Address review comments

* Thu Nov 2 2023 Tom Rix <trix@redhat.com> - 2.1.0-7
- Address review comments
- remove pyproject option

* Thu Oct 19 2023 Tom Rix <trix@redhat.com> - 2.1.0-6
- Address review comments

* Wed Oct 18 2023 Tom Rix <trix@redhat.com> - 2.1.0-5
- Address review comments

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

