#!/bin/bash
set +e
set +u
set +o pipefail 2>/dev/null || true

M489_RUN="$1"
M489_ROOT="/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5"
M489_BASE="/home/bingxing2/apps/anaconda/2021.11/envs/py310torch251cu121"
M489_POLICY="$M489_ROOT/02_env/gr00t_n15_py310_t251_cu121"

printf 'job_id=%s\narray_job_id=%s\ntask_index=%s\nnode=%s\n' \
    "${SLURM_JOB_ID-}" "${SLURM_ARRAY_JOB_ID-}" \
    "${SLURM_ARRAY_TASK_ID-}" "$(hostname)"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES-}"

if ! type module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    fi
fi

module purge &&
module load compilers/gcc/11.3.0 &&
module load compilers/cuda/12.1 &&
module load cudnn/8.9.5.29_cuda12.x
M489_MODULE_RC=$?
printf 'module_setup_rc=%s\n' "$M489_MODULE_RC"
if [ "$M489_MODULE_RC" -ne 0 ]; then
    exit 1
fi

export PATH="$M489_POLICY/bin:$PATH"
export LD_LIBRARY_PATH="$M489_BASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/home/bingxing2/apps/compilers/cuda/cuda-12.1"

export TMPDIR="$M489_RUN/tmp"
mkdir -p -- "$TMPDIR" || exit 1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$M489_RUN/hf_cache"
export HF_MODULES_CACHE="$M489_RUN/hf_modules"
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="$M489_RUN/numba_cache"
export TRITON_CACHE_DIR="$M489_RUN/triton_cache"

unset TRANSFORMERS_CACHE
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"

cd "$M489_RUN" || exit 1
"$M489_BASE/bin/python" -I -B -u "$M489_RUN/worker.py" orchestrate
M489_RC=$?
printf 's33_task_rc=%s\n' "$M489_RC"
exit "$M489_RC"
