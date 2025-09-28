#!/usr/bin/env bash
# create_runner_and_copy_env.sh
# 用 root 执行：bash create_runner_and_copy_env.sh
set -euo pipefail

USERNAME="runner"
USER_SHELL="/bin/bash"
WORKDIR="/workspace"        # 按需修改工作目录
SET_PASSWORD="no"           # "yes" 则给 runner 设一个同名密码
GRANT_WORKDIR="yes"         # "yes" 则把 WORKDIR 授权给 runner

# ---------- 1) 创建用户/组 ----------
if id -u "${USERNAME}" >/dev/null 2>&1; then
    echo "[INFO] User ${USERNAME} already exists."
else
    # 若希望与宿主机 UID/GID 对齐，可把 1000 改为你的 UID/GID
    groupadd -g 1000 "${USERNAME}" 2>/dev/null || true
    useradd -m -u 1000 -g 1000 -s "${USER_SHELL}" "${USERNAME}"
    echo "[OK] Created user ${USERNAME} (uid=1000,gid=1000)."

    if [[ "${SET_PASSWORD}" == "yes" ]]; then
    echo "${USERNAME}:${USERNAME}" | chpasswd
    echo "[OK] Set password for ${USERNAME}."
    fi

    USER_HOME="$(getent passwd "${USERNAME}" | cut -d: -f6)"
    [[ -d "${USER_HOME}" ]] || { echo "[ERR] Home not found for ${USERNAME}"; exit 1; }

    # ---------- 2) 发现 root 正在用的 python ----------
    # 优先 python3，其次 python；解析真实路径
    ROOT_PY_BIN="$(command -v python3 || command -v python || true)"
    if [[ -z "${ROOT_PY_BIN}" ]]; then
    echo "[ERR] No python found in root PATH"; exit 1
    fi
    ROOT_PY_REAL="$(readlink -f "${ROOT_PY_BIN}")"
    ROOT_PY_DIR="$(dirname "${ROOT_PY_REAL}")"
    echo "[INFO] Root python: ${ROOT_PY_REAL}"

    # ---------- 3) 组装要继承的环境变量 ----------
    # 白名单（可自行添加/删除）
    VARS=("PATH" "LD_LIBRARY_PATH" "PYTHONPATH" "CUDA_HOME" "CUDA_ROOT" "NCCL_DEBUG" "NCCL_IB_DISABLE" "NCCL_SOCKET_IFNAME" "HF_HOME" "TRANSFORMERS_CACHE" "TORCH_HOME")

    # 取当前 shell 的这些变量值（如果设置了）
    ENV_EXPORTS=""
    for v in "${VARS[@]}"; do
    val="${!v-}"
    if [[ -n "${val}" ]]; then
        # 转义双引号
        val_escaped="${val//\"/\\\"}"
        ENV_EXPORTS+="export ${v}=\"${val_escaped}\"\n"
    fi
    done

    # 额外确保 python 路径在最前（以使用 root 的 python）
    # 如果已经在 PATH 中，仍然把 ROOT_PY_DIR 放到最前面，避免被系统别的 python 抢先
    ENV_EXPORTS="export PATH=\"${ROOT_PY_DIR}:\$PATH\"\n${ENV_EXPORTS}"

    # ---------- 4) 写入 runner 的 ~/.bashrc 末尾 ----------
    BASHRC="${USER_HOME}/.bashrc"
    {
    echo ""
    echo "# ===== Imported from root on $(date) ====="
    # shellcheck disable=SC2059
    printf "%b" "${ENV_EXPORTS}"
    echo "# ===== End imported vars ====="
    } >> "${BASHRC}"

    chown "${USERNAME}:${USERNAME}" "${BASHRC}"
    echo "[OK] Appended env to ${BASHRC}"

    # ---------- 5) 可选：给 runner 一个 /usr/local/bin/python 指向 root python 的软链接 ----------
    # 不推荐强制覆盖系统 python；若你明确想让 'python' 命令指向 root 的 python，可取消注释：
    # ln -sf "${ROOT_PY_REAL}" /usr/local/bin/python
    # ln -sf "${ROOT_PY_REAL}" /usr/local/bin/python3
    # echo "[OK] Symlinked /usr/local/bin/python* -> ${ROOT_PY_REAL}"

    # ---------- 6) 可选：授权工作目录 ----------
    if [[ "${GRANT_WORKDIR}" == "yes" ]]; then
    mkdir -p "${WORKDIR}"
    chown -R "${USERNAME}:${USERNAME}" "${WORKDIR}"
    echo "[OK] Chowned ${WORKDIR} to ${USERNAME}:${USERNAME}"
    fi

    # ---------- 7) 可选：如果 root 的 python 来自 /root 下的 venv/conda，给读权限 ----------
    # 仅当你确认需要（否则保持默认权限更安全）
    # ROOT_PY_PREFIX 推断：往上两级可能是 env 前缀；具体视你的环境调整
    # ROOT_PY_PREFIX="$(dirname "$(dirname "${ROOT_PY_REAL}")")"
    # chmod -R a+rX "${ROOT_PY_PREFIX}" || true
    # echo "[INFO] Granted a+rX to ${ROOT_PY_PREFIX} (if applicable)."

    echo "[DONE] User ${USERNAME} ready. Test with:"
    echo "  su - ${USERNAME} -c 'whoami && which python && python -V && env | egrep \"(PYTHONPATH|LD_LIBRARY_PATH|CUDA_HOME)\"'"

fi

# 如果没有这个目录，则拷贝
if [ ! -d "${WORKDIR}/SandboxFusion" ]; then
    cp -r /root/SandboxFusion ${WORKDIR}/
    chown -R "${USERNAME}:${USERNAME}" "${WORKDIR}/SandboxFusion"
    echo "[OK] Chowned ${WORKDIR}/SandboxFusion to ${USERNAME}:${USERNAME}"
fi

# ---------- 运行脚本为 runner（展开 WORKDIR，加载登录环境） ----------
if command -v runuser >/dev/null 2>&1; then
  runuser -u "${USERNAME}" -- bash -lc "${WORKDIR}/SandboxFusion/scripts/run.sh"
else
  su - "${USERNAME}" -c "bash -lc '${WORKDIR}/SandboxFusion/scripts/run.sh'"
fi
