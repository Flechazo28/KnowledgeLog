#!/usr/bin/env bash

set -Eeuo pipefail

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) \
    || die "当前目录不在 Git 仓库中。"
cd "$repo_root"

git remote get-url origin >/dev/null 2>&1 \
    || die "未配置名为 origin 的 GitHub 远端。"

branch=$(git branch --show-current)
[[ -n "$branch" ]] || die "当前处于 detached HEAD，无法确定要推送的分支。"

printf '仓库：%s\n' "$repo_root"
printf '分支：%s\n' "$branch"

# 先检查远端，避免无意中覆盖 GitHub 上更新的提交。
printf '正在检查 GitHub 远端...\n'
git fetch origin "$branch"
if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    if ! git merge-base --is-ancestor "origin/$branch" HEAD; then
        die "GitHub 上存在本地尚未合并的提交。请先运行：git pull --rebase origin $branch"
    fi
fi

# .gitignore 会自动排除运行结果、数据集、日志、缓存和本地密钥。
git add -A

if git diff --cached --quiet; then
    printf '没有需要提交的变更。\n'
    exit 0
fi

# 即使敏感文件曾被强制跟踪，也不允许脚本继续提交。
while IFS= read -r -d '' path; do
    filename=${path##*/}
    case "$filename" in
        .env|.env.*|openai_key.txt|*.pem|*.key)
            die "检测到可能包含密钥的暂存文件：$path"
            ;;
    esac

    if [[ -f "$path" ]]; then
        size=$(wc -c <"$path")
        if (( size > 100 * 1024 * 1024 )); then
            die "文件超过 GitHub 普通 Git 的 100 MiB 限制：$path"
        fi
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR -z)

# 检查本次新增内容中的常见访问令牌格式，不打印令牌本身。
staged_diff=$(mktemp)
trap 'rm -f "$staged_diff"' EXIT
git diff --cached --no-ext-diff --unified=0 >"$staged_diff"
if grep -Eq '^\+.*(sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})' "$staged_diff"; then
    die "新增内容中疑似包含访问令牌，请移除后再提交。"
fi

if (( $# > 0 )); then
    commit_message="$*"
else
    commit_message="Sync code $(date '+%Y-%m-%d %H:%M:%S %Z')"
fi

printf '\n即将提交：\n'
git diff --cached --stat
git commit -m "$commit_message"

printf '\n正在推送到 GitHub...\n'
if ! git push -u origin "$branch"; then
    die "本地提交已经创建，但推送失败；网络恢复后运行 git push 即可。"
fi

printf '\n同步完成：\n'
git log -1 --oneline --decorate
