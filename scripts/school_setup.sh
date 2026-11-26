#!/usr/bin/env bash
# One-shot setup on a 42 school computer (Linux, small home quota, big /sgoinfre).
#
#   git clone <repo> rag-against-the-machine && cd rag-against-the-machine
#   # put vllm-0.10.1.zip and datasets_public.zip here, or pass --zips DIR
#   make school                 # or: ./scripts/school_setup.sh
#   make run                    # index + search + evaluate, once the log says ALL DONE
#
#   make purge                  # after the defense: venv, corpus, caches. The clone stays.
#
# What goes where
#   home          uv binary only (~40 MB), plus a symlink if the clone was moved
#   sgoinfre      the clone (so .venv from a plain `uv sync` is not in home), the uv
#                 cache, the Python install, the Hugging Face weights and the corpus
#
# A reviewer who then runs `uv sync` and `uv run python -m src ...` from the subject
# does not need this script: the clone is on sgoinfre (or .venv and data/ are
# symlinks), and the uv / Hugging Face caches are symlinked out of home.
#
# Options
#   --sgo DIR      cache folder (default /sgoinfre/goinfre/Perso/$USER/rag-cache)
#   --zips DIR     folder that holds vllm-0.10.1.zip and datasets_public.zip
#   --no-prefetch  do not uv sync, download weights or index in the background
#   --yes          do not ask before moving the clone out of home
#   --uninstall    remove the venv, the corpus and the caches (the clone is kept)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
ARGS=("$@")
SGO_BASE="/sgoinfre/goinfre/Perso/$USER"
CACHE=""; ZIPS=""; PREFETCH=1; YES=0; UNINSTALL=0
REPO_DEST="$SGO_BASE/rag-against-the-machine"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sgo) CACHE="$2"; shift 2 ;;
    --zips) ZIPS="$2"; shift 2 ;;
    --no-prefetch) PREFETCH=0; shift ;;
    --yes|-y) YES=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
done
CACHE="${CACHE:-$SGO_BASE/rag-cache}"
STATE="$CACHE/state"; LOGS="$CACHE/logs"
G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'
say()  { echo "${B}==>${N} $*"; }
ok()   { echo "  ${G}ok${N}  $*"; }
warn() { echo "  ${Y}!!${N}  $*"; }
die()  { echo "  ${R}xx${N}  $*"; exit 1; }
ask()  { [[ $YES == 1 ]] && return 0; read -r -p "  $1 [Y/n] " a; [[ -z "$a" || "$a" =~ ^[Yy] ]]; }
mark() { mkdir -p "$STATE"; echo "$1" >> "$STATE/done"; }
did()  { [[ -f "$STATE/done" ]] && grep -qx "$1" "$STATE/done"; }
phys() { (cd "$1" && pwd -P); }
T0=$SECONDS

# Drop a home path that we created and that points into the cache or the moved clone.
unlink_if_ours() {
  local link="$1" target
  [[ -L "$link" ]] || return 0
  target=$(readlink "$link")
  case "$target/" in
    "$CACHE/"*|"$REPO_DEST/"*) rm -f "$link" && ok "removed symlink $link" ;;
    *) warn "kept $link (it does not point at this setup)" ;;
  esac
}

# ------------------------------------------------------------------ uninstall
if [[ $UNINSTALL == 1 ]]; then
  did setup || die "$CACHE was not created by this script (no $STATE/done): nothing removed — check --sgo"
  say "Removing what school_setup.sh created"
  if [[ -f "$STATE/prefetch.pid" ]]; then
    kill "$(cat "$STATE/prefetch.pid")" 2>/dev/null && ok "stopped the background prefetch"
  fi
  unlink_if_ours "$HOME/.cache/uv"
  unlink_if_ours "$HOME/.cache/huggingface"
  unlink_if_ours "$HOME/.local/share/uv"
  unlink_if_ours "$ROOT/.venv"
  unlink_if_ours "$ROOT/data"
  unlink_if_ours "$ROOT/moulinette"
  # generated weight: the virtualenv and the corpus. Zips and the clone stay.
  rm -rf "$ROOT/.venv" "$ROOT/data" && ok "venv and data/ inside the clone"
  rm -f "$ROOT/.school.env" && ok ".school.env"
  if did uv-installed; then
    rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx" && ok "uv binary"
  fi
  repo_link=""
  [[ -f "$STATE/repo-link" ]] && repo_link=$(cat "$STATE/repo-link")
  chmod -R u+w "$CACHE" 2>/dev/null
  rm -rf "$CACHE" && ok "$CACHE"
  echo
  kept=$(phys "$ROOT")
  echo "Left: the clone at $kept"
  [[ -n "$repo_link" && -L "$repo_link" ]] && echo "Same directory via $repo_link"
  echo "Remove the clone with:  rm -rf \"$kept\""
  exit 0
fi

# ------------------------------------------------------------------ 1. places
say "1. Folders"
[[ "$(uname -s)" == Linux ]] || warn "not Linux: fine for a dry run, the script targets the school computers"
if [[ ! -d "$SGO_BASE" ]]; then
  mkdir -p "$SGO_BASE" 2>/dev/null || die "cannot create $SGO_BASE — is /sgoinfre mounted? use --sgo DIR"
fi
mkdir -p "$CACHE" "$LOGS" "$STATE" || die "cannot write $CACHE"
chmod 700 "$CACHE" 2>/dev/null || true
did setup || mark setup
ok "caches in $CACHE ($(df -h "$CACHE" | awk 'NR==2{print $4}') free), home has $(df -h "$HOME" | awk 'NR==2{print $4}') free"
free_kb=$(df -Pk "$CACHE" | awk 'NR==2{print $4}')
[[ "${free_kb:-0}" -ge 8000000 ]] || warn "less than 8 GB free on $CACHE — PyTorch and Qwen3 need about that"

# ------------------------------------------------------------------ 2. clone out of home
say "2. Clone"
root_phys=$(phys "$ROOT")
case "$root_phys/" in
  "$REPO_DEST/"*) ok "clone is already on sgoinfre ($root_phys)" ;;
  "$HOME"/*)
    warn "this clone is in home; a plain uv sync would put PyTorch there"
    if ask "move it to $REPO_DEST and leave a symlink here?"; then
      [[ -e "$REPO_DEST" ]] && die "$REPO_DEST already exists"
      mv "$root_phys" "$REPO_DEST"
      ln -s "$REPO_DEST" "$root_phys"
      echo "$root_phys" > "$STATE/repo-link"
      mark repo-moved
      ok "clone moved to $REPO_DEST"
      exec "$REPO_DEST/scripts/school_setup.sh" "${ARGS[@]}"
    else
      warn "kept in home: .venv and data/ will be symlinks into $CACHE"
      mark repo-stayed
    fi ;;
  *) ok "clone is outside home ($root_phys); leaving it there" ;;
esac

# A home path that must not hold gigabytes. Existing directories are moved, not copied.
link_out() {
  local link="$1" target="$2" label="$3"
  mkdir -p "$(dirname "$link")"
  if [[ -L "$link" ]]; then
    ok "$label -> $(readlink "$link")"; return 0
  fi
  if [[ -e "$link" ]]; then
    warn "$label is a real directory in home"
    if ask "move $label to $target?"; then
      rm -rf "$target"
      mkdir -p "$(dirname "$target")"
      mv "$link" "$target"
      ln -s "$target" "$link"
      ok "moved $label"
    else
      warn "kept $label in home: watch your quota (df -h ~)"
    fi
    return 0
  fi
  mkdir -p "$target"
  ln -s "$target" "$link"
  ok "$label -> $target"
}

link_project() {
  local name="$1" target="$2"
  local link="$ROOT/$name"
  if [[ -L "$link" ]]; then ok "$name -> $(readlink "$link")"; return 0; fi
  if [[ -e "$link" ]]; then ok "$name stays in the clone"; return 0; fi
  mkdir -p "$target"
  ln -s "$target" "$link"
  ok "$name -> $target"
}

# ------------------------------------------------------------------ 3. caches out of home
say "3. Caches"
link_out "$HOME/.cache/uv" "$CACHE/uv-cache" "uv cache"
link_out "$HOME/.cache/huggingface" "$CACHE/hf" "Hugging Face cache"
link_out "$HOME/.local/share/uv" "$CACHE/uv-data" "uv Python install"
mkdir -p "$CACHE/hf/hub" "$CACHE/uv-data/python"
# Only when the clone itself stayed in home. On sgoinfre, uv sync may create .venv in place.
if did repo-stayed; then
  link_project .venv "$CACHE/venv"
  link_project data "$CACHE/data"
fi

# ------------------------------------------------------------------ 4. uv
say "4. uv"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}')"
else
  command -v curl >/dev/null || die "no curl, and uv is not installed"
  say "installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh \
    || die "uv install failed (no network?)"
  export PATH="$HOME/.local/bin:$PATH"
  hash -r || true
  command -v uv >/dev/null || die "uv installed but not on PATH ($HOME/.local/bin)"
  mark uv-installed
  ok "uv $(uv --version | awk '{print $2}')"
fi

# ------------------------------------------------------------------ 5. shell / make file
{
  echo "# written by scripts/school_setup.sh. make loads it; in a shell: set -a; . ./.school.env; set +a"
  echo "export UV_CACHE_DIR=$CACHE/uv-cache"
  echo "export UV_PYTHON_INSTALL_DIR=$CACHE/uv-data/python"
  echo "export HF_HOME=$CACHE/hf"
  echo "export HF_HUB_CACHE=$CACHE/hf/hub"
  echo "export HUGGINGFACE_HUB_CACHE=$CACHE/hf/hub"
  echo "export PATH=$HOME/.local/bin:$PATH"
} > "$ROOT/.school.env"
ok "wrote .school.env"
export UV_CACHE_DIR="$CACHE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$CACHE/uv-data/python"
export HF_HOME="$CACHE/hf"
export HF_HUB_CACHE="$CACHE/hf/hub"
export HUGGINGFACE_HUB_CACHE="$CACHE/hf/hub"

# ------------------------------------------------------------------ 6. attachments
say "5. Corpus and datasets"
find_zip() {
  local base="$1" dir
  local -a dirs=()
  [[ -n "$ZIPS" ]] && dirs+=("$ZIPS")
  dirs+=("$ROOT" "$ROOT/.." "$HOME/Downloads")
  for dir in "${dirs[@]}"; do
    [[ -f "$dir/$base" ]] && echo "$dir/$base" && return 0
  done
  return 1
}
extract() {
  local zip="$1" dest="$2"
  mkdir -p "$dest"
  if command -v unzip >/dev/null; then unzip -qo "$zip" -d "$dest"
  else python3 -m zipfile -e "$zip" "$dest"; fi
}
mkdir -p "$ROOT/data/raw" "$ROOT/data/datasets"
if [[ -f "$ROOT/data/raw/vllm-0.10.1/README.md" ]]; then
  ok "corpus already at data/raw/vllm-0.10.1"
elif zip=$(find_zip vllm-0.10.1.zip); then
  extract "$zip" "$ROOT/data/raw"
  ok "corpus from $zip"
else
  warn "no vllm-0.10.1.zip — put it in the clone or pass --zips DIR, then re-run"
fi
if [[ -f "$ROOT/data/datasets/AnsweredQuestions/dataset_docs_public.json" ]]; then
  ok "datasets already in data/datasets"
elif zip=$(find_zip datasets_public.zip); then
  tmp=$(mktemp -d)
  extract "$zip" "$tmp"
  mkdir -p "$ROOT/data/datasets"
  cp -a "$tmp/datasets_public/public/." "$ROOT/data/datasets/"
  rm -rf "$tmp"
  ok "datasets from $zip"
else
  warn "no datasets_public.zip — search_dataset needs data/datasets"
fi
if [[ -x "$ROOT/moulinette" ]]; then
  ok "moulinette"
elif zip=$(find_zip moulinette.zip); then
  os_id=$(. /etc/os-release 2>/dev/null && echo "${ID:-}")
  case "$os_id" in
    ubuntu|debian|pop) bin=moulinette-ubuntu ;;
    *) bin=moulinette-fedora ;;
  esac
  if command -v unzip >/dev/null; then unzip -p "$zip" "$bin" > "$CACHE/moulinette"
  else python3 -c 'import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); sys.stdout.buffer.write(z.read(sys.argv[2]))' \
        "$zip" "$bin" > "$CACHE/moulinette"; fi
  chmod +x "$CACHE/moulinette"
  ln -sfn "$CACHE/moulinette" "$ROOT/moulinette"
  ok "moulinette ($bin) -> ./moulinette"
else
  warn "no moulinette.zip — the defense uses the staff binary; yours is only for a local check"
fi

# ------------------------------------------------------------------ 7. background install, weights, index
if [[ $PREFETCH == 1 ]]; then
  say "6. Prefetch"
  if [[ -f "$STATE/prefetch.pid" ]] && kill -0 "$(cat "$STATE/prefetch.pid")" 2>/dev/null; then
    ok "already running (pid $(cat "$STATE/prefetch.pid")) — tail -f $LOGS/prefetch.log"
  else
    nohup env \
      PATH="$PATH" \
      UV_CACHE_DIR="$UV_CACHE_DIR" \
      UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
      HF_HOME="$HF_HOME" \
      HF_HUB_CACHE="$HF_HUB_CACHE" \
      HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
      ROOT="$ROOT" \
      bash -c '
        set -uo pipefail
        cd "$ROOT"
        echo "uv sync"
        uv sync && echo "done sync" || { echo "FAILED sync"; exit 1; }
        echo "download Qwen/Qwen3-0.6B"
        uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-0.6B\")" \
          && echo "done qwen" || echo "FAILED qwen"
        echo "download all-MiniLM-L6-v2"
        uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(\"sentence-transformers/all-MiniLM-L6-v2\")" \
          && echo "done minilm" || echo "FAILED minilm"
        if [[ -f data/raw/vllm-0.10.1/README.md ]]; then
          echo "index"
          uv run python -m src index --max_chunk_size 2000 && echo "done index" || echo "FAILED index"
        else
          echo "skip index (no corpus)"
        fi
        echo "ALL DONE"
      ' > "$LOGS/prefetch.log" 2>&1 &
    echo $! > "$STATE/prefetch.pid"
    ok "uv sync, Qwen3, MiniLM and the index are running in the background"
    ok "progress: tail -f $LOGS/prefetch.log"
  fi
fi

# ------------------------------------------------------------------ summary
echo
say "Ready in $((SECONDS - T0)) s.  Clone: $(phys "$ROOT")"
cat <<EOF

  ${B}make run${N}       # after the log prints ALL DONE: index (if needed) + search + evaluate
  make test      # pytest
  tail -f $LOGS/prefetch.log
  make purge     # when you leave: venv, corpus, weights, uv cache. The clone stays.

  The subject commands work in this shell and in a new one:
    uv run python -m src index --max_chunk_size 2000
EOF
