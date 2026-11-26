#!/usr/bin/env bash
# One-shot setup on a 42 school computer (Linux, small home quota, big /sgoinfre).
#
#   git clone <repo> rag-against-the-machine && cd rag-against-the-machine
#   # put vllm-0.10.1.zip and datasets_public.zip here (or anywhere in $HOME)
#   make school                 # setup, weights, index, search, evaluate
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
#   --sgo DIR      cache folder (default: the first writable goinfre/$USER/rag-cache)
#   --zips DIR     folder that holds vllm-0.10.1.zip and datasets_public.zip
#   --no-prefetch  only prepare folders and unpack the zips
#   --yes          do not ask before moving a cache out of home
#   --uninstall    remove the venv, the corpus and the caches (the clone is kept)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
ARGS=("$@")
CACHE=""; ZIPS=""; PREFETCH=1; YES=0; UNINSTALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sgo) CACHE="$2"; shift 2 ;;
    --zips) ZIPS="$2"; shift 2 ;;
    --no-prefetch) PREFETCH=0; shift ;;
    --yes|-y) YES=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
done

# Campuses do not mount the big disk in the same place. Use a directory that
# already exists and is writable; never try to create /sgoinfre itself.
find_goinfre() {
  local dir parent
  local -a found=()
  local -a candidates=(
    "/sgoinfre/goinfre/Perso/$USER"
    "/sgoinfre/students/$USER"
    "/sgoinfre/$USER"
    "/goinfre/$USER"
    "$HOME/sgoinfre"
    "$HOME/goinfre"
  )
  for dir in "${candidates[@]}"; do
    [[ -d "$dir" && -w "$dir" ]] || continue
    if [[ -f "$dir/rag-cache/state/done" ]]; then
      echo "$dir"
      return 0
    fi
    found+=("$dir")
  done
  if [[ ${#found[@]} -gt 0 ]]; then
    echo "${found[0]}"
    return 0
  fi
  for parent in /sgoinfre/goinfre/Perso /sgoinfre/students /sgoinfre /goinfre; do
    [[ -d "$parent" && -w "$parent" ]] || continue
    mkdir -p "$parent/$USER" && chmod 700 "$parent/$USER" || continue
    echo "$parent/$USER"
    return 0
  done
  return 1
}

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

if [[ -n "$CACHE" ]]; then
  SGO_BASE="$(dirname "$CACHE")"
else
  SGO_BASE="$(find_goinfre)" || die "no writable goinfre for $USER
  Looked for /sgoinfre/goinfre/Perso/$USER, /sgoinfre/students/$USER,
  /sgoinfre/$USER, /goinfre/$USER, ~/sgoinfre and ~/goinfre.
  Create your folder there (chmod 700) or rerun with --sgo DIR"
  CACHE="$SGO_BASE/rag-cache"
fi
REPO_DEST="$SGO_BASE/rag-against-the-machine"
STATE="$CACHE/state"; LOGS="$CACHE/logs"

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
[[ -d "$SGO_BASE" && -w "$SGO_BASE" ]] || die "cannot write $SGO_BASE — pass --sgo DIR"
ok "goinfre: $SGO_BASE"
mkdir -p "$CACHE" "$LOGS" "$STATE" || die "cannot write $CACHE"
chmod 700 "$CACHE" 2>/dev/null || true
did setup || mark setup
ok "caches in $CACHE ($(df -h "$CACHE" | awk 'NR==2{print $4}') free), home has $(df -h "$HOME" | awk 'NR==2{print $4}') free"
free_kb=$(df -Pk "$CACHE" | awk 'NR==2{print $4}')
[[ "${free_kb:-0}" -ge 8000000 ]] || warn "less than 8 GB free on $CACHE — PyTorch and Qwen3 need about that"

# ------------------------------------------------------------------ 2. clone stays put
# Moving the clone onto goinfre deletes it on the home filesystem (mv across
# devices). The shell is still inside that directory, so the next `make`
# dies with "getcwd: No such file or directory". Heavy files are symlinks.
say "2. Clone"
root_phys=$(phys "$ROOT")
goinfre_dev=$(df -P "$SGO_BASE" | awk 'NR==2{print $1}')
clone_dev=$(df -P "$root_phys" | awk 'NR==2{print $1}')
if [[ "$clone_dev" == "$goinfre_dev" ]]; then
  ok "clone is on the goinfre disk ($root_phys)"
else
  ok "clone stays at $root_phys"
  ok ".venv and data/ go to $CACHE, so PyTorch is not stored in home"
  mark repo-stayed
fi

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
  local base="$1" dir hit
  local -a dirs=()
  [[ -n "$ZIPS" ]] && dirs+=("$ZIPS")
  dirs+=("$ROOT" "$ROOT/.." "$HOME/Downloads" "$HOME/Desktop" "$SGO_BASE")
  for dir in "${dirs[@]}"; do
    [[ -f "$dir/$base" ]] && echo "$dir/$base" && return 0
  done
  hit=$(find "$HOME" "$SGO_BASE" -maxdepth 5 -type f -name "$base" \
    -not -path '*/.venv/*' -not -path '*/rag-cache/*' 2>/dev/null | head -1)
  [[ -n "$hit" ]] && echo "$hit" && return 0
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
  missing=1
  warn "no vllm-0.10.1.zip"
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
  missing=1
  warn "no datasets_public.zip"
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

if [[ "${missing:-0}" == 1 && $PREFETCH == 1 ]]; then
  die "put vllm-0.10.1.zip and datasets_public.zip in this directory
  (they are the intra attachments; they are not in git) and run make school again"
fi

# ------------------------------------------------------------------ 7. install, weights, index, evaluate
if [[ $PREFETCH == 1 ]]; then
  say "6. Install, models, index, evaluate"
  cd "$ROOT"
  uv sync || die "uv sync failed"
  ok "dependencies"
  uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-0.6B"); snapshot_download("sentence-transformers/all-MiniLM-L6-v2")' \
    || die "could not download Qwen3 or MiniLM"
  ok "Qwen3-0.6B and MiniLM"
  uv run python -m src index --max_chunk_size 2000 || die "index failed"
  ok "index"
  for ds in code docs; do
    uv run python -m src search_dataset \
      --dataset_path "data/datasets/UnansweredQuestions/dataset_${ds}_public.json" \
      --k 5 \
      --save_directory "data/output/search_results/UnansweredQuestions" \
      || die "search failed on $ds"
    echo "== $ds"
    uv run python -m src evaluate \
      --student_search_results_path "data/output/search_results/UnansweredQuestions/dataset_${ds}_public.json" \
      --dataset_path "data/datasets/AnsweredQuestions/dataset_${ds}_public.json" \
      || die "evaluate failed on $ds"
  done
fi

# ------------------------------------------------------------------ summary
echo
say "Ready in $((SECONDS - T0)) s.  Clone: $(phys "$ROOT")"
echo "  make purge    # when you leave: venv, corpus, weights, uv cache. The clone stays."
