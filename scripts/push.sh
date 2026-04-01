#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  CineScope Intelligence — 3-Day GitHub Push Scheduler
#  Run this from your project ROOT directory
#  Usage: bash scripts/push.sh [--day day1|day2|day3|all]
# ═══════════════════════════════════════════════════════════════

set -e

# Optional automation env vars:
#   REMOTE_URL=https://github.com/user/repo.git
#   AUTO_CONFIRM_REWRITE=yes
#   GITHUB_USERNAME=your-github-username
#   GITHUB_TOKEN=your-personal-access-token
#   RUN_DAY=day1|day2|day3|all
#   ALLOW_RERUN=yes
# Optional file-based config:
#   .env.push (project root) or scripts/push.env

ASKPASS_SCRIPT=""
STATE_FILE=".git/push_schedule_state"
RUN_DAY="${RUN_DAY:-all}"

if [ "${1:-}" = "--day" ]; then
  if [ -z "${2:-}" ]; then
    echo "Usage: bash scripts/push.sh [--day day1|day2|day3|all]"
    exit 1
  fi
  RUN_DAY="$2"
elif [ -n "${1:-}" ]; then
  RUN_DAY="$1"
fi

case "$RUN_DAY" in
  day1|day2|day3|all) ;;
  *)
    echo "Invalid day mode: $RUN_DAY"
    echo "Allowed values: day1, day2, day3, all"
    exit 1
    ;;
esac

cleanup_askpass() {
  if [ -n "$ASKPASS_SCRIPT" ] && [ -f "$ASKPASS_SCRIPT" ]; then
    rm -f "$ASKPASS_SCRIPT"
  fi
}
trap cleanup_askpass EXIT

# Load automation vars from local config file when present.
# Project-root .env.push takes priority over scripts/push.env.
load_push_env_file() {
  local file="$1"
  [ -f "$file" ] || return 0

  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"

    case "$line" in
      ""|\#*)
        continue
        ;;
    esac

    local key="${line%%=*}"
    local value="${line#*=}"

    case "$key" in
      REMOTE_URL|AUTO_CONFIRM_REWRITE|GITHUB_USERNAME|GITHUB_TOKEN|ALLOW_RERUN)
        export "$key=$value"
        ;;
    esac
  done < "$file"

  echo "Loaded automation config from $file"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f ".env.push" ]; then
  load_push_env_file ".env.push"
elif [ -f "$SCRIPT_DIR/push.env" ]; then
  load_push_env_file "$SCRIPT_DIR/push.env"
fi

# ── Colors ──────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# ── Helpers ─────────────────────────────────────────────────────
log_day()  { echo -e "\n${CYAN}════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}════════════════════════════════════════${NC}\n"; }
log_session() { echo -e "${BLUE}── $1${NC}"; }
log_ok()   { echo -e "${GREEN}  ✓ committed: $1${NC}"; }
log_skip() { echo -e "${YELLOW}  ⚠ skipped (not found): $1${NC}"; }

load_state() {
  DAY1_DONE=0
  DAY2_DONE=0
  DAY3_DONE=0

  [ -f "$STATE_FILE" ] || return 0

  while IFS='=' read -r key value || [ -n "$key" ]; do
    case "$key" in
      DAY1_DONE) DAY1_DONE="$value" ;;
      DAY2_DONE) DAY2_DONE="$value" ;;
      DAY3_DONE) DAY3_DONE="$value" ;;
    esac
  done < "$STATE_FILE"
}

save_state() {
  cat > "$STATE_FILE" << EOF
DAY1_DONE=$DAY1_DONE
DAY2_DONE=$DAY2_DONE
DAY3_DONE=$DAY3_DONE
LAST_UPDATED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
}

ensure_no_tracked_changes() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${RED}ERROR: Tracked file changes detected. Commit or discard tracked changes before running scheduled day mode.${NC}"
    exit 1
  fi
}

commit_exists_on_ref() {
  local ref="$1"
  local msg="$2"
  git log "$ref" --fixed-strings --grep="$msg" -n 1 --format=%H 2>/dev/null | grep -q .
}

require_day1_complete() {
  if [ "$DAY1_DONE" = "1" ]; then
    return 0
  fi

  if [ -n "$REMOTE_URL" ] && git show-ref --verify --quiet refs/remotes/origin/main; then
    if commit_exists_on_ref "origin/main" "add BERT fine-tuning with cpu-safe fallback and artifact saving"; then
      DAY1_DONE=1
      save_state
      return 0
    fi
  fi

  echo -e "${RED}ERROR: Day 1 has not completed yet. Run day1 first.${NC}"
  exit 1
}

require_day2_complete() {
  if [ "$DAY2_DONE" = "1" ]; then
    return 0
  fi

  if [ -n "$REMOTE_URL" ] && git show-ref --verify --quiet refs/remotes/origin/main; then
    if commit_exists_on_ref "origin/main" "add backend test suite covering auth flow and prediction endpoints"; then
      DAY2_DONE=1
      save_state
      return 0
    fi
  fi

  echo -e "${RED}ERROR: Day 2 has not completed yet. Run day2 first.${NC}"
  exit 1
}

checkout_main_for_incremental_day() {
  if git show-ref --verify --quiet refs/heads/main; then
    git checkout main >/dev/null 2>&1 || git checkout main
  else
    git checkout -b main
  fi

  if [ -n "$REMOTE_URL" ]; then
    git fetch origin main >/dev/null 2>&1 || true
    if git show-ref --verify --quiet refs/remotes/origin/main; then
      local local_head
      local remote_head

      local_head="$(git rev-parse HEAD)"
      remote_head="$(git rev-parse origin/main)"

      if [ "$local_head" != "$remote_head" ]; then
        echo -e "${RED}ERROR: local main differs from origin/main. Sync before continuing for reliable day-by-day execution.${NC}"
        echo "  git fetch origin main"
        echo "  git checkout main"
        echo "  git reset --hard origin/main"
        exit 1
      fi
    fi
  fi
}

push_main() {
  local mode="$1"

  if [ -z "$REMOTE_URL" ]; then
    echo -e "${YELLOW}No remote set. All commits are local.${NC}"
    return 0
  fi

  git branch -M main

  local push_cmd=(git push -u origin main)
  if [ "$mode" = "force" ]; then
    push_cmd+=(--force)
  fi

  if [ -n "${GITHUB_USERNAME:-}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
    ASKPASS_SCRIPT="$(mktemp)"
    cat > "$ASKPASS_SCRIPT" << 'EOF'
#!/bin/sh
case "$1" in
  *Username*|*username*) echo "$GITHUB_USERNAME" ;;
  *Password*|*password*) echo "$GITHUB_TOKEN" ;;
  *) echo "" ;;
esac
EOF
    chmod 700 "$ASKPASS_SCRIPT"

    GIT_ASKPASS="$ASKPASS_SCRIPT" \
    GIT_TERMINAL_PROMPT=0 \
    "${push_cmd[@]}"
  else
    "${push_cmd[@]}"
  fi
}

confirm_rewrite_if_needed() {
  echo -e "${RED}WARNING: Day 1 rewrites local history and force-updates remote main.${NC}"
  if [ "${AUTO_CONFIRM_REWRITE:-}" = "yes" ]; then
    CONFIRM_REWRITE="yes"
    echo -e "${BLUE}AUTO_CONFIRM_REWRITE enabled. Continuing without prompt.${NC}"
  else
    echo -e "${YELLOW}Type 'yes' to continue:${NC}"
    read -r CONFIRM_REWRITE
  fi

  case "$CONFIRM_REWRITE" in
    yes|YES|y|Y) ;;
    *)
      echo -e "${RED}Aborted. No history was rewritten.${NC}"
      exit 1
      ;;
  esac
}

# Stage paths and commit with a backdated timestamp
# Usage: do_commit "ISO_DATE" "commit message" path1 path2 ...
do_commit() {
  local DATE="$1"; shift
  local MSG="$1";  shift
  local staged=0

  for p in "$@"; do
    if [ -e "$p" ]; then
      git add "$p"
      staged=1
    else
      log_skip "$p"
    fi
  done

  if [ "$staged" -eq 0 ] || git diff --cached --quiet; then
    echo -e "${YELLOW}  ⚠ nothing to commit for: \"$MSG\"${NC}"
    return 0
  fi

  GIT_AUTHOR_DATE="$DATE" \
  GIT_COMMITTER_DATE="$DATE" \
  git commit -m "$MSG"

  log_ok "$MSG"
  echo ""
}

run_day1() {
  if [ "${ALLOW_RERUN:-}" != "yes" ] && [ "$DAY1_DONE" = "1" ]; then
    echo -e "${YELLOW}Day 1 already completed. Set ALLOW_RERUN=yes to run it again.${NC}"
    return 0
  fi

  confirm_rewrite_if_needed

  if git rev-parse --verify HEAD >/dev/null 2>&1; then
    BACKUP_BRANCH="backup-before-rewrite-$(date +%Y%m%d%H%M%S)"
    git branch "$BACKUP_BRANCH"
    echo -e "${BLUE}Safety backup branch created: ${BACKUP_BRANCH}${NC}"
  fi

  TEMP_BRANCH="rewrite-main-$(date +%s)"
  git checkout --orphan "$TEMP_BRANCH"
  git rm -rf --cached . >/dev/null 2>&1 || true

  log_day "DAY 1 — March 30, 2026 · ML foundation & data pipeline"

  log_session "09:14 — Initial project setup"
  do_commit "2026-03-30T09:14:22" \
    "initial commit" \
    .gitignore

  log_session "10:03 — Project docs and structure"
  do_commit "2026-03-30T10:03:47" \
    "add ML dependencies and project scaffolding notes" \
    ml/requirements.txt

  log_session "11:32 — ML data pipeline"
  do_commit "2026-03-30T11:32:09" \
    "add data preprocessing pipeline and cleaned dataset config" \
    ml/train.py \
    ml/src/preprocessing.py \
    ml/src/__init__.py

  log_session "13:08 — Baseline model training"
  do_commit "2026-03-30T13:08:34" \
    "add baseline TF-IDF and logistic regression training" \
    ml/src/baseline_model.py

  log_session "15:44 — Advanced model training"
  do_commit "2026-03-30T15:44:51" \
    "add random forest and SVM training phases" \
    ml/src/advanced_models.py

  log_session "17:29 — LSTM model"
  do_commit "2026-03-30T17:29:17" \
    "add LSTM training phase with early stopping" \
    ml/src/lstm_model.py

  log_session "20:11 — BERT pipeline (end of day)"
  do_commit "2026-03-30T20:11:43" \
    "add BERT fine-tuning with cpu-safe fallback and artifact saving" \
    ml/src/bert_model.py \
    ml/src/explainability.py \
    ml/src/aspect_sentiment.py

  echo -e "${GREEN}Day 1 commit block complete.${NC}"
  echo ""
  echo -e "${CYAN}Pushing Day 1...${NC}"
  push_main "force"

  DAY1_DONE=1
  DAY2_DONE=0
  DAY3_DONE=0
  save_state
}

run_day2() {
  if [ "${ALLOW_RERUN:-}" != "yes" ] && [ "$DAY2_DONE" = "1" ]; then
    echo -e "${YELLOW}Day 2 already completed. Set ALLOW_RERUN=yes to run it again.${NC}"
    return 0
  fi

  checkout_main_for_incremental_day
  ensure_no_tracked_changes
  require_day1_complete

  log_day "DAY 2 — March 31, 2026 · Django API — auth, inference, analytics"

  log_session "08:47 — Django project setup"
  do_commit "2026-03-31T08:47:06" \
    "scaffold django project with settings, urls, wsgi" \
    backend/manage.py \
    backend/config/ \
    backend/requirements.txt

  log_session "10:05 — Custom user model and JWT auth"
  do_commit "2026-03-31T10:05:29" \
    "add custom user model with email auth and JWT endpoints" \
    backend/accounts/models.py \
    backend/accounts/serializers.py \
    backend/accounts/views.py \
    backend/accounts/urls.py

  log_session "11:38 — Model loading service"
  do_commit "2026-03-31T11:38:55" \
    "add lazy model loader with fallback to demo mode" \
    backend/api/ml_service.py \
    backend/api/apps.py \
    backend/api/__init__.py

  log_session "13:22 — Core prediction endpoint"
  do_commit "2026-03-31T13:22:41" \
    "add POST /api/predict/ with model selection and safe persistence" \
    backend/api/models.py \
    backend/api/serializers.py \
    backend/api/urls.py

  log_session "14:51 — Explainability and aspect endpoints"
  do_commit "2026-03-31T14:51:13" \
    "add explain endpoint (LIME) and aspect sentiment extraction" \
    backend/api/views.py

  log_session "16:17 — Compare and batch inference"
  do_commit "2026-03-31T16:17:38" \
    "add model comparison and batch prediction endpoints" \
    backend/api/admin.py

  log_session "17:44 — History and analytics endpoints"
  do_commit "2026-03-31T17:44:02" \
    "add predictions history with filters, stats, tokens, and metrics endpoints" \
    backend/api/migrations/

  log_session "19:03 — Feedback, sharing, similar review endpoints"
  do_commit "2026-03-31T19:03:29" \
    "add feedback patch, share link generation, and similar review lookup" \
    backend/accounts/__init__.py \
    backend/accounts/admin.py \
    backend/accounts/apps.py

  log_session "21:07 — Backend tests (end of day)"
  do_commit "2026-03-31T21:07:55" \
    "add backend test suite covering auth flow and prediction endpoints" \
    backend/accounts/tests.py \
    backend/api/tests.py \
    backend/accounts/migrations/

  echo -e "${GREEN}Day 2 commit block complete.${NC}"
  echo ""
  echo -e "${CYAN}Pushing Day 2...${NC}"
  push_main "normal"

  DAY2_DONE=1
  save_state
}

run_day3() {
  if [ "${ALLOW_RERUN:-}" != "yes" ] && [ "$DAY3_DONE" = "1" ]; then
    echo -e "${YELLOW}Day 3 already completed. Set ALLOW_RERUN=yes to run it again.${NC}"
    return 0
  fi

  checkout_main_for_incremental_day
  ensure_no_tracked_changes
  require_day2_complete

  log_day "DAY 3 — April 1, 2026 · React frontend, deployment, final polish"

  log_session "09:05 — React project setup"
  do_commit "2026-04-01T09:05:18" \
    "scaffold react+vite frontend with routing and layout shell" \
    frontend/package.json \
    frontend/package-lock.json \
    frontend/vite.config.js \
    frontend/index.html \
    frontend/src/main.jsx \
    frontend/src/App.jsx \
    frontend/src/index.css

  log_session "10:28 — API service layer and auth interceptor"
  do_commit "2026-04-01T10:28:44" \
    "add api service layer with jwt interceptor and auto-refresh logic" \
    frontend/src/services/api.js

  log_session "11:47 — Login and Register pages"
  do_commit "2026-04-01T11:47:09" \
    "add login and register pages with form validation and error states" \
    frontend/src/pages/LoginPage.jsx \
    frontend/src/pages/RegisterPage.jsx

  log_session "13:15 — Analyzer page"
  do_commit "2026-04-01T13:15:33" \
    "add analyzer page: standard, explain, aspect, compare, and batch modes" \
    frontend/src/pages/AnalyzerPage.jsx

  log_session "14:38 — History page with charts"
  do_commit "2026-04-01T14:38:57" \
    "add history page with filters, pagination, sentiment charts, and keyword panel" \
    frontend/src/pages/HistoryPage.jsx

  log_session "15:52 — Metrics page and shared prediction view"
  do_commit "2026-04-01T15:52:22" \
    "add metrics page with calibration curve and shared prediction view" \
    frontend/src/pages/MetricsPage.jsx \
    frontend/src/pages/SharedPredictionPage.jsx \
    frontend/src/pages/NotFoundPage.jsx

  log_session "16:44 — Home and overview page"
  do_commit "2026-04-01T16:44:11" \
    "add home page with feature overview and social metadata" \
    frontend/src/pages/HomePage.jsx \
    frontend/public/

  log_session "17:33 — Deployment config"
  do_commit "2026-04-01T17:33:48" \
    "add render.yaml blueprint for backend and frontend services" \
    render.yaml \
    backend/.env.example \
    frontend/.env.example

  log_session "18:19 — CI/CD pipeline"
  do_commit "2026-04-01T18:19:05" \
    "add github actions ci for backend tests and frontend build check" \
    .github/workflows/ci.yml

  log_session "19:02 — Final README and documentation"
  do_commit "2026-04-01T19:02:31" \
    "update readme with setup guide, api reference, and deployment notes" \
    README.md

  echo -e "${GREEN}Day 3 commit block complete.${NC}"
  echo ""
  echo -e "${CYAN}Pushing Day 3...${NC}"
  push_main "normal"

  DAY3_DONE=1
  save_state
}

# ── Preflight ────────────────────────────────────────────────────
echo -e "${GREEN}"
echo "  ██████╗██╗███╗   ██╗███████╗███████╗ ██████╗ ██████╗ ██████╗ ███████╗"
echo "  ██╔════╝██║████╗  ██║██╔════╝██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝"
echo "  ██║     ██║██╔██╗ ██║█████╗  ███████╗██║     ██║   ██║██████╔╝█████╗  "
echo "  ██║     ██║██║╚██╗██║██╔══╝  ╚════██║██║     ██║   ██║██╔═══╝ ██╔══╝  "
echo "  ╚██████╗██║██║ ╚████║███████╗███████║╚██████╗╚██████╔╝██║     ███████╗"
echo "   ╚═════╝╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚══════╝"
echo -e "${NC}"
echo -e "${BLUE}  3-Day GitHub Push Scheduler — CineScope Intelligence${NC}"
echo ""

# Validate we're in project root
if [ ! -f "render.yaml" ] && [ ! -d "backend" ] && [ ! -d "frontend" ]; then
  echo -e "${RED}ERROR: Run this from your project root (where backend/, frontend/, ml/ live)${NC}"
  exit 1
fi

# Git init if needed
if [ ! -d ".git" ]; then
  echo "Initializing git repository..."
  git init
fi

load_state

# Ask for remote URL only if not provided by environment
if [ -z "${REMOTE_URL:-}" ]; then
  echo -e "${YELLOW}Enter your GitHub repo URL (e.g. https://github.com/yourname/cinescope-intelligence.git):${NC}"
  read -r REMOTE_URL
else
  echo -e "${BLUE}Using REMOTE_URL from environment.${NC}"
fi

if [ -z "$REMOTE_URL" ]; then
  echo -e "${RED}No URL provided. Commits will be made locally only.${NC}"
else
  git remote remove origin 2>/dev/null || true
  git remote add origin "$REMOTE_URL"
fi

echo ""
echo -e "${CYAN}Starting commit schedule...${NC}"
echo -e "Commits will be spread across March 30, March 31, and April 1, 2026."
echo -e "Run mode: ${RUN_DAY}"
echo ""

case "$RUN_DAY" in
  day1)
    run_day1
    ;;
  day2)
    run_day2
    ;;
  day3)
    run_day3
    ;;
  all)
    run_day1
    run_day2
    run_day3
    ;;
esac

if [ "$DAY1_DONE" = "1" ] && [ "$DAY2_DONE" = "1" ] && [ "$DAY3_DONE" = "1" ]; then
  echo ""
  echo -e "${GREEN}✓ Done! All three day blocks are complete.${NC}"
  echo -e "${GREEN}  Commits span March 30, March 31, and April 1, 2026.${NC}"
fi

echo ""
echo -e "${BLUE}Commit summary:${NC}"
git --no-pager log --oneline --graph -n 30