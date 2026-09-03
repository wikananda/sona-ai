#!/bin/bash

set -u

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
frontend_dir="$project_root/frontend"
runtime_dir="${SONA_RUNTIME_DIR:-$project_root/.sona-runtime}"
backend_port="${SONA_BACKEND_PORT:-8000}"
frontend_port="${SONA_FRONTEND_PORT:-3000}"
conda_environment="${SONA_CONDA_ENV:-sona-ai}"
startup_timeout="${SONA_START_TIMEOUT_SECONDS:-1800}"
backend_url="http://127.0.0.1:$backend_port"
frontend_url="http://127.0.0.1:$frontend_port"
browser_url="http://localhost:$frontend_port"

launcher_pid_file="$runtime_dir/launcher.pid"
backend_pid_file="$runtime_dir/backend.pid"
frontend_pid_file="$runtime_dir/frontend.pid"
nemotron_pid_file="$runtime_dir/nemotron.pid"
whisper_live_marker="$runtime_dir/whisper-live.started"
backend_log="$runtime_dir/backend.log"
frontend_log="$runtime_dir/frontend.log"
sidecar_log="$runtime_dir/sidecars.log"

backend_pid=""
frontend_pid=""
nemotron_pid=""
whisper_live_started=0
cleanup_started=0

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    return 1
}

is_positive_integer() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) [ "$1" -gt 0 ] ;;
    esac
}

read_pid() {
    pid_file="$1"
    [ -f "$pid_file" ] || return 1
    pid_value="$(tr -d '[:space:]' < "$pid_file")"
    case "$pid_value" in
        ''|*[!0-9]*) return 1 ;;
    esac
    printf '%s\n' "$pid_value"
}

process_is_running() {
    process_pid="$1"
    case "$process_pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    kill -0 "$process_pid" 2>/dev/null
}

process_command() {
    ps -p "$1" -o command= 2>/dev/null || true
}

pid_matches() {
    candidate_pid="$1"
    expected_text="$2"
    process_is_running "$candidate_pid" || return 1
    case "$(process_command "$candidate_pid")" in
        *"$expected_text"*) return 0 ;;
        *) return 1 ;;
    esac
}

write_pid() {
    pid_value="$1"
    pid_file="$2"
    temporary_file="$pid_file.tmp.$$"
    printf '%s\n' "$pid_value" > "$temporary_file"
    mv "$temporary_file" "$pid_file"
}

remove_pid_if_owned() {
    pid_file="$1"
    owned_pid="$2"
    saved_pid="$(read_pid "$pid_file" 2>/dev/null || true)"
    if [ -n "$owned_pid" ] && [ "$saved_pid" = "$owned_pid" ]; then
        rm -f "$pid_file"
    fi
}

resolve_conda() {
    if [ -n "${SONA_CONDA_BIN:-}" ] && [ -x "$SONA_CONDA_BIN" ]; then
        printf '%s\n' "$SONA_CONDA_BIN"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return 0
    fi
    user_home="${HOME:-}"
    for candidate in \
        "$user_home/miniconda3/bin/conda" \
        "$user_home/anaconda3/bin/conda" \
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda" \
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
        "/usr/local/miniconda3/bin/conda"
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

env_flag_enabled() {
    flag_name="$1"
    flag_value="$(printenv "$flag_name" 2>/dev/null || true)"
    if [ -z "$flag_value" ] && [ -f "$project_root/.env" ]; then
        flag_value="$(awk -F= -v key="$flag_name" '
            $1 == key {
                sub(/^[^=]*=/, "")
                gsub(/^[[:space:]\047\"]+|[[:space:]\047\"]+$/, "")
                print
                exit
            }
        ' "$project_root/.env")"
    fi
    case "$flag_value" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

http_ready() {
    curl --silent --fail --max-time 2 --output /dev/null "$1" 2>/dev/null
}

port_listener() {
    port="$1"
    if [ -x /usr/sbin/lsof ]; then
        /usr/sbin/lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1
    elif command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1
    fi
}

ensure_port_available() {
    port="$1"
    service_name="$2"
    listener="$(port_listener "$port")"
    if [ -n "$listener" ]; then
        fail "$service_name cannot start because port $port is already used by PID $listener."
        return 1
    fi
    return 0
}

wait_for_service() {
    service_name="$1"
    health_url="$2"
    service_pid="$3"
    timeout_seconds="$4"
    started_at="$(date +%s)"
    next_update=10

    say "Waiting for $service_name..."
    while :; do
        if http_ready "$health_url"; then
            say "  $service_name is ready."
            return 0
        fi
        if ! process_is_running "$service_pid"; then
            fail "$service_name exited during startup."
            return 1
        fi
        now="$(date +%s)"
        elapsed=$((now - started_at))
        if [ "$elapsed" -ge "$timeout_seconds" ]; then
            fail "$service_name did not become ready within ${timeout_seconds}s."
            return 1
        fi
        if [ "$elapsed" -ge "$next_update" ]; then
            say "  Still loading $service_name (${elapsed}s)..."
            next_update=$((next_update + 10))
        fi
        sleep 1
    done
}

show_log_tail() {
    log_file="$1"
    label="$2"
    if [ -s "$log_file" ]; then
        say ""
        say "Last $label messages:"
        tail -n 30 "$log_file"
    fi
}

terminate_tree() {
    tree_pid="$1"
    process_is_running "$tree_pid" || return 0
    child_pids="$(pgrep -P "$tree_pid" 2>/dev/null || true)"
    for child_pid in $child_pids; do
        terminate_tree "$child_pid"
    done
    kill -TERM "$tree_pid" 2>/dev/null || true
}

wait_until_stopped() {
    stopped_pid="$1"
    wait_seconds="${2:-20}"
    waited=0
    while process_is_running "$stopped_pid" && [ "$waited" -lt "$wait_seconds" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    ! process_is_running "$stopped_pid"
}

stop_whisper_live_if_managed() {
    [ -f "$whisper_live_marker" ] || return 0
    if command -v docker >/dev/null 2>&1; then
        say "Stopping managed WhisperLive sidecar..."
        docker compose -f "$project_root/compose.whisper-live.yml" stop >> "$sidecar_log" 2>&1 || true
    fi
    rm -f "$whisper_live_marker"
}

cleanup_children() {
    [ "$cleanup_started" -eq 0 ] || return 0
    cleanup_started=1

    if [ -n "$nemotron_pid" ] && process_is_running "$nemotron_pid"; then
        say "Stopping Nemotron..."
        terminate_tree "$nemotron_pid"
        wait_until_stopped "$nemotron_pid" 20 || true
    fi
    if [ -n "$frontend_pid" ] && process_is_running "$frontend_pid"; then
        say "Stopping Sona web..."
        terminate_tree "$frontend_pid"
        wait_until_stopped "$frontend_pid" 20 || true
    fi
    if [ -n "$backend_pid" ] && process_is_running "$backend_pid"; then
        say "Stopping Sona backend..."
        terminate_tree "$backend_pid"
        wait_until_stopped "$backend_pid" 30 || true
    fi
    if [ "$whisper_live_started" -eq 1 ]; then
        stop_whisper_live_if_managed
    fi

    remove_pid_if_owned "$nemotron_pid_file" "$nemotron_pid"
    remove_pid_if_owned "$frontend_pid_file" "$frontend_pid"
    remove_pid_if_owned "$backend_pid_file" "$backend_pid"
    remove_pid_if_owned "$launcher_pid_file" "$$"
}

handle_shutdown() {
    say ""
    say "Shutting down Sona..."
    cleanup_children
    exit 0
}

open_browser() {
    if env_flag_enabled SONA_NO_BROWSER; then
        say "Open $browser_url in your browser."
        return 0
    fi
    if command -v open >/dev/null 2>&1; then
        open "$browser_url"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$browser_url" >/dev/null 2>&1 &
    else
        say "Open $browser_url in your browser."
    fi
}

start_optional_sidecars() {
    if env_flag_enabled SONA_LAUNCH_WHISPERLIVE; then
        if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
            say "Starting optional WhisperLive sidecar..."
            if docker compose -f "$project_root/compose.whisper-live.yml" up -d >> "$sidecar_log" 2>&1; then
                whisper_live_started=1
                : > "$whisper_live_marker"
            else
                say "Warning: WhisperLive could not start; Sona will use its local fallback."
            fi
        else
            say "Warning: Docker is unavailable; skipping optional WhisperLive."
        fi
    fi

    if env_flag_enabled SONA_LAUNCH_NEMOTRON; then
        say "Starting optional Nemotron sidecar..."
        if [ -f "$project_root/.env" ]; then
            (
                cd "$project_root" || exit 1
                exec "$conda_bin" run --no-capture-output -n "$conda_environment" \
                    dotenv -f "$project_root/.env" run -- \
                    "$project_root/tools/nemotron/run_server.sh"
            ) >> "$sidecar_log" 2>&1 &
        else
            (
                cd "$project_root" || exit 1
                exec "$project_root/tools/nemotron/run_server.sh"
            ) >> "$sidecar_log" 2>&1 &
        fi
        nemotron_pid=$!
        write_pid "$nemotron_pid" "$nemotron_pid_file"
        sleep 1
        if ! process_is_running "$nemotron_pid"; then
            say "Warning: Nemotron could not start; see $sidecar_log"
            remove_pid_if_owned "$nemotron_pid_file" "$nemotron_pid"
            nemotron_pid=""
        fi
    fi
}

validate_startup() {
    if ! is_positive_integer "$backend_port" || [ "$backend_port" -gt 65535 ]; then
        fail "SONA_BACKEND_PORT must be between 1 and 65535."
        return 1
    fi
    if ! is_positive_integer "$frontend_port" || [ "$frontend_port" -gt 65535 ]; then
        fail "SONA_FRONTEND_PORT must be between 1 and 65535."
        return 1
    fi
    if ! is_positive_integer "$startup_timeout"; then
        fail "SONA_START_TIMEOUT_SECONDS must be a positive integer."
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        fail "curl is required to perform startup health checks."
        return 1
    fi
    conda_bin="$(resolve_conda)" || {
        fail "Conda was not found. Install Miniconda or set SONA_CONDA_BIN."
        return 1
    }
    if ! "$conda_bin" run -n "$conda_environment" python -c "pass" >/dev/null 2>&1; then
        fail "The Conda environment '$conda_environment' is unavailable. Follow Backend Setup in README.md."
        return 1
    fi
    next_bin="$frontend_dir/node_modules/.bin/next"
    if [ ! -x "$next_bin" ]; then
        fail "Frontend dependencies are missing. Run 'cd frontend && npm install' once."
        return 1
    fi
    ensure_port_available "$backend_port" "Sona backend" || return 1
    ensure_port_available "$frontend_port" "Sona web" || return 1
    return 0
}

start_sona() {
    mkdir -p "$runtime_dir" || return 1
    chmod 700 "$runtime_dir" 2>/dev/null || true

    existing_launcher="$(read_pid "$launcher_pid_file" 2>/dev/null || true)"
    if [ -n "$existing_launcher" ] && pid_matches "$existing_launcher" "sona_launcher.sh start"; then
        say "Sona is already starting or running."
        if http_ready "$frontend_url"; then
            open_browser
        else
            say "Watch the original launcher window for startup progress."
        fi
        return 0
    fi
    rm -f "$launcher_pid_file"

    if [ -f "$backend_pid_file" ] || [ -f "$frontend_pid_file" ] \
        || [ -f "$nemotron_pid_file" ] || [ -f "$whisper_live_marker" ]; then
        say "Cleaning up an interrupted Sona session..."
        orphan_result=0
        stop_managed_pid "$nemotron_pid_file" "nemotron/run_server.sh" "orphaned Nemotron" \
            || orphan_result=1
        stop_managed_pid "$frontend_pid_file" "next dev" "orphaned Sona web" \
            || orphan_result=1
        stop_managed_pid "$backend_pid_file" "sona_ai.api.main:app" "orphaned Sona backend" \
            || orphan_result=1
        stop_whisper_live_if_managed
        [ "$orphan_result" -eq 0 ] || return 1
    fi

    validate_startup || return 1
    if env_flag_enabled SONA_LAUNCHER_DRY_RUN; then
        say "Sona launcher check passed."
        say "Backend: $backend_url (Conda environment: $conda_environment)"
        say "Frontend: $frontend_url"
        say "Browser: $browser_url"
        return 0
    fi

    : > "$backend_log"
    : > "$frontend_log"
    : > "$sidecar_log"
    write_pid "$$" "$launcher_pid_file"
    trap handle_shutdown INT TERM HUP
    trap cleanup_children EXIT

    start_optional_sidecars

    say "Starting Sona backend..."
    if [ -f "$project_root/.env" ]; then
        (
            cd "$project_root" || exit 1
            export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
            exec "$conda_bin" run --no-capture-output -n "$conda_environment" \
                dotenv -f "$project_root/.env" run --override -- \
                uvicorn sona_ai.api.main:app \
                --app-dir "$project_root/backend/src" \
                --host 127.0.0.1 \
                --port "$backend_port"
        ) >> "$backend_log" 2>&1 &
    else
        (
            cd "$project_root" || exit 1
            export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
            exec "$conda_bin" run --no-capture-output -n "$conda_environment" \
                uvicorn sona_ai.api.main:app \
                --app-dir "$project_root/backend/src" \
                --host 127.0.0.1 \
                --port "$backend_port"
        ) >> "$backend_log" 2>&1 &
    fi
    backend_pid=$!
    write_pid "$backend_pid" "$backend_pid_file"

    say "Starting Sona web..."
    (
        cd "$frontend_dir" || exit 1
        export NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-$backend_url}"
        exec "$next_bin" dev --hostname 127.0.0.1 --port "$frontend_port"
    ) >> "$frontend_log" 2>&1 &
    frontend_pid=$!
    write_pid "$frontend_pid" "$frontend_pid_file"

    say "The first launch may download and load the configured speech model."
    if ! wait_for_service "backend" "$backend_url/docs" "$backend_pid" "$startup_timeout"; then
        show_log_tail "$backend_log" "backend"
        return 1
    fi
    if ! wait_for_service "web app" "$frontend_url" "$frontend_pid" 120; then
        show_log_tail "$frontend_log" "frontend"
        return 1
    fi

    say ""
    say "Sona is ready: $browser_url"
    say "Backend log: $backend_log"
    say "Frontend log: $frontend_log"
    say "Keep this window open while using Sona. Press Control-C to stop it."
    open_browser

    while process_is_running "$backend_pid" && process_is_running "$frontend_pid"; do
        sleep 2
    done

    if ! process_is_running "$backend_pid"; then
        fail "The backend stopped unexpectedly."
        show_log_tail "$backend_log" "backend"
    else
        fail "The web app stopped unexpectedly."
        show_log_tail "$frontend_log" "frontend"
    fi
    return 1
}

stop_managed_pid() {
    pid_file="$1"
    expected_text="$2"
    service_name="$3"
    managed_pid="$(read_pid "$pid_file" 2>/dev/null || true)"
    [ -n "$managed_pid" ] || return 0
    if pid_matches "$managed_pid" "$expected_text"; then
        say "Stopping $service_name..."
        terminate_tree "$managed_pid"
        wait_until_stopped "$managed_pid" 30 || {
            say "Warning: $service_name is still shutting down."
            return 1
        }
    fi
    rm -f "$pid_file"
    return 0
}

stop_sona() {
    mkdir -p "$runtime_dir" || return 1
    launcher_pid="$(read_pid "$launcher_pid_file" 2>/dev/null || true)"
    if [ -n "$launcher_pid" ] && pid_matches "$launcher_pid" "sona_launcher.sh start"; then
        say "Asking the Sona launcher to shut everything down..."
        kill -TERM "$launcher_pid" 2>/dev/null || true
        if wait_until_stopped "$launcher_pid" 45; then
            say "Sona stopped."
            return 0
        fi
        say "Warning: the launcher did not finish shutdown; checking managed services."
    fi
    rm -f "$launcher_pid_file"

    stop_result=0
    stop_managed_pid "$nemotron_pid_file" "nemotron/run_server.sh" "Nemotron" || stop_result=1
    stop_managed_pid "$frontend_pid_file" "next dev" "Sona web" || stop_result=1
    stop_managed_pid "$backend_pid_file" "sona_ai.api.main:app" "Sona backend" || stop_result=1
    stop_whisper_live_if_managed

    if [ "$stop_result" -eq 0 ]; then
        say "Sona is not running."
    fi
    return "$stop_result"
}

show_status() {
    launcher_pid="$(read_pid "$launcher_pid_file" 2>/dev/null || true)"
    launcher_status="stopped"
    backend_status="stopped"
    frontend_status="stopped"
    if [ -n "$launcher_pid" ] && pid_matches "$launcher_pid" "sona_launcher.sh start"; then
        launcher_status="$launcher_pid (running)"
    fi
    if http_ready "$backend_url/docs"; then backend_status="ready"; fi
    if http_ready "$frontend_url"; then frontend_status="ready"; fi

    say "Sona launcher: $launcher_status"
    say "Backend: $backend_status ($backend_url)"
    say "Frontend: $frontend_status ($frontend_url)"
    say "Logs: $runtime_dir"
}

PATH="/opt/homebrew/bin:/usr/local/bin:${PATH:-/usr/bin:/bin}"
export PATH

command_name="${1:-start}"
case "$command_name" in
    start) start_sona ;;
    stop) stop_sona ;;
    status) show_status ;;
    *)
        fail "Unknown command '$command_name'. Use start, stop, or status."
        exit 2
        ;;
esac
