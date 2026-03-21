#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# imu_dual.sh — Stream two LPMS IMUs by MAC address (no scan needed)
#
# Usage:
#   ./scripts/imu_dual.sh --mac1 AA:BB:CC:DD:EE:FF --mac2 AA:BB:CC:DD:EE:FF \
#                         [--all|--euler|--quat|--raw] [--count N] [--rate N]
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMU_SUBMODULE="$PROJECT_DIR/source/imu"
BUILD_DIR="$IMU_SUBMODULE/build"
LOGGER_BIN="${OPENZEN_LOGGER_BIN:-$BUILD_DIR/examples/OpenZenImuLogger}"

MAC1=""
MAC2=""
LOGGER_ARGS=()

log_info()  { echo -e "\033[1;34m[*]\033[0m $*"; }
log_ok()    { echo -e "\033[1;32m[✓]\033[0m $*"; }
log_warn()  { echo -e "\033[1;33m[!]\033[0m $*"; }
log_error() { echo -e "\033[1;31m[✗]\033[0m $*"; }
die()       { log_error "$*"; exit 1; }

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mac1)   MAC1="$2";  shift 2 ;;
        --mac2)   MAC2="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: $0 --mac1 <MAC> --mac2 <MAC> [--all|--euler|--quat|--raw] [--count N] [--rate N]"
            exit 0 ;;
        *)        LOGGER_ARGS+=("$1"); shift ;;
    esac
done

[[ -n "$MAC1" ]] || die "--mac1 requis   ex: --mac1 00:04:3E:5A:2B:61"
[[ -n "$MAC2" ]] || die "--mac2 requis   ex: --mac2 00:04:3E:6C:52:90"

# ── Prerequisites ────────────────────────────────────────────────────────────
for cmd in bluetoothctl rfkill; do
    command -v "$cmd" &>/dev/null || die "$cmd manquant (sudo apt install bluez rfkill)"
done

rfkill list bluetooth | grep -q "Hard blocked: yes" && die "Bluetooth bloque physiquement"

if rfkill list bluetooth | grep -q "Soft blocked: yes"; then
    log_warn "Bluetooth soft-blocked, deblocage..."
    rfkill unblock bluetooth && sleep 1
fi

if ! systemctl is-active bluetooth &>/dev/null; then
    log_warn "Service bluetooth inactif, demarrage..."
    sudo systemctl start bluetooth && sleep 2
fi

bluetoothctl list &>/dev/null || die "Aucun adaptateur Bluetooth"

# ── Build si nécessaire ───────────────────────────────────────────────────────
if [[ ! -x "$LOGGER_BIN" ]]; then
    log_info "Compilation d'OpenZenImuLogger depuis source/imu..."
    [[ -f "$IMU_SUBMODULE/CMakeLists.txt" ]] || die "Submodule source/imu vide — lance: git submodule update --init source/imu"
    mkdir -p "$BUILD_DIR"
    ( cd "$BUILD_DIR" && cmake .. -DZEN_BLUETOOTH=ON && cmake --build . --target OpenZenImuLogger -j"$(nproc)" )
    [[ -x "$LOGGER_BIN" ]] || die "Echec de compilation."
    log_ok "Compilation OK."
fi

# ── Pair/trust ───────────────────────────────────────────────────────────────
for MAC in "$MAC1" "$MAC2"; do
    PAIRED=$(bluetoothctl info "$MAC" 2>/dev/null | grep "Paired:" | awk '{print $2}' || echo "no")
    if [[ "$PAIRED" != "yes" ]]; then
        log_info "Appairage $MAC..."
        bluetoothctl pair "$MAC"  &>/dev/null || true
        sleep 2
        bluetoothctl trust "$MAC" &>/dev/null || true
        sleep 1
        log_ok "Appaire: $MAC"
    else
        log_ok "Deja appaire: $MAC"
    fi
done

# ── Launch ───────────────────────────────────────────────────────────────────
echo ""
log_info "Streaming IMU 1 : $MAC1"
log_info "Streaming IMU 2 : $MAC2"
echo ""

exec "$LOGGER_BIN" \
    --sensor Bluetooth "$MAC1" \
    --sensor Bluetooth "$MAC2" \
    "${LOGGER_ARGS[@]}"
