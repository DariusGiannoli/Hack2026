#!/usr/bin/env bash
set -euo pipefail

#############################################################################
# imu_grab_all.sh — Scan, pair, and stream ALL LPMS IMUs in Bluetooth range
# Zero interaction needed. Just run it.
#
# Usage:  ./scripts/imu_grab_all.sh [--all|--euler|--quat|--raw] [--rate N]
#############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMU_SUBMODULE="$PROJECT_DIR/source/imu"
BUILD_DIR="$IMU_SUBMODULE/build"
LOGGER_BIN="${OPENZEN_LOGGER_BIN:-$BUILD_DIR/examples/OpenZenImuLogger}"

SCAN_TIMEOUT="${SCAN_TIMEOUT:-15}"
DEVICE_PREFIX="LPMS"
LOGGER_ARGS=()

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: $0 [--all|--euler|--quat|--raw|--debug] [--rate N] [--count N]"
            echo ""
            echo "Scanne automatiquement TOUS les capteurs LPMS a portee Bluetooth,"
            echo "les appaire si besoin, et stream les donnees de chacun."
            echo ""
            echo "Variables d'environnement:"
            echo "  SCAN_TIMEOUT=15   Duree du scan BT en secondes"
            exit 0 ;;
    esac
done
LOGGER_ARGS=("$@")

log_info()  { echo -e "\033[1;34m[*]\033[0m $*"; }
log_ok()    { echo -e "\033[1;32m[✓]\033[0m $*"; }
log_warn()  { echo -e "\033[1;33m[!]\033[0m $*"; }
log_error() { echo -e "\033[1;31m[✗]\033[0m $*"; }
die()       { log_error "$*"; exit 1; }

# ── Prerequisites ────────────────────────────────────────────────────────────
for cmd in bluetoothctl rfkill; do
    command -v "$cmd" &>/dev/null || die "$cmd manquant (sudo apt install bluez rfkill)"
done

rfkill list bluetooth | grep -q "Hard blocked: yes" && die "Bluetooth bloque physiquement (hard block)"

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
    [[ -x "$LOGGER_BIN" ]] || die "Echec de compilation. Vérifie les dépendances (cmake, gcc, libbluetooth-dev)."
    log_ok "Compilation OK."
fi

# ── Scan ─────────────────────────────────────────────────────────────────────
log_info "Scan Bluetooth (${SCAN_TIMEOUT}s)..."

bluetoothctl --timeout "$SCAN_TIMEOUT" scan on &>/dev/null &
SCAN_PID=$!
sleep "$((SCAN_TIMEOUT + 2))" 2>/dev/null || true
kill "$SCAN_PID" 2>/dev/null || true
wait "$SCAN_PID" 2>/dev/null || true

mapfile -t FOUND < <(bluetoothctl devices 2>/dev/null | grep -i "$DEVICE_PREFIX" || true)

if [[ ${#FOUND[@]} -eq 0 ]]; then
    die "Aucun capteur $DEVICE_PREFIX a portee. Verifie qu'ils sont allumes et en mode BT."
fi

declare -a MACS=()
declare -a NAMES=()
for line in "${FOUND[@]}"; do
    MAC=$(echo "$line" | awk '{print $2}')
    NAME=$(echo "$line" | cut -d' ' -f3-)
    MACS+=("$MAC")
    NAMES+=("$NAME")
done

log_ok "${#MACS[@]} capteur(s) detecte(s):"
for i in "${!MACS[@]}"; do
    echo -e "    \033[1;36m[$i]\033[0m ${NAMES[$i]}  (${MACS[$i]})"
done

# ── Pair all ─────────────────────────────────────────────────────────────────
log_info "Appairage..."
for MAC in "${MACS[@]}"; do
    PAIRED=$(bluetoothctl info "$MAC" 2>/dev/null | grep "Paired:" | awk '{print $2}' || echo "no")
    if [[ "$PAIRED" != "yes" ]]; then
        bluetoothctl pair "$MAC" &>/dev/null || true
        sleep 2
        bluetoothctl trust "$MAC" &>/dev/null || true
        sleep 1
        log_ok "Appaire: $MAC"
    else
        log_ok "Deja appaire: $MAC"
    fi
done

# ── Launch ───────────────────────────────────────────────────────────────────
SENSOR_ARGS=()
for MAC in "${MACS[@]}"; do
    SENSOR_ARGS+=("--sensor" "Bluetooth" "$MAC")
done

echo ""
log_info "Streaming de ${#MACS[@]} capteur(s) — Ctrl+C pour arreter"
echo ""

exec "$LOGGER_BIN" "${SENSOR_ARGS[@]}" "${LOGGER_ARGS[@]}"
