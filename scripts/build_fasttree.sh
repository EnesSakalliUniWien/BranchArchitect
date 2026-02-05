#!/bin/bash
#
# build_fasttree.sh - Build FastTree from source for the current platform
#
# This script downloads and compiles FastTree, placing the binary in the
# appropriate platform-specific directory under bin/
#
# Usage:
#   ./scripts/build_fasttree.sh           # Build for current platform
#   ./scripts/build_fasttree.sh --check   # Check if FastTree is available
#   ./scripts/build_fasttree.sh --clean   # Remove built binaries
#
# Requirements:
#   - C compiler (gcc, clang, or cc)
#   - curl or wget
#   - macOS: Xcode Command Line Tools (xcode-select --install)
#   - Linux: build-essential (apt install build-essential)
#
# Alternative (no compiler needed):
#   conda install bioconda::fasttree
#

set -e

# Configuration
FASTTREE_VERSION="2.2.0"
FASTTREE_SOURCE_URL="https://raw.githubusercontent.com/morgannprice/fasttree/v${FASTTREE_VERSION}/FastTree.c"
FASTTREE_FALLBACK_URL="https://raw.githubusercontent.com/morgannprice/fasttree/main/FastTree.c"

# Resolve script directory (works even when called from different locations)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

detect_platform() {
    case "$(uname -s)" in
        Darwin) echo "darwin" ;;
        Linux)  echo "linux" ;;
        MINGW*|CYGWIN*|MSYS*) echo "win32" ;;
        *) echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        arm64|aarch64) echo "arm64" ;;
        *) echo "$(uname -m)" ;;
    esac
}

get_compiler_flags() {
    local arch="$1"
    case "$arch" in
        arm64)
            # Apple Silicon (M1/M2/M3) or Linux ARM64
            if [[ "$(uname -s)" == "Darwin" ]]; then
                echo "-march=armv8.4-a"
            else
                echo "-march=armv8-a"
            fi
            ;;
        x86_64)
            echo "-fopenmp-simd -march=x86-64"
            ;;
        *)
            echo ""
            ;;
    esac
}

detect_compiler() {
    # Try compilers in order of preference
    for cc in gcc clang cc; do
        if command -v "$cc" &> /dev/null; then
            echo "$cc"
            return 0
        fi
    done
    return 1
}

download_source() {
    local output="$1"
    local url="$FASTTREE_SOURCE_URL"

    log_info "Downloading FastTree ${FASTTREE_VERSION} source..."

    if command -v curl &> /dev/null; then
        if ! curl -sL --fail -o "$output" "$url" 2>/dev/null; then
            log_warn "Version-specific URL failed, trying main branch..."
            if ! curl -sL --fail -o "$output" "$FASTTREE_FALLBACK_URL"; then
                log_error "Download failed from both URLs"
                return 1
            fi
        fi
    elif command -v wget &> /dev/null; then
        if ! wget -q -O "$output" "$url" 2>/dev/null; then
            log_warn "Version-specific URL failed, trying main branch..."
            if ! wget -q -O "$output" "$FASTTREE_FALLBACK_URL"; then
                log_error "Download failed from both URLs"
                return 1
            fi
        fi
    else
        log_error "Neither curl nor wget found. Please install one of them."
        return 1
    fi

    # Verify download - check file size and content
    if [[ ! -s "$output" ]]; then
        log_error "Downloaded file is empty"
        return 1
    fi

    # Check that it looks like valid FastTree C source code
    if ! head -20 "$output" | grep -q "FastTree" 2>/dev/null; then
        log_error "Downloaded file doesn't appear to be valid C source"
        return 1
    fi

    log_info "Download complete ($(wc -c < "$output" | tr -d ' ') bytes)"
}

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

cmd_check() {
    local platform=$(detect_platform)
    local binary="${BIN_DIR}/${platform}/fasttree"
    local exe_suffix=""
    [[ "$platform" == "win32" ]] && exe_suffix=".exe"
    binary="${binary}${exe_suffix}"

    echo "FastTree Status:"
    echo "  Platform: ${platform} ($(detect_arch))"
    echo "  Expected: ${binary}"

    if [[ -x "$binary" ]]; then
        echo -e "  Status:   ${GREEN}Available${NC}"
        "$binary" -expert 2>&1 | head -1 || true
        return 0
    elif command -v fasttree &> /dev/null; then
        echo -e "  Status:   ${GREEN}Available (system PATH)${NC}"
        which fasttree
        return 0
    else
        echo -e "  Status:   ${RED}Not found${NC}"
        echo ""
        echo "To install FastTree:"
        echo "  Option 1: ./scripts/build_fasttree.sh"
        echo "  Option 2: conda install bioconda::fasttree"
        return 1
    fi
}

cmd_clean() {
    log_info "Cleaning built FastTree binaries..."

    for platform in darwin linux win32; do
        local dir="${BIN_DIR}/${platform}"
        if [[ -d "$dir" ]]; then
            rm -f "${dir}/fasttree" "${dir}/fasttree.exe" "${dir}/FastTreeMP" 2>/dev/null || true
            log_info "Cleaned ${dir}"
        fi
    done

    log_info "Clean complete."
}

cmd_build() {
    local platform=$(detect_platform)
    local arch=$(detect_arch)
    local exe_suffix=""
    [[ "$platform" == "win32" ]] && exe_suffix=".exe"

    log_info "Building FastTree for ${platform}/${arch}..."

    # Check for compiler
    local CC
    if ! CC=$(detect_compiler); then
        log_error "No C compiler found!"
        echo ""
        echo "Please install a C compiler:"
        echo "  macOS:   xcode-select --install"
        echo "  Ubuntu:  sudo apt install build-essential"
        echo "  Fedora:  sudo dnf install gcc"
        echo "  Windows: Install MinGW-w64 or use WSL"
        echo ""
        echo "Or install FastTree via conda (no compiler needed):"
        echo "  conda install bioconda::fasttree"
        exit 1
    fi
    log_info "Using compiler: ${CC}"

    # Get platform-specific flags
    local EXTRA_FLAGS=$(get_compiler_flags "$arch")
    log_info "Compiler flags: -O3 -funsafe-math-optimizations ${EXTRA_FLAGS}"

    # Create output directory
    local OUTPUT_DIR="${BIN_DIR}/${platform}"
    mkdir -p "$OUTPUT_DIR"

    # Download source to temp file (with .c extension for compiler recognition)
    local TEMP_DIR=$(mktemp -d)
    local TEMP_SOURCE="${TEMP_DIR}/FastTree.c"
    trap "rm -rf '$TEMP_DIR'" EXIT

    if ! download_source "$TEMP_SOURCE"; then
        log_error "Failed to download FastTree source"
        exit 1
    fi

    # Build
    local OUTPUT_BINARY="${OUTPUT_DIR}/fasttree${exe_suffix}"
    log_info "Compiling..."

    $CC -O3 -funsafe-math-optimizations $EXTRA_FLAGS \
        -o "$OUTPUT_BINARY" \
        "$TEMP_SOURCE" \
        -lm

    chmod +x "$OUTPUT_BINARY"

    # Verify build
    if [[ -x "$OUTPUT_BINARY" ]]; then
        log_info "Build successful!"
        echo ""
        echo "Binary location: ${OUTPUT_BINARY}"
        echo "Version info:"
        "$OUTPUT_BINARY" -expert 2>&1 | head -3 || true
    else
        log_error "Build failed - binary not created"
        exit 1
    fi
}

cmd_help() {
    cat << EOF
FastTree Build Script

Usage:
  $(basename "$0") [command]

Commands:
  build     Build FastTree for the current platform (default)
  check     Check if FastTree is available
  clean     Remove built binaries
  help      Show this help message

Examples:
  ./scripts/build_fasttree.sh           # Build FastTree
  ./scripts/build_fasttree.sh --check   # Check availability
  ./scripts/build_fasttree.sh --clean   # Clean builds

Alternative Installation (no compiler needed):
  conda install bioconda::fasttree

Environment Variables:
  CC              Override C compiler (default: auto-detect)
  FASTTREE_PATH   Set this to use a custom FastTree binary

For more information:
  https://morgannprice.github.io/fasttree/
EOF
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    local cmd="${1:-build}"

    case "$cmd" in
        build|"")
            cmd_build
            ;;
        --check|check|-c)
            cmd_check
            ;;
        --clean|clean)
            cmd_clean
            ;;
        --help|help|-h)
            cmd_help
            ;;
        *)
            log_error "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
