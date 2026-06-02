#!/usr/bin/env bash
# =============================================================================
# GEPRiS – conda environment setup
# =============================================================================
# Usage:
#   bash setup_gepris_env.sh          # create the environment
#   bash setup_gepris_env.sh --remove  # remove it and start fresh
# =============================================================================

set -euo pipefail

ENV_NAME="gepris"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"

# ── Optional: remove existing env ────────────────────────────────────────────
if [[ "${1:-}" == "--remove" ]]; then
    echo ">>> Removing existing '$ENV_NAME' environment..."
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    echo ">>> Done."
    exit 0
fi

# ── 1. Create / update from YAML ─────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Environment '$ENV_NAME' already exists — updating..."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo ">>> Creating environment '$ENV_NAME'..."
    conda env create -f "$ENV_FILE"
fi

# ── 2. Quick smoke-test ───────────────────────────────────────────────────────
echo ""
echo ">>> Running import smoke-test..."
conda run -n "$ENV_NAME" python - <<'PYEOF'
import sys, importlib

packages = {
    "numpy":      "np.__version__",
    "scipy":      "scipy.__version__",
    "pandas":     "pd.__version__",
    "matplotlib": "matplotlib.__version__",
    "uproot":     "uproot.__version__",
    "iminuit":    "iminuit.__version__",
    "ROOT":       "ROOT.gROOT.GetVersion()",
}

ok = True
for pkg, ver_expr in packages.items():
    try:
        mod = importlib.import_module(pkg)
        # evaluate the version expression in the module's namespace
        ver = eval(ver_expr, {pkg.split(".")[0]: mod, **{pkg: mod}})
        print(f"  ✓  {pkg:<14} {ver}")
    except Exception as e:
        print(f"  ✗  {pkg:<14} FAILED: {e}")
        ok = False

if not ok:
    sys.exit(1)
PYEOF

echo ""
echo ">>> All imports OK."
echo ""
echo "=== How to activate ==================================================="
echo "  conda activate $ENV_NAME"
echo ""
echo "=== How to launch Jupyter ============================================="
echo "  conda activate $ENV_NAME"
echo "  jupyter lab                     # opens run_fit.ipynb in the browser"
echo ""
echo "=== How to write a TGraph to a .root file (PyROOT snippet) ==========="
cat <<'SNIPPET'

  import ROOT, numpy as np

  x = np.array([1.0, 2.0, 3.0, 4.0])
  y = np.array([1.1, 3.9, 9.1, 16.0])

  g = ROOT.TGraph(len(x), x, y)
  g.SetName("my_curve")
  g.SetTitle("My Curve;x;y")

  f = ROOT.TFile("output.root", "RECREATE")
  g.Write()
  f.Close()
  print("Saved output.root")

SNIPPET
echo "========================================================================"
