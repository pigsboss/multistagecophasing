#!/usr/bin/env bash
# MCPC 回归执行脚本（证据归档规范配套，RE-001 I-7）
# 用法: ./run_regression.sh <L1|L2|L3|L4|ALL> [备注]
# 说明:
#   - L1..L4 运行对应级别的测试文件（P1 起逐步落地，见 DD-003 建议文件清单）；
#   - ALL 运行全量 tests/；
#   - 输出落盘到 workbench/design_dev/evidence/M{0..4}/（ALL -> evidence/ALL/），
#     命名规范: YYYYMMDD_HHMMSS_<级别>_<备注>.log
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LEVEL="${1:-ALL}"
NOTE="${2:-manual}"
TS="$(date +%Y%m%d_%H%M%S)"

case "$LEVEL" in
  L1) TESTS="tests/test_central_gravity.py tests/test_l1_physical_baseline.py"; DIR="M1" ;;
  L2) TESTS="tests/test_l2_formation_wiring.py"; DIR="M2" ;;
  L3) TESTS="tests/test_l3_payload_phasing.py"; DIR="M3" ;;
  L4) TESTS="tests/test_l4_engineering.py"; DIR="M4" ;;
  LS) TESTS="tests/lunar_swing/"; DIR="LS" ;;
  AS) TESTS="tests/atmospheric_swing/"; DIR="AS" ;;
  ALL) TESTS="tests/"; DIR="ALL" ;;
  *) echo "级别必须是 L1|L2|L3|L4|LS|AS|ALL" >&2; exit 2 ;;
esac

OUT_DIR="$ROOT/workbench/design_dev/evidence/$DIR"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/${TS}_${LEVEL}_${NOTE}.log"

cd "$ROOT"
echo "# 回归执行: level=$LEVEL note=$NOTE" | tee "$LOG"
echo "# 命令: python3 -m pytest $TESTS -q" | tee -a "$LOG"
echo "# 开始: $(date '+%F %T')" | tee -a "$LOG"

python3 -m pytest $TESTS -q 2>&1 | tee -a "$LOG"
EXIT=${PIPESTATUS[0]}

echo "# 结束: $(date '+%F %T')  exit=$EXIT" | tee -a "$LOG"
echo "证据已归档: $LOG"
exit "$EXIT"
