#!/usr/bin/env bash
# Полный прогон стенда: сборка -> тесты (+ASan/UBSan) -> фаззинг -> сверка на
# контрольных примерах -> бенчмарки -> профиль по строкам -> графики.
# Аналоги подключаются, если их исполняемые файлы есть в build/ (см. README, раздел «Аналоги»).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD; B=$ROOT/build; R=$ROOT/results
mkdir -p "$R"

echo "== 1. build"
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA:-} >/dev/null
cmake --build "$B" -j"$(nproc)" >/dev/null
cmake -S . -B "$B-asan" -DTLC_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug >/dev/null
cmake --build "$B-asan" -j"$(nproc)" --target tlc_tests tlc_fuzz >/dev/null

echo "== 2. unit tests (ASan+UBSan)"
"$B-asan/tlc_tests" "$ROOT/data" | tee "$R/unit_tests.txt"

echo "== 3. fuzzing (ASan+UBSan), ${FUZZ_ITERS:=50000} iterations"
"$B-asan/tlc_fuzz" "$FUZZ_ITERS" "${FUZZ_SEED:-2026}" | tee "$R/fuzz.txt"

echo "== 4. control examples (paper, §4-6)"
for f in data/*.txt; do "$B/tlc" solve "$f"; echo; done > "$R/control_examples.txt"
"$B/tlc" solve data/task1_saaty_hu_p124.txt --trace > "$R/task1_trace.txt"
if [[ -x "$B/analog2_cli" ]]; then
  for f in data/*.txt; do
    n=$(basename "$f" .txt)
    "$B/tlc" kernels "$f" > "$R/kernels_tlc_$n.txt"
    "$B/analog2_cli" kernels "$f" > "$R/kernels_analog2_$n.txt"
    python3 scripts/compare_kernels.py "$R/kernels_tlc_$n.txt" "$R/kernels_analog2_$n.txt" | sed "s/^/[$n] /"
  done | tee "$R/compare_analog2.txt"
fi
if [[ -x "$B/analog1_cli" ]]; then
  for f in data/*.txt; do "$B/analog1_cli" solve "$f"; echo; done > "$R/control_examples_analog1.txt"
fi

echo "== 5. benchmark: kernels (λ + Kleene star)"
"$B/bench_kernels" "${REPS:-5}" 100 ${BENCH_N:-10 20 40 60 80 100 120 150 160 200 300} > "$R/kernels.csv"
cat "$R/kernels.csv"

echo "== 6. benchmark: full pipeline on random problems (m = 5 criteria)"
mkdir -p "$R/gen"
echo "impl,problem,reps,total_ms,us_per_call" > "$R/pipeline.csv"
for n in 5 10 20 40 80; do
  "$B/tlc" gen "$n" 5 9 "$n" > "$R/gen/rand_n$n.txt"
  "$B/tlc" bench "$R/gen/rand_n$n.txt" 50 >> "$R/pipeline.csv"
  "$B/tlc" bench "$R/gen/rand_n$n.txt" 10 --naive >> "$R/pipeline.csv"
  [[ -x "$B/analog1_cli" ]] && "$B/analog1_cli" bench "$R/gen/rand_n$n.txt" 10 >> "$R/pipeline.csv"
done
cat "$R/pipeline.csv"

echo "== 7. line profile (callgrind)"
if command -v valgrind >/dev/null; then
  cmake -S . -B "$B-prof" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null && cmake --build "$B-prof" --target bench_kernels >/dev/null
  valgrind --tool=callgrind --callgrind-out-file="$R/callgrind.out" "$B-prof/bench_kernels" 1 100 60 >/dev/null 2>"$R/callgrind.log"
  callgrind_annotate --auto=yes "$R/callgrind.out" src/tropical.cpp > "$R/callgrind_annotated.txt" 2>/dev/null || true
  grep -m3 -E "tropical.cpp:tlc::" "$R/callgrind_annotated.txt" || true
fi

echo "== 8. plots"
python3 scripts/plot_results.py "$R/kernels.csv" || echo "(matplotlib not available)"
echo "done -> $R"
