# Split Reduction Optimization Toolkit

**Data-driven optimization of IREE's split reduction constants for GPU performance.**

Automatically tests, analyzes, and validates optimal `limitParallelLoops` configurations for your workload, then generates production-ready C++ code.

---

## 📖 Documentation

📚 **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- Prerequisites and setup verification
- Run optimization (quick or full sweep)
- Understand the comprehensive analysis
- Apply recommendations

🔬 **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Technical deep-dive
- Architecture and algorithms
- Analysis methodology (6-part report)
- Clustering and threshold derivation
- Extension guide

---

## ⚡ Quick Start

```bash
# 1. Verify setup
cd /home/vivizhan/split_reduction_optimization
./check_setup.sh

# 2. Run optimization sweep
./run_parameter_search.sh quick ~/test.txt  # ~5 min
# OR
./run_parameter_search.sh full ~/test.txt   # ~1 hour

# 3. View comprehensive analysis (includes validation!)
cat ../<test_name>_results/comprehensive_analysis.txt
# Example: ../prod_weight_shapes_results/comprehensive_analysis.txt
```

The analysis includes:
- ✅ Performance summary for all tested limits
- ✅ **C++ code recommendations** (ready to copy/paste)
- ✅ **Validated performance** (20x speedup confirmation)
- ✅ Production readiness assessment

---

## 🎯 What This Does

1. **Tests** 11 configurations (baseline + limits 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
2. **Analyzes** which configuration works best for each operation
3. **Derives** optimal C++ threshold constants using data-driven clustering
4. **Validates** recommendations by testing the optimized configuration
5. **Reports** everything in one comprehensive file

**All automated. All in one command.**

---

## 📁 Project Structure

```
split_reduction_optimization/
├── README.md                      ← You are here
├── QUICKSTART.md                  ← User guide
├── IMPLEMENTATION.md              ← Technical details
│
├── run_parameter_search.sh        ← Main script (run this!)
├── optimize_single_limit.py       ← Tests single limit in isolation
├── analyze_results.py             ← Comprehensive analysis + validation
├── create_json_summary.py         ← Aggregates CSV results
│
├── check_setup.sh                 ← Verify environment
├── create_small_test.sh           ← Create test subset
└── split_test_files.sh            ← Split by operation type
```

**Results:** Saved to `../<test_name>_results/` (e.g., `../prod_weight_shapes_results/`)

---

## 🛠️ Requirements

- **IREE** compiler source and build directory
- **iree-turbine** with virtual environment
- **PyTorch** installed in turbine venv
- **GPU** (tested on MI300)
- **Python 3.8+**

---

## 📝 Citation

If you use this toolkit, please reference:
- IREE Project: https://github.com/iree-org/iree

---

## 🤝 Contributing

This toolkit can be extended to:
- Test other compiler passes
- Support additional GPU architectures
- Optimize for different operation types
- Integrate with CI/CD pipelines

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for extension guidelines.

---

## 📄 License

Part of the IREE project. See IREE's license for details.
