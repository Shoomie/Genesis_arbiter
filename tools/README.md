# Genesis Arbiter: Utility Scripts

This directory contains utility scripts for corpus analysis, tokenizer training, model evaluation, and automated data augmentation for the Arbiter research infrastructure.

---

---


---


## 🔍 Running Scripts

```bash
# Data preparation example
python tools/update_data_cache.py
```



---

## 📝 Script Summary

| Script | Purpose | Output | Phase |
|--------|---------|--------|-------|
| `update_data_cache.py` | VRAM Cache generation | `genesis_data_cache.pt` | Phase 2 |
| `generate_boundary_map.py` | WWM Boundary generation | `genesis_boundaries.pt` | Phase 3 |

---

## 🧪 Recommended Workflow (In progress)

1. **Update Data Cache**:
   ```bash
   python update_data_cache.py
   ```

2. **Train Model**:
   ```bash
   python run.py --option 1a
   ```

---

---

**Last Updated**: 2026-02-03  
