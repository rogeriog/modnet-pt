# STATUS — modnet-pt

- **Phase:** 0 — design draft (no implementation)
- **Last action:** 2026-11-05 — Rogério green-lit starting the project;
  `DESIGN.md` drafted (API surface, module map, parity/testing plan,
  milestones M0–M8); package name `modnet.pt` locked (2026-10-31)
- **Last action (cont.):** decisions locked (modnet-pt / modnet_pt, MIT,
  core featurization, torch ≥2.0 CPU-first); M0 scaffold built — package
  layout per DESIGN.md §5, stub modules with upstream file:line citations,
  sklearn-glue tests (params/clone/NotImplementedError contract, defaults
  mirror vanilla.py:48), GitHub Actions CI (py3.10–3.12, CPU torch)
- **Next step:** Rogério creates `github.com/rogeriog/modnet-pt`, then run
  `bin/gh-setup.sh modnet-pt` → push M0 → CI green → M1 (MatminerFeaturizer)
- **Waiting on Rogério:** GitHub repo creation
