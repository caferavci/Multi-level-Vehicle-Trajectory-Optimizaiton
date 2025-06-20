# Multi-level-Vehicle-Trajectory-Optimizaiton
S3 + RL based trajectory optimization(macro&amp;micro) including lane changing behavior


# dp-new

This repo contains a fully reproducible implementation of a 4-layer Backward Dynamic Programming (Backward DP) solver for lane-changing trajectory optimization, as described in my research.

- **Core features:**  
  - Implements a four-dimensional DP (time, speed, headway, control) using backward recursion and trajectory backtracking.
  - Compatible with NGSIM-style datasets and MATLAB `.mat` files.
  - The code strictly follows the Algorithm 1 structure in the paper, using real vehicle data for each time step.
  - The cost function combines acceleration, speed error, headway error, and control effort as quadratic terms.
- **Main script:** `dp-new.py` (all logic is in this notebook/script)

**Note:**  
The cost function in this code matches the main form described in the paper. If you notice any mismatch or need to add specific penalty/energy terms from the paper, please double-check the weights and formula in the cost computation section.

## Usage

1. Place your scenario file (e.g. `lc_data_20s_withpoints.mat`) in the working directory.
2. Run the notebook or script. All outputs and figures will be saved automatically.
3. See the comments in the code for details on each step.

---

If you have any questions or notice any inconsistency in the cost function, feel free to open an issue or contact me.
