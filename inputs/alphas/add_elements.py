import os
import pandas as pd
import numpy as np

path = "./"  # Update this to your data directory path
output_filename = "JUNO_stopping_alpha.txt"

# Define your liquid scintillator cocktail (must sum to 1.0)
composition = {
    'carbon.txt': 0.87924,
    'hydrogen.txt': 0.1201,
    'oxygen.txt': 0.00034,
    'nitrogen.txt': 0.00027,
    'silicon.txt': 0.00005, #in lieu of Sulfur, missing in ASTAR database
}

# Verify weights sum close to 1
total_weight = sum(composition.values())
if not np.isclose(total_weight, 1.0):
    print(f"Warning: Composition weights sum to {total_weight}, not 1.0. Normalizing...")
    composition = {k: v / total_weight for k, v in composition.items()}

E_vals = None
total_mass_stopping = None

for filename, weight in composition.items():
    full_file_path = os.path.join(path, filename)
    print(f"Processing {filename} with weight {weight:.5f}...")
    
    # Read txt
    df_element = pd.read_csv(full_file_path, sep=r'\s+', header=None, engine='python')
    current_E = df_element[0].to_numpy()
    current_stopping = df_element[1].to_numpy()

    if E_vals is None:
        E_vals = current_E
        total_mass_stopping = np.zeros_like(E_vals, dtype=float)
    else:
        # Safety check: Ensure NIST (or YOU!) didn't change the energy grid between files
        if not np.allclose(E_vals, current_E):
            raise ValueError(f"Energy grid mismatch in {filename} compared to previous files!")
            
    total_mass_stopping += weight * current_stopping

# --- 3. Export in Exact NIST ASTAR Spacing Format ---
output_path = os.path.join(path, output_filename)
print(f"Writing combined spectrum to {output_path}...")

with open(output_path, 'w') as f:
    for e_val, stopping_val in zip(E_vals, total_mass_stopping):
        # Format matching NIST's standard scientific notation alignment:
        # Column 1: Kinetic Energy (MeV)
        # Column 2: Total Mass Stopping Power (MeV cm^2/g)
        # Uses 12-character width padding, left-aligned, 4 decimal places in scientific notation
        line = f"%-12.4E %-12.4E\n" % (e_val, stopping_val)
        f.write(line)

print("Addition finished!")
