# Data conventions

Each fold has disjoint train.csv, val.csv and test.csv files. Identifiers must be unique within and across these sets. Repeated images/masks across sets are rejected. Keep all observations of one patient in the same split. The loader supports five fold indices.

Required columns: patient_id, image_path, label_path, stage, time, recur_label. HECKTOR additionally reads T-stage and N-stage (unknown categories are encoded explicitly). Paths must remain within data_root. Durations must be positive and finite; zero-time handling is controlled by zero_time_policy and exclusions are recorded by the loader. Use only documented clinical rules for such exclusions.

## Spatial processing

Image/mask affines define physical RAS millimetres. Image and mask shape/affine must match. Preprocessing constructs lesion-centered volumetric crops and closed, consistently oriented surfaces from the largest connected component, with preprocessing metadata recording component information. The dense reference mesh has at most 128 vertices by default; this is a numerical discretization, not a claim of mesh-resolution convergence.

NPC uses normalized unit intensities, HECKTOR uses HU clipping, and Foshan uses its MRI adapter. These modes are configured per cohort and must match the actual source images.

## Anatomical reference CSV

Provide one row per patient with:

    patient_id,origin_x,origin_y,origin_z,length_mm

Coordinates are physical RAS millimetres. length_mm is positive and lesion-independent. The origin must be obtained using a documented cranio-cervical reference procedure. Optional axis_r_x, axis_r_y, axis_r_z, axis_a_x, axis_a_y, axis_a_z, axis_s_x, axis_s_y, axis_s_z columns encode all nine entries of an orthonormal right-handed frame; omitting all axis columns uses physical RAS axes.

Geometric training requires a supplied reference table. A field-of-view center is not automatically a validated cranio-cervical origin. Document the fixed extraction procedure and its applicability to the cohort. Treatment-initiation duration provenance and reference provenance are recorded in the data_provenance configuration fields; these are operator attestations, not automatic verification of clinical records.

Clinical data and reference coordinates must be obtained through their approved access procedures. Generated examples contain no real patients.
