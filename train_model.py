"""
train_model.py
Run this ONCE to train and save the model:
    python train_model.py
"""
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("thermal_conductivity_dataset.csv")
print(f"Loaded: {df.shape}")

# ── Pass 1: Drop metadata & raw arrays ───────────────────────────────────────
drop_cols = [
    'aflow_version','aflowlib_date','aflowlib_version','aurl',
    'calculation_cores','calculation_memory','calculation_time',
    'catalog','code','data_api','data_source','files','loop',
    'node_CPU_Cores','node_CPU_MHz','node_CPU_Model','node_RAM_GB',
    'title','kpoints','kpoints_bands_nkpts','kpoints_bands_path',
    'kpoints_relax','kpoints_static','energy_cutoff',
    'delta_electronic_energy_convergence','delta_electronic_energy_threshold',
    'Bravais_lattice_lattice_system_orig','Bravais_lattice_lattice_type_orig',
    'Bravais_lattice_lattice_variation_type_orig','Bravais_lattice_orig',
    'Bravais_superlattice_lattice_system_orig','Bravais_superlattice_lattice_type_orig',
    'Bravais_superlattice_lattice_variation_type_orig',
    'Pearson_symbol_orig','Pearson_symbol_superlattice_orig',
    'Wyckoff_letters_orig','Wyckoff_multiplicities_orig','Wyckoff_site_symmetries_orig',
    'crystal_class_orig','crystal_family_orig','crystal_system_orig',
    'density_orig','geometry_orig','lattice_system_orig','lattice_variation_orig',
    'point_group_Hermann_Mauguin_orig','point_group_Schoenflies_orig',
    'point_group_orbifold_orig','point_group_order_orig',
    'point_group_structure_orig','point_group_type_orig',
    'reciprocal_geometry_orig','reciprocal_lattice_type_orig',
    'reciprocal_lattice_variation_type_orig','reciprocal_volume_cell_orig',
    'spacegroup_orig_x','volume_atom_orig','volume_cell_orig',
    'anrl_label_orig','anrl_parameter_list_orig','anrl_parameter_values_orig',
    'forces','positions_cartesian','positions_fractional',
    'stress_tensor','bader_atomic_volumes','bader_net_charges',
    'nbondxx','geometry','anrl_parameter_list_relax','anrl_parameter_values_relax',
    'dft_type','ldau_TLUJ','ldau_j','ldau_l','ldau_type','ldau_u',
    'species_pp','species_pp_ZVAL','species_pp_version',
    'pressure','pressure_residual','ael_applied_pressure',
    'ael_average_external_pressure','Pulay_stress',
]
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ── Pass 2: Drop redundant features ──────────────────────────────────────────
drop_final = [
    'Bravais_lattice_relax','Bravais_superlattice_lattice_system',
    'Bravais_superlattice_lattice_type','Bravais_superlattice_lattice_variation_type',
    'lattice_system_relax','lattice_variation_relax',
    'crystal_class','crystal_family',
    'reciprocal_lattice_type','reciprocal_lattice_variation_type',
    'sg2','natoms_orig',
    'energy_cell','enthalpy_cell','eentropy_cell','PV_cell',
    'agl_vibrational_entropy_300K_cell','agl_vibrational_free_energy_300K_cell',
    'agl_bulk_modulus_static_300K','volume_cell_x',
    'ael_bulk_modulus_reuss','ael_bulk_modulus_voigt',
    'ael_shear_modulus_reuss','ael_shear_modulus_voigt',
    'spin_cell','agl_poisson_ratio_source',
    'compound','composition','Wyckoff_letters','Wyckoff_site_symmetries',
    'Pearson_symbol_superlattice','anrl_label_relax','prototype',
    'reciprocal_geometry','stoichiometry','species',
]
df_final = df_clean.drop(columns=[c for c in drop_final if c in df_clean.columns])

# ── Drop leaky features ───────────────────────────────────────────────────────
all_leaky = [
    'agl_acoustic_debye','agl_bulk_modulus_isothermal_300K','agl_bulk_modulus_static_300K',
    'agl_debye','agl_gruneisen','agl_heat_capacity_Cp_300K','agl_heat_capacity_Cv_300K',
    'agl_poisson_ratio_source','agl_thermal_expansion_300K',
    'agl_vibrational_entropy_300K_atom','agl_vibrational_entropy_300K_cell',
    'agl_vibrational_free_energy_300K_atom','agl_vibrational_free_energy_300K_cell',
    'ael_debye_temperature','ael_speed_sound_average',
    'ael_speed_sound_longitudinal','ael_speed_sound_transverse',
    'ael_bulk_modulus_vrh','ael_bulk_modulus_reuss','ael_bulk_modulus_voigt',
    'ael_shear_modulus_vrh','ael_shear_modulus_reuss','ael_shear_modulus_voigt',
    'ael_youngs_modulus_vrh','ael_poisson_ratio','ael_pughs_modulus_ratio','ael_elastic_anisotropy',
    'scintillation_attenuation_length',
    'sg','spacegroup_relax','Pearson_symbol_relax',
]
df_model = df_final.drop(columns=[c for c in all_leaky if c in df_final.columns])

# ── Encode ────────────────────────────────────────────────────────────────────
drop_str = [
    'crystal_system','point_group_Hermann_Mauguin',
    'point_group_Schoenflies','point_group_orbifold',
    'spinD','Wyckoff_multiplicities',
]
df_enc = df_model.drop(columns=[c for c in drop_str if c in df_model.columns]).copy()

onehot_cols = [
    'Bravais_lattice_lattice_system','Bravais_lattice_lattice_type',
    'Bravais_lattice_lattice_variation_type',
    'Egap_type','point_group_type','point_group_structure',
]
df_enc = pd.get_dummies(
    df_enc, columns=[c for c in onehot_cols if c in df_enc.columns], drop_first=False
)

df_enc['log_kl'] = np.log1p(df_enc['thermal_conductivity_target'])
df_enc = df_enc.drop(columns=['thermal_conductivity_target'])

X = df_enc.drop(columns=['log_kl'])
y = df_enc['log_kl']

for col in X.select_dtypes(include='bool').columns:
    X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.111, random_state=42)

# ── Train full LightGBM ───────────────────────────────────────────────────────
params = {
    'objective':'regression','metric':['mae','rmse'],
    'learning_rate':0.05,'num_leaves':63,'max_depth':-1,
    'min_child_samples':20,'feature_fraction':0.8,
    'bagging_fraction':0.8,'bagging_freq':5,
    'lambda_l1':0.1,'lambda_l2':0.1,'verbose':-1,'random_state':42,
}
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=100),
]
lgbm = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    valid_names=['val'],
    callbacks=callbacks,
)

# ── SHAP top-10 ───────────────────────────────────────────────────────────────
import shap
explainer   = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_test)
shap_importance = (
    pd.DataFrame({'feature': X_test.columns,
                  'mean_shap': np.abs(shap_values).mean(axis=0)})
    .sort_values('mean_shap', ascending=False)
)
top10 = shap_importance.head(10)['feature'].tolist()
print("Top 10 features:", top10)

# ── Lean model ────────────────────────────────────────────────────────────────
lgbm_top10 = lgb.train(
    params,
    lgb.Dataset(X_train[top10], label=y_train),
    num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val[top10], label=y_val)],
    valid_names=['val'],
    callbacks=callbacks,
)

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(lgbm_top10, 'kl_predictor_lgbm.pkl')
joblib.dump(top10,       'kl_top10_features.pkl')

# Compute dataset stats for each feature (for UI hints)
stats = {}
for f in top10:
    col = df[f] if f in df.columns else None
    if col is not None:
        stats[f] = {
            'min': float(col.min()), 'max': float(col.max()),
            'mean': float(col.mean()), 'std': float(col.std())
        }
joblib.dump(stats, 'feature_stats.pkl')

print("✅ Model saved → kl_predictor_lgbm.pkl")
print("✅ Features saved → kl_top10_features.pkl")
print("✅ Stats saved → feature_stats.pkl")

# Quick test
sample = X_test[top10].iloc[[0]]
pred   = np.expm1(lgbm_top10.predict(sample))[0]
true   = np.expm1(y_test.iloc[0])
print(f"\nSample prediction: {pred:.4f} W/mK  |  True: {true:.4f} W/mK")
