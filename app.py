import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("ckd_rf_recall_model.pkl")

st.title("Chronic Kidney Disease (CKD) Risk Prediction")
st.write("Enter patient details")

patient_age = st.number_input("Age", min_value=1, max_value=120)

gender_input = st.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender_input == "Male" else 1

bp_systolic = st.number_input("BP Systolic")
bp_diastolic = st.number_input("BP Diastolic")
blood_urea = st.number_input("Blood Urea")
serum_creatinine = st.number_input("Serum Creatinine")
albumin = st.number_input("Albumin")
blood_glucose_random = st.number_input("Blood Glucose Random")
diabetes = st.number_input("Diabetes (0 = No, 1 = Yes)")
hypertension = st.number_input("Hypertension (0 = No, 1 = Yes)")

drug_dosage_mg = st.number_input("Drug Dosage (mg)")
exposure_days = st.number_input("Exposure Days")
nephrotoxic_label = st.number_input("Nephrotoxic Label")

mol_weight = st.number_input("Molecular Weight")
logP = st.number_input("logP")
hbond_donors = st.number_input("H-Bond Donors")
hbond_acceptors = st.number_input("H-Bond Acceptors")
rotatable_bonds = st.number_input("Rotatable Bonds")
tpsa = st.number_input("TPSA")
shape_index_3d = st.number_input("Shape Index 3D")
inertia_x = st.number_input("Inertia X")
inertia_y = st.number_input("Inertia Y")
inertia_z = st.number_input("Inertia Z")
charge_distribution = st.number_input("Charge Distribution")
clearance_rate = st.number_input("Clearance Rate")
half_life_hr = st.number_input("Half Life (hr)")
bioavailability_pct = st.number_input("Bioavailability (%)")
volume_of_distribution = st.number_input("Volume of Distribution")

kidney_cell_viability_pct = st.number_input("Kidney Cell Viability (%)")
mitochondrial_damage = st.number_input("Mitochondrial Damage")
oxidative_stress = st.number_input("Oxidative Stress")
protein_binding_pct = st.number_input("Protein Binding (%)")
serum_creatinine_change_pct = st.number_input("Creatinine Change (%)")
toxicity_score_composite = st.number_input("Toxicity Score Composite")
pk_toxic_interaction_score = st.number_input("PK Toxic Interaction Score")

drug_name = st.selectbox(
    "Drug Name",
    [
        "Ibuprofen", "Vancomycin", "Cisplatin", "Aspirin",
        "Gentamicin", "Paracetamol", "Tobramycin", "Amphotericin-B"
    ]
)
feature_names = model.feature_names_in_
input_df = pd.DataFrame(0, index=[0], columns=feature_names)

input_df['patient_age'] = patient_age
input_df['gender'] = gender
input_df['bp_systolic'] = bp_systolic
input_df['bp_diastolic'] = bp_diastolic
input_df['blood_urea'] = blood_urea
input_df['serum_creatinine'] = serum_creatinine
input_df['albumin'] = albumin
input_df['blood_glucose_random'] = blood_glucose_random
input_df['diabetes'] = diabetes
input_df['hypertension'] = hypertension
input_df['drug_dosage_mg'] = drug_dosage_mg
input_df['exposure_days'] = exposure_days
input_df['nephrotoxic_label'] = nephrotoxic_label
input_df['mol_weight'] = mol_weight
input_df['logP'] = logP
input_df['hbond_donors'] = hbond_donors
input_df['hbond_acceptors'] = hbond_acceptors
input_df['rotatable_bonds'] = rotatable_bonds
input_df['tpsa'] = tpsa
input_df['shape_index_3d'] = shape_index_3d
input_df['inertia_x'] = inertia_x
input_df['inertia_y'] = inertia_y
input_df['inertia_z'] = inertia_z
input_df['charge_distribution'] = charge_distribution
input_df['clearance_rate'] = clearance_rate
input_df['half_life_hr'] = half_life_hr
input_df['bioavailability_pct'] = bioavailability_pct
input_df['volume_of_distribution'] = volume_of_distribution
input_df['kidney_cell_viability_pct'] = kidney_cell_viability_pct
input_df['mitochondrial_damage'] = mitochondrial_damage
input_df['oxidative_stress'] = oxidative_stress
input_df['protein_binding_pct'] = protein_binding_pct
input_df['serum_creatinine_change_pct'] = serum_creatinine_change_pct
input_df['toxicity_score_composite'] = toxicity_score_composite
input_df['pk_toxic_interaction_score'] = pk_toxic_interaction_score

drug_column = f"drug_name_{drug_name}"
if drug_column in input_df.columns:
    input_df[drug_column] = 1

if st.button("Predict CKD Risk"):
    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("🟢 Low Risk of CKD")
    elif prediction == 1:
        st.warning("🟡 Moderate Risk of CKD")
    else:
        st.error("🔴 High Risk of CKD")

model.predict_proba(input_df)
