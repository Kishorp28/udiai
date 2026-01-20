# ==========================================================
# UIDAI DATA HACKATHON 2026 - FULL ANALYSIS CODE
# Author : Kishore Project
# ==========================================================

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

BASE = r"E:\UDAI\hack"

# ----------------------------------------------------------
# 1. FUNCTION TO LOAD MULTIPLE CSV FILES
# ----------------------------------------------------------

def load_folder(path):
    files = glob.glob(path + "/*.csv")
    df_list = []

    for f in files:
        print("Loading:", f)
        df_list.append(pd.read_csv(f))

    df = pd.concat(df_list, ignore_index=True)
    return df


# ----------------------------------------------------------
# 2. LOAD ALL DATASETS
# ----------------------------------------------------------

print("---- Loading Enrolment ----")
enrol = load_folder(BASE + "/api_data_aadhar_enrolment")

print("---- Loading Demographic ----")
demo = load_folder(BASE + "/api_data_aadhar_demographic")

print("---- Loading Biometric ----")
bio = load_folder(BASE + "/api_data_aadhar_biometric")


# ----------------------------------------------------------
# 3. CLEANING
# ----------------------------------------------------------

for d in [enrol, demo, bio]:
    d['date'] = pd.to_datetime(d['date'], errors='coerce', dayfirst=True)
    d.fillna(0, inplace=True)
    d.drop_duplicates(inplace=True)

print("Rows After Cleaning")
print("Enrol :", len(enrol))
print("Demo  :", len(demo))
print("Bio   :", len(bio))


# ----------------------------------------------------------
# 4. KPI 1 – STATE WISE ENROLMENT
# ----------------------------------------------------------

state_enrol = enrol.groupby("state")[
    ["age_0_5", "age_5_17", "age_18_greater"]
].sum()

state_enrol["total"] = state_enrol.sum(axis=1)

state_enrol = state_enrol.sort_values("total", ascending=False)

state_enrol.to_csv("state_enrolment_summary.csv")

print("\nTOP 10 STATES BY ENROLMENT")
print(state_enrol.head(10))


# ----------------------------------------------------------
# 5. KPI 2 – AGE DISTRIBUTION
# ----------------------------------------------------------

age_dist = enrol[["age_0_5", "age_5_17", "age_18_greater"]].sum()

age_dist.to_csv("age_distribution.csv")

age_dist.plot(kind='bar', title="Age Wise Enrolment")
plt.savefig("age_distribution.png")
plt.close()


# ----------------------------------------------------------
# 6. KPI 3 – MONTHLY TREND
# ----------------------------------------------------------

enrol['month'] = enrol['date'].dt.to_period("M")

trend = enrol.groupby("month")["age_18_greater"].sum()

trend.to_csv("monthly_trend.csv")

trend.plot(title="Monthly Adult Enrolment")
plt.savefig("monthly_trend.png")
plt.close()


# ----------------------------------------------------------
# 7. KPI 4 – DEMOGRAPHIC UPDATES
# ----------------------------------------------------------

demo_state = demo.groupby("state")[
    ["demo_age_5_17", "demo_age_17_"]
].sum()

demo_state.to_csv("demographic_state.csv")


# ----------------------------------------------------------
# 8. KPI 5 – BIOMETRIC UPDATES
# ----------------------------------------------------------

bio_state = bio.groupby("state")[
    ["bio_age_5_17", "bio_age_17_"]
].sum()

bio_state.to_csv("biometric_state.csv")


# ----------------------------------------------------------
# 9. SERVICE GAP INDEX
# ----------------------------------------------------------

merged = state_enrol.join(demo_state, how="left")

merged["update_ratio"] = merged["demo_age_17_"] / (merged["age_18_greater"] + 1)

merged.to_csv("service_gap_index.csv")

print("\nLOW UPDATE RATIO STATES")
print(merged.sort_values("update_ratio").head(10))


# ----------------------------------------------------------
# 10. DISTRICT LEVEL ANALYSIS
# ----------------------------------------------------------

district = enrol.groupby(["state", "district"])[
    ["age_0_5", "age_5_17", "age_18_greater"]
].sum()

district["total"] = district.sum(axis=1)

district = district.sort_values("total")

district.head(50).to_csv("low_performing_districts.csv")


# ----------------------------------------------------------
# 11. ANOMALY DETECTION
# ----------------------------------------------------------

daily = enrol.groupby("date")["age_18_greater"].sum()

mean = daily.mean()
std  = daily.std()

anomaly = daily[daily > mean + 2*std]

anomaly.to_csv("spike_days.csv")


# ----------------------------------------------------------
# 12. FINAL INSIGHTS EXPORT
# ----------------------------------------------------------

with open("project_findings.txt","w") as f:

    f.write("UIDAI DATA HACKATHON ANALYSIS\n\n")

    f.write("Top States by Enrolment\n")
    f.write(str(state_enrol.head(10)))

    f.write("\n\nLow Update Ratio States\n")
    f.write(str(merged.sort_values("update_ratio").head(10)))

    f.write("\n\nAnomaly Dates\n")
    f.write(str(anomaly.head(20)))


print("\n---- PROJECT COMPLETED ----")
print("All result files generated in current folder")
