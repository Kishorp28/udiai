# ==========================================================
# UIDAI DATA HACKATHON 2026 – END TO END PIPELINE
# Loads Data → Analytics → Charts → PDF Report
# Author : Kishore Project
# ==========================================================

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# ----------------------------------------------------------
# PATH CONFIGURATION – CHANGE ONLY THIS
# ----------------------------------------------------------
BASE = r"E:\UDAI\hack"
OUTPUT = "uidai_output"

os.makedirs(OUTPUT, exist_ok=True)

# ----------------------------------------------------------
# 1. FUNCTION TO LOAD CSV FOLDER
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
# 2. LOAD DATASETS
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

print("After Cleaning")
print("Enrol :", len(enrol))
print("Demo  :", len(demo))
print("Bio   :", len(bio))


# ----------------------------------------------------------
# 4. KPI – STATE WISE ENROLMENT
# ----------------------------------------------------------
state_enrol = enrol.groupby("state")[
    ["age_0_5", "age_5_17", "age_18_greater"]
].sum()

state_enrol["total"] = state_enrol.sum(axis=1)
state_enrol = state_enrol.sort_values("total", ascending=False)

state_enrol.to_csv(OUTPUT + "/state_enrolment_summary.csv")


# ----------------------------------------------------------
# 5. AGE DISTRIBUTION CHART
# ----------------------------------------------------------
age_dist = enrol[["age_0_5", "age_5_17", "age_18_greater"]].sum()

plt.figure()
age_dist.plot(kind='bar', title="Age Wise Enrolment")
plt.savefig(OUTPUT + "/age_distribution.png")
plt.close()


# ----------------------------------------------------------
# 6. MONTHLY TREND
# ----------------------------------------------------------
enrol['month'] = enrol['date'].dt.to_period("M")
trend = enrol.groupby("month")["age_18_greater"].sum()

plt.figure()
trend.plot(title="Monthly Adult Enrolment")
plt.savefig(OUTPUT + "/monthly_trend.png")
plt.close()


# ----------------------------------------------------------
# 7. DEMOGRAPHIC & BIOMETRIC SUMMARY
# ----------------------------------------------------------
demo_state = demo.groupby("state")[["demo_age_5_17", "demo_age_17_"]].sum()
bio_state  = bio.groupby("state")[["bio_age_5_17", "bio_age_17_"]].sum()

demo_state.to_csv(OUTPUT + "/demographic_state.csv")
bio_state.to_csv(OUTPUT + "/biometric_state.csv")


# ----------------------------------------------------------
# 8. SERVICE GAP INDEX
# ----------------------------------------------------------
merged = state_enrol.join(demo_state, how="left")
merged["update_ratio"] = merged["demo_age_17_"] / (merged["age_18_greater"] + 1)

merged.to_csv(OUTPUT + "/service_gap_index.csv")


# ----------------------------------------------------------
# 9. ANOMALY DETECTION
# ----------------------------------------------------------
daily = enrol.groupby("date")["age_18_greater"].sum()

mean = daily.mean()
std  = daily.std()

anomaly = daily[daily > mean + 2*std]
anomaly.to_csv(OUTPUT + "/spike_days.csv")


# ==========================================================
# 10. AUTOMATIC PDF REPORT GENERATION
# ==========================================================

styles = getSampleStyleSheet()
pdf = SimpleDocTemplate(OUTPUT + "/UIDAI_Analytics_Report.pdf")
story = []

story.append(Paragraph("UIDAI DATA HACKATHON – ANALYTICS REPORT", styles['Title']))
story.append(Spacer(1, 0.2*inch))

# ---- Overview ----
story.append(Paragraph("1. Dataset Overview", styles['Heading2']))
story.append(Paragraph(
    f"Total Enrol Records: {len(enrol)}<br/>"
    f"Total Demographic Updates: {len(demo)}<br/>"
    f"Total Biometric Updates: {len(bio)}",
    styles['Normal']
))

story.append(Spacer(1, 0.2*inch))

# ---- Top States Table ----
story.append(Paragraph("2. Top States by Enrolment", styles['Heading2']))

tbl_data = [["State","Total"]]
for i,r in state_enrol.head(8).iterrows():
    tbl_data.append([i, int(r["total"])])

t = Table(tbl_data)
t.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.lightblue),
    ('GRID',(0,0),(-1,-1),1,colors.black)
]))
story.append(t)

story.append(Spacer(1, 0.2*inch))

# ---- Charts ----
story.append(Paragraph("3. Age Distribution", styles['Heading2']))
story.append(Image(OUTPUT + "/age_distribution.png", width=5*inch, height=3*inch))

story.append(Paragraph("4. Monthly Trend", styles['Heading2']))
story.append(Image(OUTPUT + "/monthly_trend.png", width=5*inch, height=3*inch))

# ---- Insights ----
story.append(Paragraph("5. Key Insights", styles['Heading2']))

insight = f"""
• Highest enrolment state: {state_enrol.index[0]} <br/>
• Average adult enrol per day: {int(mean)} <br/>
• Spike days detected: {len(anomaly)} <br/>
• States with low update ratio need camps
"""

story.append(Paragraph(insight, styles['Normal']))

# ---- Hackathon Sections ----
story.append(Paragraph("6. Novelty", styles['Heading2']))
story.append(Paragraph(
"Service Gap Index, Anomaly Detection and Age Lifecycle analytics using real UIDAI structure.",
styles['Normal']))

story.append(Paragraph("7. Feasibility", styles['Heading2']))
story.append(Paragraph(
"Built using open-source Python stack deployable at state UIDAI dashboards.",
styles['Normal']))

story.append(Paragraph("8. Impact", styles['Heading2']))
story.append(Paragraph(
"Helps identify underserved PIN codes and plan Aadhaar Seva Kendras.",
styles['Normal']))

story.append(Paragraph("9. Alignment with Aadhaar Values", styles['Heading2']))
story.append(Paragraph(
"Privacy preserving, aggregated analysis, no individual identification.",
styles['Normal']))

pdf.build(story)

print("\n---- PROJECT COMPLETED ----")
print("Files generated in:", OUTPUT)
