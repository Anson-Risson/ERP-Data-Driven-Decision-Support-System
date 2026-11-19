"""Interactive Gradio application for the ERP decision-support project."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_data, encode_features, build_cluster_features
from src.model import run_kmeans

RAW, ARREARS_COLS, _ = load_data()
ENCODED, _ = encode_features(RAW)
FEATURES = build_cluster_features(ENCODED)
DATA, _, _ = run_kmeans(RAW, FEATURES)

# Make the app's semester columns consistent and numeric.
DATA["S1"] = pd.to_numeric(DATA["NO OF ARREARS S1"], errors="coerce").fillna(0)
DATA["S2"] = pd.to_numeric(DATA["NO OF ARREARS S2"], errors="coerce").fillna(0)
DATA["S3"] = pd.to_numeric(DATA["NO OF ARREARS S3"], errors="coerce").fillna(0)
DATA["S4"] = pd.to_numeric(DATA["NO OF ARREARS S4"], errors="coerce").fillna(0)
DATA["S5"] = pd.to_numeric(DATA["NO OF ARREARS S5"], errors="coerce").fillna(0)

SGPA_MAP = {
    "Sem 1": "SGPA1",
    "Sem 2": "SGPA2",
    "Sem 3": "SGPA3",
    "Sem 4": "SGPA4",
    "Sem 5": "SGPA 5",
}

def student_info(name):
    name = (name or "").strip().lower()
    row = DATA[DATA["Name"].astype(str).str.strip().str.lower() == name]
    if row.empty:
        return "Student not found.", "", "", None, None

    r = row.iloc[0]
    cluster = int(r["cluster"])
    label = str(r["cluster_label"])
    recommendation = str(r["recommendations"])

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(["S1","S2","S3","S4","S5"], [r["S1"],r["S2"],r["S3"],r["S4"],r["S5"]])
    ax1.set_title(f"No. of Arrears: {r['Name']}")
    ax1.set_ylabel("Arrears")
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    labels, values = [], []
    for label_name, col in SGPA_MAP.items():
        if col in DATA.columns:
            value = r[col]
            if pd.notna(value):
                labels.append(label_name)
                values.append(float(value))
    if values:
        ax2.plot(labels, values, marker="o")
        ax2.set_ylim(0, 10)
        ax2.set_ylabel("SGPA")
        ax2.set_title(f"Semester Performance: {r['Name']}")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "SGPA data unavailable", ha="center", va="center")
        ax2.set_axis_off()

    return str(cluster), label, recommendation, fig1, fig2

choices = DATA["Name"].tolist()

with gr.Blocks(title="ERP Student Performance Analysis") as demo:
    gr.Markdown("# ERP Student Performance & Decision Support")
    gr.Markdown("Analyze student performance clusters and view personalized recommendations.")
    student = gr.Dropdown(choices=choices, label="Select Student")
    button = gr.Button("Analyze")
    with gr.Row():
        cluster = gr.Textbox(label="Cluster")
        cluster_name = gr.Textbox(label="Performance Group")
    recommendation = gr.Textbox(label="Recommendation", lines=5)
    with gr.Row():
        arrears = gr.Plot(label="Arrears by Semester")
        sgpa = gr.Plot(label="SGPA Trend")
    button.click(student_info, student, [cluster, cluster_name, recommendation, arrears, sgpa])

if __name__ == "__main__":
    demo.launch()
