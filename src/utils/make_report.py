"""Generate the final project report as reports/final_report.pdf.

Uses fpdf2 (pure Python) so it runs in a headless environment. The report
embeds the architecture diagram, the evaluation results, the feature catalog,
the design discussion, and the per-stage execution screenshots. A roll-number
watermark is printed on every page.
"""
from __future__ import annotations

import csv
from pathlib import Path

from fpdf import FPDF

from src.utils.common import resolve

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SHOTS = REPORTS / "screenshots"

STUDENT = "Akshay Agrawal"
ROLL = "2025AG05201"
REPO_URL = "https://github.com/AKbhai18/recommendation-pipeline"

NAVY = (15, 23, 42)
BLUE = (37, 99, 235)
GREY = (90, 100, 115)
LIGHT = (241, 245, 249)


class Report(FPDF):
    def header(self):
        # roll-number watermark (top-right, every page)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(170, 175, 185)
        self.set_xy(-60, 6)
        self.cell(54, 5, f"Roll No: {ROLL}", align="R")

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 155, 165)
        self.cell(0, 4, f"{STUDENT}  |  {ROLL}  |  Page {self.page_no()}",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(37, 99, 235)
        self.cell(0, 4, f"Repo: {REPO_URL}", align="C", link=REPO_URL)

    # ---- typography helpers ------------------------------------------------
    def h1(self, text):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*NAVY)
        self.ln(1)
        self.multi_cell(0, 8, text)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        y = self.get_y() + 1
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(3)

    def h2(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*BLUE)
        self.ln(1)
        self.multi_cell(0, 6.5, text)
        self.ln(0.5)

    def body(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(40, 45, 55)
        self.multi_cell(0, 5.4, text)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(40, 45, 55)
        x = self.get_x()
        self.cell(5, 5.4, chr(149))
        self.set_x(x + 5)
        self.multi_cell(0, 5.4, text)
        self.ln(0.3)

    def image_block(self, path, caption, max_w=180):
        path = Path(path)
        if not path.exists():
            return
        if self.get_y() > self.h - 70:
            self.add_page()
        self.image(str(path), w=max_w, x=(self.w - max_w) / 2)
        self.ln(1)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*GREY)
        self.multi_cell(0, 4.5, caption, align="C")
        self.ln(3)


def _read_results():
    rows = []
    with open(REPORTS / "evaluation_results.csv") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def _results_table(pdf: Report):
    rows = _read_results()
    cols = ["model", "precision@10", "recall@10", "ndcg@10",
            "coverage", "diversity", "rmse", "mae"]
    headers = ["Model", "P@10", "R@10", "NDCG", "Cov.", "Div.", "RMSE", "MAE"]
    widths = [46, 18, 18, 18, 16, 16, 18, 18]

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(255, 255, 255)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=0, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(35, 40, 50)
    for i, r in enumerate(rows):
        fill = i % 2 == 1
        pdf.set_fill_color(*LIGHT)
        for c, w in zip(cols, widths):
            val = r[c]
            if c == "model":
                val = val.replace("_", " ")
                pdf.set_font("Helvetica", "", 8.2)
                pdf.cell(w, 6.5, " " + val, border=0, align="L", fill=fill)
                pdf.set_font("Helvetica", "", 9)
            else:
                try:
                    val = f"{float(val):.3f}"
                except ValueError:
                    pass
                pdf.cell(w, 6.5, val, border=0, align="C", fill=fill)
        pdf.ln()
    pdf.ln(3)


def _feature_catalog_table(pdf: Report):
    from src.features.build_features import FEATURE_CATALOG
    widths = [26, 52, 102]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(255, 255, 255)
    for h, w in zip(["Entity", "Feature", "Description"], widths):
        pdf.cell(w, 7, " " + h, border=0, align="L", fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8.2)
    pdf.set_text_color(35, 40, 50)
    for i, (entity, name, dtype, desc) in enumerate(FEATURE_CATALOG):
        fill = i % 2 == 1
        pdf.set_fill_color(*LIGHT)
        if pdf.get_y() > pdf.h - 24:
            pdf.add_page()
        y0 = pdf.get_y()
        x0 = pdf.get_x()
        # measure wrapped height of the description
        lines = pdf.multi_cell(widths[2], 4.6, desc, border=0, align="L",
                               dry_run=True, output="LINES")
        n_lines = max(1, len(lines))
        row_h = max(6.0, 4.6 * n_lines)
        pdf.set_xy(x0, y0)
        pdf.cell(widths[0], row_h, " " + entity, border=0, fill=fill)
        pdf.cell(widths[1], row_h, " " + name, border=0, fill=fill)
        x_desc = pdf.get_x()
        pdf.multi_cell(widths[2], row_h / n_lines, desc, border=0, fill=fill)
        pdf.set_xy(x0, y0 + row_h)
    pdf.ln(3)


def build():
    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(15, 14, 15)

    # ---------------- Title ----------------
    pdf.add_page()
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 9, "End-to-End Data Management Pipeline\nfor a Recommendation System",
                   align="C")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 6, "Data Management for Machine Learning - Assignment 2",
                   align="C")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*BLUE)
    pdf.multi_cell(0, 6, f"{STUDENT}   |   Roll No: {ROLL}", align="C")
    pdf.ln(5)

    # ---- Prominent GitHub repository link box (first visible deliverable) ----
    box_w = 180
    box_x = (pdf.w - box_w) / 2
    box_y = pdf.get_y()
    box_h = 20
    pdf.set_fill_color(37, 99, 235)
    pdf.rect(box_x, box_y, box_w, box_h, style="F")
    pdf.set_xy(box_x, box_y + 3.5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(box_w, 6, "GitHub Repository", align="C")
    pdf.set_xy(box_x, box_y + 10)
    pdf.set_font("Helvetica", "BU", 12)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(box_w, 6, REPO_URL, align="C", link=REPO_URL)
    pdf.set_xy(pdf.l_margin, box_y + box_h)
    pdf.ln(6)

    # ---------------- 1. Overview & design ----------------
    pdf.h1("1. Overview and Design Decisions")
    pdf.body(
        "This project implements a modular, reproducible data-management "
        "pipeline for a MovieLens-style movie recommendation system. It covers "
        "the full lifecycle: data ingestion, validation, preparation, "
        "transformation, feature management, model training, evaluation, and "
        "orchestration. The emphasis is a clean, modular, reproducible pipeline "
        "rather than only maximising accuracy."
    )
    pdf.h2("Key design decisions")
    pdf.bullet("Config-driven and seeded: all parameters live in a single "
               "configs/config.yaml and one global seed makes every run "
               "deterministic and reproducible.")
    pdf.bullet("Validation precedes preparation: ingestion is intentionally "
               "lenient so the validation stage observes and reports the data "
               "exactly as it arrived, quarantining bad rows with a reason.")
    pdf.bullet("Time-based train/test split: the most recent portion of each "
               "user's history is held out, mimicking the real task of "
               "predicting future interactions and avoiding data leakage.")
    pdf.bullet("A reusable feature store: user, item and interaction features "
               "are computed from the training split only and persisted, so the "
               "identical definitions serve both training and prediction.")
    pdf.bullet("Two orchestration options: a dependency-free Python DAG runner "
               "for easy reproduction, plus an equivalent Apache Airflow DAG.")

    pdf.h2("Tools used")
    pdf.body(
        "Python, pandas and NumPy for data handling; Pandera for schema and "
        "data-quality validation; scikit-learn (TruncatedSVD, cosine "
        "similarity) and SciPy for modelling; MLflow for experiment tracking; "
        "Parquet/CSV for storage; and Apache Airflow (plus a structured Python "
        "workflow) for orchestration. Matplotlib is used for the diagram and "
        "figures."
    )

    pdf.image_block(REPORTS / "architecture_diagram.png",
                    "Figure 1. Pipeline architecture - eight stages, the "
                    "storage layers they read/write, MLflow tracking, and the "
                    "orchestration layer.")

    # ---------------- 2. Data & validation ----------------
    pdf.add_page()
    pdf.h1("2. Data, Ingestion and Validation")
    pdf.body(
        "The pipeline generates a documented synthetic dataset modelled on "
        "MovieLens (users, items, and explicit 1-5 interaction ratings). "
        "Interactions are drawn from a latent-factor taste model so there is "
        "genuine collaborative structure for the personalised models to "
        "recover. A small, configurable fraction of rows is deliberately "
        "corrupted (missing/out-of-range ratings, invalid timestamps, dangling "
        "foreign keys, duplicates) to exercise the validation stage. Real "
        "datasets can be used by placing matching CSVs in data/raw/."
    )
    pdf.body(
        "The validation stage combines Pandera schema checks on the dimension "
        "tables with custom row-level checks on the interactions. Every invalid "
        "row is written to data/errors/invalid_interactions.csv with a per-row "
        "reason, and a summary report is produced. In the run shown below, "
        "8,716 of 61,800 interaction rows (14.1%) were quarantined across all "
        "the injected problem types."
    )
    pdf.image_block(SHOTS / "01_ingestion.png",
                    "Figure 2. Stage 1 - data ingestion of users, items and "
                    "interactions.", max_w=180)
    pdf.image_block(SHOTS / "02_validation.png",
                    "Figure 3. Stage 2 - data validation: schema checks and "
                    "row-level quarantining of invalid interactions.", max_w=180)

    # ---------------- 3. Preparation & features ----------------
    pdf.add_page()
    pdf.h1("3. Preparation, Transformation and Features")
    pdf.body(
        "Preparation standardises formats, removes cold users with too few "
        "interactions, performs the per-user time-based split, and builds the "
        "derived datasets: a user activity summary, item popularity, and the "
        "user-item rating matrix (built from the training split only). The "
        "feature layer then computes reusable user, item and interaction "
        "features and persists them as a Parquet feature store."
    )
    pdf.image_block(SHOTS / "03_preparation.png",
                    "Figure 4. Stages 3 & 4 - preparation and transformation: "
                    "cleaning, time-based split, and derived datasets.", max_w=180)
    pdf.image_block(SHOTS / "04_features.png",
                    "Figure 5. Stage 5 - feature management: the persisted "
                    "feature store.", max_w=150)

    pdf.h2("Feature catalog")
    _feature_catalog_table(pdf)

    # ---------------- 4. Training & evaluation ----------------
    pdf.add_page()
    pdf.h1("4. Model Training and Evaluation")
    pdf.body(
        "Three models are trained and compared: a non-personalised popularity "
        "baseline, item-based collaborative filtering (cosine item-item "
        "similarity), and a matrix-factorisation model (truncated SVD on the "
        "mean-centred user-item matrix). Every run's parameters, metrics and "
        "serialised model are logged to MLflow. Models are evaluated on the "
        "held-out time-based test split using Precision@K, Recall@K, NDCG@K, "
        "Coverage and Diversity (K=10), plus RMSE and MAE for rating "
        "prediction."
    )
    _results_table(pdf)
    pdf.body(
        "Both personalised models clearly outperform the popularity baseline on "
        "ranking quality. Item-based collaborative filtering achieves the best "
        "top-K ranking (Precision@10 of 0.057 versus 0.022 for the baseline) "
        "and by far the widest catalog coverage (0.94 versus 0.04), because the "
        "baseline recommends the same popular items to everyone. Matrix "
        "factorisation gives the best explicit-rating prediction (lowest RMSE)."
    )
    pdf.image_block(SHOTS / "07_model_comparison.png",
                    "Figure 6. Model comparison across ranking metrics and "
                    "rating error.")
    pdf.image_block(SHOTS / "05_training_evaluation.png",
                    "Figure 7. Stages 6 & 7 - training and evaluation console "
                    "output with the comparison table.", max_w=180)
    pdf.image_block(SHOTS / "08_mlflow_runs.png",
                    "Figure 8. MLflow experiment tracking - the three logged "
                    "runs with their parameters and metrics.", max_w=180)

    # ---------------- 5. Orchestration ----------------
    pdf.add_page()
    pdf.h1("5. Orchestration and Reproducibility")
    pdf.body(
        "The complete workflow is automated as a directed acyclic graph with "
        "correct task dependencies: generate_data -> ingest -> validate -> "
        "prepare -> build_features -> train_and_evaluate. The structured Python "
        "orchestrator performs a topological sort and runs each stage in "
        "dependency order with timing, while an equivalent Apache Airflow DAG "
        "is provided for scheduled production use. Because every parameter is "
        "config-driven and seeded, the end-to-end run is fully reproducible."
    )
    pdf.image_block(SHOTS / "06_orchestration.png",
                    "Figure 9. Stage 8 - a successful end-to-end orchestrated "
                    "run, showing each stage completing in dependency order.",
                    max_w=180)

    # ---------------- 6. Limitations ----------------
    pdf.h1("6. Limitations and Future Work")
    pdf.bullet("The dataset is synthetic. Although modelled on MovieLens with a "
               "realistic latent-taste structure, absolute metric values differ "
               "from a real dataset; the pipeline accepts real CSVs unchanged.")
    pdf.bullet("The collaborative-filtering and matrix-factorisation models are "
               "classical. Stronger models (e.g. implicit-feedback ALS, LightFM, "
               "or neural recommenders) would likely improve ranking quality.")
    pdf.bullet("Absolute Precision@K values are modest because the catalog is "
               "large (500 items) and each user's test set is small, so the "
               "relative comparison between models is the key result.")
    pdf.bullet("The feature store is file-based. A production system would use a "
               "dedicated store (e.g. Feast) with online serving and point-in-"
               "time correctness.")
    pdf.bullet("Validation thresholds are fixed in config; a production setup "
               "would add alerting and quarantine-rate monitoring over time.")

    out = REPORTS / "final_report.pdf"
    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
