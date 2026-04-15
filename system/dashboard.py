"""
Dashboard — AG FedAvg / MNIST
Uso: streamlit run dashboard.py
"""

import streamlit as st
import subprocess
import json
import os
import csv
import time
from datetime import datetime

import pandas as pd

st.set_page_config(
    page_title="AG FedAvg/MNIST",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSV_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "csv"))
INDIVIDUALS_CSV = os.path.join(CSV_DIR, "optimization_individuals.csv")
GENERATIONS_CSV = os.path.join(CSV_DIR, "optimization_generations.csv")
AG_SUMMARY_CSV  = os.path.join(CSV_DIR, "ag_mnist_fedavg_summary_latest.csv")

BASELINE_FITNESS = 0.4222
LAMBDA           = 0.1


# ── helpers ───────────────────────────────────────────────────────────────────

def process_alive():
    try:
        r = subprocess.run(["pgrep", "-f", "run_ag_mnist_fedavg.py"],
                           capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def get_gpu_info():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ], text=True)
        gpus = []
        for line in out.strip().split("\n"):
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 5:
                gpus.append(dict(id=p[0], util=float(p[1]),
                                 mem_used=float(p[2]), mem_total=float(p[3]),
                                 temp=float(p[4])))
        return gpus
    except Exception:
        return []


def load_individuals():
    if not os.path.exists(INDIVIDUALS_CSV):
        return []
    rows = []
    with open(INDIVIDUALS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            try:
                genes = json.loads(r.get("sigma_per_client", "[]"))
                fitness = float(r["fitness"])
                acc     = float(r["accuracy"])
                eps     = float(r["epsilon"])
                rows.append({
                    "Gen":     int(r["generation"]),
                    "Ind":     int(r["individual_idx"]),
                    "σ_start": round(float(genes[0]), 3) if len(genes) > 0 else None,
                    "σ_end":   round(float(genes[1]), 3) if len(genes) > 1 else None,
                    "C":       round(float(genes[2]), 3) if len(genes) > 2 else None,
                    "Acc":     round(acc,     4),
                    "ε":       round(eps,     4),
                    "Fitness": round(fitness, 4),
                    "Δ fixo":  round(fitness - BASELINE_FITNESS, 4),
                })
            except Exception:
                continue
    return rows


def load_generations():
    if not os.path.exists(GENERATIONS_CSV):
        return []
    rows = []
    with open(GENERATIONS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "Gen":          int(r["generation"]),
                    "Total_gen":    int(r["total_generations"]),
                    "Best_fitness": round(float(r["best_fitness"]),  4),
                    "Avg_fitness":  round(float(r["avg_fitness"]),   4),
                    "Best_acc":     round(float(r["best_accuracy"]), 4),
                    "Best_ε":       round(float(r["best_epsilon"]),  4),
                })
            except Exception:
                continue
    return rows


def load_summary():
    if not os.path.exists(AG_SUMMARY_CSV):
        return pd.DataFrame()
    try:
        return pd.read_csv(AG_SUMMARY_CSV)
    except Exception:
        return pd.DataFrame()


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    refresh = st.slider("Auto-refresh (s)", 2, 30, 5)

# ── título ────────────────────────────────────────────────────────────────────

st.title("AG — FedAvg / MNIST")
st.caption(
    "Cromossomo [σ_start, σ_end, C]  ·  Fitness = Acc − 0.1·ε  ·  "
    "20 clientes · 20 rounds  ·  Referência σ=4 fixo: Fitness = 0.4222"
)

alive = process_alive()
inds  = load_individuals()
gens  = load_generations()

# ── Status bar ────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if alive:
        st.success("AG rodando")
    else:
        st.info("Aguardando execução")

with c2:
    last_gen  = gens[-1]["Gen"]       if gens else 0
    total_gen = gens[-1]["Total_gen"] if gens else 20
    st.metric("Geração", f"{last_gen} / {total_gen}")

with c3:
    st.metric("Indivíduos", len(inds))

with c4:
    if inds:
        best = max(inds, key=lambda x: x["Fitness"])
        delta_str = f"{'▲' if best['Δ fixo'] >= 0 else '▼'} {best['Δ fixo']:+.4f} vs fixo"
        st.metric("Melhor Fitness", f"{best['Fitness']:.4f}", delta=delta_str)
    else:
        st.metric("Melhor Fitness", "—")

with c5:
    st.metric("Alvo (σ=4 fixo)", f"{BASELINE_FITNESS:.4f}")

# GPU
gpus = get_gpu_info()
if gpus:
    gcols = st.columns(len(gpus))
    for col, gpu in zip(gcols, gpus):
        with col:
            st.markdown(
                f"**GPU {gpu['id']}** — {gpu['util']:.0f}% · "
                f"{gpu['mem_used']:.0f}/{gpu['mem_total']:.0f} MiB · {gpu['temp']:.0f}°C"
            )
            st.progress(gpu["util"] / 100)

st.divider()

# ── Curvas de treinamento por indivíduo ──────────────────────────────────────

st.subheader("Curvas por indivíduo (rounds de avaliação)")

training_csv = os.path.join(CSV_DIR, "training_rounds.csv")
if os.path.exists(training_csv):
    try:
        df_tr = pd.read_csv(training_csv)
        df_opt = df_tr[df_tr["goal"] == "optimization"].copy()

        if not df_opt.empty:
            # Monta label por run_id usando os indivíduos carregados
            ind_map = {}
            for ind in inds:
                genes_key = f"{ind['σ_start']}_{ind['σ_end']}_{ind['C']}"
                ind_map[genes_key] = f"G{ind['Gen']}-I{ind['Ind']}"

            acc_chart = {}
            eps_chart = {}
            fit_chart = {}

            for run_id, grp in df_opt.groupby("run_id"):
                grp = grp.sort_values("round")
                # Tenta associar ao indivíduo pelo sigma_mean
                label = str(run_id)[:8]
                for ind in inds:
                    if ind["σ_start"] is not None:
                        sm = grp["sigma_mean"].iloc[0] if not grp.empty else None
                        if sm is not None and abs(sm - ind["σ_start"]) < 0.5:
                            label = f"G{ind['Gen']}-I{ind['Ind']}"
                            break

                acc_s = grp["test_acc"].reset_index(drop=True)
                eps_s = grp["epsilon"].reset_index(drop=True)
                acc_chart[label] = acc_s
                eps_chart[label] = eps_s
                fit_chart[label] = (acc_s - LAMBDA * eps_s)

            df_acc = pd.DataFrame(acc_chart)
            df_eps = pd.DataFrame(eps_chart)
            df_fit = pd.DataFrame(fit_chart)

            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.markdown("**Acurácia × Round**")
                st.line_chart(df_acc)
            with tc2:
                st.markdown("**ε × Round**")
                st.line_chart(df_eps)
            with tc3:
                st.markdown("**Fitness × Round**")
                if not df_fit.empty:
                    df_fit["Referência"] = BASELINE_FITNESS
                st.line_chart(df_fit)
        else:
            st.info("Aguardando primeiro indivíduo ser avaliado...")
    except Exception as e:
        st.warning(f"Erro ao carregar curvas: {e}")
else:
    st.info("Aguardando dados de treinamento...")

st.divider()

# ── Convergência por geração ──────────────────────────────────────────────────

st.subheader("Convergência do AG")

if gens:
    df_gens = pd.DataFrame(gens)

    cg1, cg2, cg3 = st.columns(3)

    with cg1:
        st.markdown("**Fitness × Geração**")
        chart = df_gens.set_index("Gen")[["Best_fitness", "Avg_fitness"]].copy()
        chart["Referência"] = BASELINE_FITNESS
        st.line_chart(chart)

    with cg2:
        st.markdown("**Best Acc × Geração**")
        st.line_chart(df_gens.set_index("Gen")[["Best_acc"]])

    with cg3:
        st.markdown("**Best ε × Geração**")
        st.line_chart(df_gens.set_index("Gen")[["Best_ε"]])
else:
    st.info("Aguardando primeira geração completa...")

st.divider()

# ── Tabela de indivíduos ──────────────────────────────────────────────────────

st.subheader("Indivíduos avaliados")

if inds:
    df_inds = pd.DataFrame(inds).sort_values(["Gen", "Ind"]).reset_index(drop=True)

    st.dataframe(
        df_inds,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fitness": st.column_config.NumberColumn(format="%.4f"),
            "Acc":     st.column_config.NumberColumn(format="%.4f"),
            "ε":       st.column_config.NumberColumn(format="%.4f"),
            "σ_start": st.column_config.NumberColumn(format="%.3f"),
            "σ_end":   st.column_config.NumberColumn(format="%.3f"),
            "C":       st.column_config.NumberColumn(format="%.3f"),
            "Δ fixo":  st.column_config.NumberColumn("Δ vs fixo", format="%.4f"),
        },
    )

    best = max(inds, key=lambda x: x["Fitness"])
    superaram = sum(1 for i in inds if i["Fitness"] > BASELINE_FITNESS)
    st.caption(
        f"Melhor: Gen {best['Gen']} Ind {best['Ind']} — "
        f"σ_start={best['σ_start']}  σ_end={best['σ_end']}  C={best['C']} — "
        f"Fitness={best['Fitness']:.4f}  Acc={best['Acc']:.4f}  ε={best['ε']:.4f}  |  "
        f"{superaram}/{len(inds)} superaram o baseline fixo"
    )
else:
    st.info("Nenhum indivíduo avaliado ainda.")

st.divider()

# ── Pareto Acc × ε ────────────────────────────────────────────────────────────

st.subheader("Pareto — Acc × ε")

if inds:
    df_pareto = pd.DataFrame(inds)[["σ_start", "σ_end", "C", "Acc", "ε", "Fitness"]].copy()
    ref = pd.DataFrame([{
        "σ_start": 4.0, "σ_end": 4.0, "C": 1.0,
        "Acc": 0.481, "ε": 0.5881, "Fitness": BASELINE_FITNESS,
    }])
    df_pareto = pd.concat([ref, df_pareto], ignore_index=True)

    pa, pb = st.columns(2)
    with pa:
        st.markdown("**Scatter Acc × ε** (canto sup-esq = melhor tradeoff)")
        st.scatter_chart(df_pareto, x="ε", y="Acc", color="Fitness")

    with pb:
        st.markdown("**Fitness por indivíduo (ordem avaliação)**")
        df_bar = pd.DataFrame(inds).copy()
        df_bar["label"] = df_bar.apply(lambda r: f"G{r['Gen']}-I{r['Ind']}", axis=1)
        bar_data = df_bar.set_index("label")[["Fitness"]].copy()
        bar_data["Referência"] = BASELINE_FITNESS
        st.bar_chart(bar_data)

st.divider()

# ── Tabela AG vs Fixos ────────────────────────────────────────────────────────

st.subheader("AG vs Sigma Fixo — Superioridade de Fitness")

if inds:
    best = max(inds, key=lambda x: x["Fitness"])
    ag_fitness = best["Fitness"]

    FIXOS = [
        {"Config": "σ=4 (melhor fixo)", "Fitness fixo": 0.4336},
        {"Config": "σ=12",              "Fitness fixo": 0.3619},
        {"Config": "σ=20",              "Fitness fixo": 0.2720},
        {"Config": "σ=1",               "Fitness fixo": 0.1055},
    ]

    rows = []
    for f in FIXOS:
        delta = ag_fitness - f["Fitness fixo"]
        pct   = (ag_fitness / f["Fitness fixo"] - 1) * 100
        rows.append({
            "Config":       f["Config"],
            "Fitness fixo": round(f["Fitness fixo"], 4),
            "AG Fitness":   round(ag_fitness, 4),
            "Δ absoluto":   round(delta, 4),
            "% superior":   round(pct, 1),
        })

    df_vs = pd.DataFrame(rows)
    st.dataframe(
        df_vs,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fitness fixo": st.column_config.NumberColumn(format="%.4f"),
            "AG Fitness":   st.column_config.NumberColumn(format="%.4f"),
            "Δ absoluto":   st.column_config.NumberColumn(format="+%.4f"),
            "% superior":   st.column_config.NumberColumn(format="+%.1f%%"),
        },
    )
    st.caption(
        f"AG melhor: Gen {best['Gen']} Ind {best['Ind']} — "
        f"σ [{best['σ_start']}→{best['σ_end']}]  C={best['C']}  "
        f"Fitness={ag_fitness:.4f}  Acc={best['Acc']*100:.2f}%  ε={best['ε']:.4f}"
    )

st.divider()

# ── Resultado final do AG ─────────────────────────────────────────────────────

st.subheader("Resultado final — melhor cromossomo")

df_summary = load_summary()
if not df_summary.empty:
    ag_rows = df_summary[df_summary["run_type"] == "adaptive"]
    if not ag_rows.empty:
        s = ag_rows.iloc[-1]
        ag_fitness = float(s["best_acc"]) - LAMBDA * float(s["final_epsilon"])
        delta = ag_fitness - BASELINE_FITNESS

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Acc máx (AG)", f"{float(s['best_acc'])*100:.2f}%")
        with r2:
            st.metric("ε final (AG)", f"{float(s['final_epsilon']):.4f}")
        with r3:
            delta_str = f"{'▲' if delta >= 0 else '▼'} {delta:+.4f} vs σ=4 fixo"
            st.metric("Fitness (AG)", f"{ag_fitness:.4f}", delta=delta_str)
        with r4:
            st.metric("C (clipping)", f"{float(s['clip']):.3f}")

        if delta > 0:
            st.success(f"AG SUPEROU o baseline fixo em +{delta:.4f}")
        else:
            st.warning(f"AG abaixo do baseline em {delta:.4f}")

        st.caption(
            f"Melhor cromossomo: σ_start={float(s['sigma_start']):.3f}  "
            f"σ_end={float(s['sigma_end']):.3f}  C={float(s['clip']):.3f}"
        )
    else:
        st.info("Resultado final disponível após o AG concluir a avaliação final.")
else:
    st.info("Resultado final disponível após o AG concluir.")

# ── Sigma Fixo — referência ───────────────────────────────────────────────────

st.divider()
st.subheader("Referência — Sigma Fixo FedAvg / MNIST")
st.caption("20 rounds · 20 clientes · C=1.0  |  Fitness = Acc − 0.1·ε  (o que o AG precisa superar)")

FIXOS_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "result-fixos", "csv",
                 "fixed_sigma_mnist_20260413_104725.csv")
)

try:
    df_fix = pd.read_csv(FIXOS_CSV)
    df_fix = df_fix[df_fix["algorithm"] == "FedAvg"].copy()

    if not df_fix.empty:
        df_fix["fitness"] = df_fix["test_acc"] - LAMBDA * df_fix["epsilon"]

        acc_chart = {}
        eps_chart = {}
        fit_chart = {}
        summary_fixo = []

        for sigma, grp in df_fix.groupby("sigma"):
            grp = grp.sort_values("round")
            label = f"σ={int(sigma)}"
            acc_chart[label] = grp["test_acc"].reset_index(drop=True)
            eps_chart[label] = grp["epsilon"].reset_index(drop=True)
            fit_chart[label] = grp["fitness"].reset_index(drop=True)

            best_fit = grp["fitness"].max()
            summary_fixo.append({
                "σ":             int(sigma),
                "Acc_max (%)":   round(grp["test_acc"].max() * 100, 2),
                "ε_final":       round(grp["epsilon"].iloc[-1], 4),
                "Fitness_max":   round(best_fit, 4),
                "Fitness_final": round(grp["fitness"].iloc[-1], 4),
            })

        fa, fb, fc = st.columns(3)

        with fa:
            st.markdown("**Fitness × Round**")
            st.line_chart(pd.DataFrame(fit_chart))

        with fb:
            st.markdown("**Acurácia × Round**")
            st.line_chart(pd.DataFrame(acc_chart))

        with fc:
            st.markdown("**ε × Round**")
            st.line_chart(pd.DataFrame(eps_chart))

        df_sf = pd.DataFrame(summary_fixo).sort_values("σ").reset_index(drop=True)
        st.dataframe(
            df_sf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Acc_max (%)":   st.column_config.NumberColumn(format="%.2f"),
                "ε_final":       st.column_config.NumberColumn(format="%.4f"),
                "Fitness_max":   st.column_config.NumberColumn(format="%.4f"),
                "Fitness_final": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        best = df_sf.loc[df_sf["Fitness_max"].idxmax()]
        st.caption(f"Melhor fixo: σ={int(best['σ'])}  →  Fitness_max={best['Fitness_max']:.4f}  "
                   f"(alvo do AG)")
    else:
        st.info("Sem dados FedAvg no CSV de referência.")

except Exception as e:
    st.error(f"Erro ao carregar referência: {e}")

# ── auto-refresh ──────────────────────────────────────────────────────────────

st.caption(f"Atualizado: {datetime.now().strftime('%H:%M:%S')} — próximo em {refresh}s")
time.sleep(refresh)
st.rerun()
