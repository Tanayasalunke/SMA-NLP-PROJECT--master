import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindTrace: Clinical Mental Health NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .highlight-phq9 { background-color: #DBEAFE; color: #1E40AF; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
    .highlight-gad7 { background-color: #FEF3C7; color: #B45309; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
    .highlight-pcl5 { background-color: #F3E8FF; color: #6B21A8; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
    .highlight-spin { background-color: #FCE7F3; color: #9D174D; padding: 2px 6px; border-radius: 4px; font-weight: 600; }
    .highlight-artifact { background-color: #FEE2E2; color: #991B1B; padding: 2px 6px; border-radius: 4px; font-weight: 600; text-decoration: line-through; }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading & Caching ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

ARTIFACT_TERMS = [
    "depression", "anxiety", "ptsd", "social anxiety", "socialanxiety",
    "diagnosed", "diagnosis", "dae", "throwaway", "tw:", "trigger warning",
    "r/depression", "r/anxiety", "r/ptsd", "r/socialanxiety",
]
ARTIFACT_TERMS_WB = ["sa", "gad"]

def scrub_text(text):
    t = str(text).lower()
    for term in ARTIFACT_TERMS:
        t = re.sub(re.escape(term), "", t)
    for term in ARTIFACT_TERMS_WB:
        t = re.sub(r"\b" + re.escape(term) + r"\b", "", t)
    return re.sub(r"\s+", " ", t).strip()

@st.cache_resource
def load_models_and_lexicons():
    with open(os.path.join(BASE, "model_scrubbed.pkl"), "rb") as f:
        scrubbed_bundle = pickle.load(f)
    with open(os.path.join(BASE, "model_unscrubbed.pkl"), "rb") as f:
        unscrubbed_bundle = pickle.load(f)
    with open(os.path.join(BASE, "lexicons.pkl"), "rb") as f:
        lex_data = pickle.load(f)
    return scrubbed_bundle, unscrubbed_bundle, lex_data

@st.cache_data
def load_all_results():
    results = {}
    for m in ["xgboost", "bilstm", "roberta", "mentalbert"]:
        p = os.path.join(BASE, f"{m}_results.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                results[m] = pickle.load(f)
    alignments = {}
    for m in ["xgboost", "roberta", "mentalbert"]:
        p = os.path.join(BASE, f"{m}_alignment.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                alignments[m] = pickle.load(f)
    return results, alignments

scrubbed_pkg, unscrubbed_pkg, lexicons_pkg = load_models_and_lexicons()
results_data, alignments_data = load_all_results()

lexicons = lexicons_pkg.get("lexicons", {})

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=72)
    st.title("MindTrace")
    st.markdown("**Multi-Class Mental Health Classifier & Clinical Alignment Validator**")
    st.markdown("---")
    
    app_mode = st.radio(
        "Navigation",
        [
            "🔮 Diagnostic Text Screener",
            "⚖️ Shortcut vs. Clinical Validator",
            "📊 Multi-Model Research Benchmark",
            "📚 Clinical Lexicons Explorer",
        ]
    )
    
    st.markdown("---")
    st.caption("Developed by Tanaya Salunke")
    st.caption("Research: Social Media NLP & Clinical Explainability")

# ── Helper for Clinical Keyword Tagging ─────────────────────────────────────
def annotate_clinical_text(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    highlighted = []
    found_instruments = set()
    
    # Flatten lexicons with instrument tag
    word_to_scale = {}
    for scale, words in lexicons.items():
        for w in words:
            word_to_scale[w.lower()] = scale
            
    artifacts = set([a.lower() for a in ARTIFACT_TERMS + ARTIFACT_TERMS_WB])

    for tok in tokens:
        low = tok.lower()
        if low in artifacts:
            highlighted.append(f'<span class="highlight-artifact">{tok} (shortcut)</span>')
            found_instruments.add("Shortcut/Artifact")
        elif low in word_to_scale:
            scale = word_to_scale[low]
            found_instruments.add(scale)
            css_cls = f"highlight-{scale.lower()}"
            highlighted.append(f'<span class="{css_cls}">{tok}<sup>[{scale}]</sup></span>')
        else:
            highlighted.append(tok)
            
    return " ".join(highlighted), found_instruments


# ────────────────────────────────────────────────────────────────────────────
# 1. Diagnostic Text Screener
# ────────────────────────────────────────────────────────────────────────────
if app_mode == "🔮 Diagnostic Text Screener":
    st.markdown('<div class="main-header">🔮 Interactive Mental Health Diagnostic</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classifies natural language Reddit posts into 5 psychological states and scans for DSM-5 / clinical scale symptoms.</div>', unsafe_allow_html=True)
    
    samples = {
        "Custom Input": "",
        "Depression Sample": "I feel completely hollow and exhausted every single day. Getting out of bed is impossible, I have lost interest in food and everything I used to enjoy. Feel like a complete burden to everyone.",
        "Anxiety Sample": "My heart is constantly pounding out of my chest and I have this persistent dread that something terrible is about to happen. Can't relax or sleep because my brain won't shut off.",
        "PTSD Sample": "Ever since the car crash assault, sudden loud noises trigger violent flashbacks. I wake up in cold sweats from recurring nightmares and constantly stay on high alert.",
        "Social Anxiety Sample": "Whenever I have to give a presentation or enter a room full of people, I start sweating, trembling, and obsessing over being scrutinized and embarrassed.",
        "Control / Neutral Sample": "Just finished upgrading my budget spreadsheet in Excel to track my monthly retirement investments and groceries."
    }
    
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_sample = st.selectbox("Load sample post:", list(samples.keys()))
    with col1:
        default_val = samples[selected_sample] if selected_sample != "Custom Input" else ""
        user_input = st.text_area("Enter post text:", value=default_val, height=140, placeholder="Type or paste post text here...")

    if user_input.strip():
        model = scrubbed_pkg["model"]
        vectorizer = scrubbed_pkg["vectorizer"]
        label_encoder = scrubbed_pkg["label_encoder"]
        
        # Scrub and vectorize
        scrubbed = scrub_text(user_input)
        vec = vectorizer.transform([scrubbed])
        probs = model.predict_proba(vec)[0]
        classes = label_encoder.classes_
        
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        pred_prob = probs[pred_idx]
        
        label_format = {
            "depression": ("Depression (PHQ-9)", "#2563EB", "🌧️"),
            "anxiety": ("Generalized Anxiety (GAD-7)", "#D97706", "⚡"),
            "ptsd": ("PTSD (PCL-5)", "#7C3AED", "🛡️"),
            "social_anxiety": ("Social Anxiety (SPIN)", "#DB2777", "👥"),
            "control": ("Control / Non-Clinical", "#059669", "🌿")
        }
        
        title, color, icon = label_format.get(pred_label, (pred_label.title(), "#4B5563", "🔍"))
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1.5, 2.5])
        
        with res_col1:
            st.markdown(f"""
            <div style="background-color: {color}15; border: 2px solid {color}; border-radius: 12px; padding: 24px; text-align: center;">
                <div style="font-size: 3rem;">{icon}</div>
                <div style="font-size: 1.1rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Predicted State</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {color}; margin-top: 4px;">{title}</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #1E293B; margin-top: 8px;">{pred_prob*100:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            prob_df = pd.DataFrame({
                "Condition": [label_format[c][0] for c in classes],
                "Probability": probs,
                "Color": [label_format[c][1] for c in classes]
            }).sort_values("Probability", ascending=True)
            
            fig = px.bar(
                prob_df,
                x="Probability",
                y="Condition",
                orientation="h",
                text=prob_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
                title="<b>Posterior Class Distribution</b>",
                color="Condition",
                color_discrete_map={row["Condition"]: row["Color"] for _, row in prob_df.iterrows()}
            )
            fig.update_layout(showlegend=False, height=240, margin=dict(l=0, r=20, t=35, b=0), xaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔬 Clinical Symptom 'X-Ray'")
        annotated_html, scales_found = annotate_clinical_text(user_input)
        
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 18px; border-radius: 8px; font-size: 1.05rem; line-height: 1.8;">
            {annotated_html}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        legend_cols = st.columns(5)
        legend_cols[0].markdown('<span class="highlight-phq9">PHQ-9 (Depression)</span>', unsafe_allow_html=True)
        legend_cols[1].markdown('<span class="highlight-gad7">GAD-7 (Anxiety)</span>', unsafe_allow_html=True)
        legend_cols[2].markdown('<span class="highlight-pcl5">PCL-5 (PTSD)</span>', unsafe_allow_html=True)
        legend_cols[3].markdown('<span class="highlight-spin">SPIN (Social Anxiety)</span>', unsafe_allow_html=True)
        legend_cols[4].markdown('<span class="highlight-artifact">Shortcut Artifact</span>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# 2. Shortcut vs. Clinical Validator
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "⚖️ Shortcut vs. Clinical Validator":
    st.markdown('<div class="main-header">⚖️ Dataset Shortcut vs. Clinical Validator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Demonstrating how Reddit models cheat by learning subreddit names ("r/ptsd", "diagnosed") versus learning genuine symptom language.</div>', unsafe_allow_html=True)
    
    st.info("💡 **Core Paper Finding:** Standard unscrubbed models achieve high accuracy artificially by memorizing community names. When scrubbed, spurious shortcuts collapse and the model is forced to rely on real clinical symptoms.")
    
    test_shortcut_text = st.text_input(
        "Test an input with Reddit artifact shortcuts:",
        value="I was diagnosed with ptsd on r/ptsd and my anxiety is terrible, having flashbacks every night."
    )
    
    if test_shortcut_text:
        scrubbed_variant = scrub_text(test_shortcut_text)
        
        col_raw, col_scrub = st.columns(2)
        
        # Unscrubbed model inference
        unscrub_model = unscrubbed_pkg["model"]
        unscrub_vec = unscrubbed_pkg["vectorizer"]
        raw_probs = unscrub_model.predict_proba(unscrub_vec.transform([test_shortcut_text]))[0]
        raw_pred = unscrubbed_pkg["label_encoder"].classes_[np.argmax(raw_probs)]
        
        # Scrubbed model inference
        scrub_model = scrubbed_pkg["model"]
        scrub_vec = scrubbed_pkg["vectorizer"]
        clean_probs = scrub_model.predict_proba(scrub_vec.transform([scrubbed_variant]))[0]
        clean_pred = scrubbed_pkg["label_encoder"].classes_[np.argmax(clean_probs)]
        
        with col_raw:
            st.markdown("#### 1️⃣ Unscrubbed Model (Susceptible to Shortcuts)")
            st.markdown(f"**Input:** *\"{test_shortcut_text}\"*")
            st.markdown(f"**Predicted:** `:red[{raw_pred.upper()}]` ({np.max(raw_probs)*100:.1f}%)")
            st.caption("Notice: If you mention 'diagnosed with ptsd', the model shortcuts straight to PTSD without evaluating the rest.")
            
        with col_scrub:
            st.markdown("#### 2️⃣ Scrubbed Model (Clinical Symptom Focus)")
            st.markdown(f"**Scrubbed Input:** *\"{scrubbed_variant}\"*")
            st.markdown(f"**Predicted:** `:green[{clean_pred.upper()}]` ({np.max(clean_probs)*100:.1f}%)")
            st.caption("Artifacts stripped away. Decision driven strictly by remaining symptom language ('flashbacks every night').")
            
        st.markdown("---")
        st.subheader("📈 Verified Symptom Alignment Score (SAS) Breakdown")
        st.markdown("SHAP mass decomposition for XGBoost under both conditions:")
        
        xg_align = alignments_data.get("xgboost", {})
        if xg_align:
            unscrub_df = pd.DataFrame(xg_align["unscrubbed"]).T.reset_index().rename(columns={"index": "Class"})
            scrub_df = pd.DataFrame(xg_align["scrubbed"]).T.reset_index().rename(columns={"index": "Class"})
            
            tab_u, tab_s = st.tabs(["Unscrubbed Arm (High Spurious Shortcut Mass)", "Scrubbed Arm (Zero Spurious Shortcuts)"])
            
            with tab_u:
                fig_u = px.bar(
                    unscrub_df,
                    x="Class",
                    y=["SAS", "SPUR", "AMB"],
                    title="Unscrubbed Model: High Spurious Shortcut Leakage (SPUR in Red)",
                    labels={"value": "Fraction of SHAP Mass", "variable": "Component"},
                    color_discrete_map={"SAS": "#10B981", "SPUR": "#EF4444", "AMB": "#94A3B8"}
                )
                st.plotly_chart(fig_u, use_container_width=True)
                
            with tab_s:
                fig_s = px.bar(
                    scrub_df,
                    x="Class",
                    y=["SAS", "SPUR", "AMB"],
                    title="Scrubbed Model: True Clinical Alignment (SAS in Green, Zero SPUR)",
                    labels={"value": "Fraction of SHAP Mass", "variable": "Component"},
                    color_discrete_map={"SAS": "#10B981", "SPUR": "#EF4444", "AMB": "#94A3B8"}
                )
                st.plotly_chart(fig_s, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# 3. Multi-Model Research Benchmark
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "📊 Multi-Model Research Benchmark":
    st.markdown('<div class="main-header">📊 Multi-Model Benchmark & Research Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating 4 model architectures across both scrubbed and unscrubbed experimental arms.</div>', unsafe_allow_html=True)
    
    rows = []
    models_display = {
        "xgboost": "XGBoost + TF-IDF",
        "bilstm": "BiLSTM + GloVe",
        "roberta": "RoBERTa (Fine-Tuned)",
        "mentalbert": "MentalBERT (Domain Specific)"
    }
    
    for m_key, m_name in models_display.items():
        if m_key in results_data:
            res = results_data[m_key]
            u_f1 = res["unscrubbed"]["classification_report"]["macro avg"]["f1-score"]
            s_f1 = res["scrubbed"]["classification_report"]["macro avg"]["f1-score"]
            delta = (s_f1 - u_f1)
            rows.append({
                "Model Architecture": m_name,
                "Unscrubbed Macro F1": u_f1,
                "Scrubbed Macro F1": s_f1,
                "Delta (Robustness Drop)": delta,
                "Retention Rate": f"{(s_f1 / u_f1)*100:.1f}%"
            })
            
    df_results = pd.DataFrame(rows)
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.dataframe(
            df_results.style.format({
                "Unscrubbed Macro F1": "{:.3f}",
                "Scrubbed Macro F1": "{:.3f}",
                "Delta (Robustness Drop)": "{:.3f}"
            }).background_gradient(subset=["Scrubbed Macro F1"], cmap="Greens"),
            use_container_width=True,
            hide_index=True
        )
    with col_t2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #64748B;">Top Performing Model</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #10B981; margin-top: 4px;">MentalBERT</div>
            <div style="font-size: 0.95rem; color: #334155; margin-top: 4px;">Scrubbed F1: <b>0.836</b></div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 6px;">Pre-trained directly on mental health domain corpora.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Visual comparison bar chart
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Unscrubbed (Raw)",
        x=df_results["Model Architecture"],
        y=df_results["Unscrubbed Macro F1"],
        marker_color="#3B82F6",
        text=df_results["Unscrubbed Macro F1"].apply(lambda v: f"{v:.3f}"),
        textposition="auto"
    ))
    fig_comp.add_trace(go.Bar(
        name="Scrubbed (Clinical Language Only)",
        x=df_results["Model Architecture"],
        y=df_results["Scrubbed Macro F1"],
        marker_color="#10B981",
        text=df_results["Scrubbed Macro F1"].apply(lambda v: f"{v:.3f}"),
        textposition="auto"
    ))
    fig_comp.update_layout(
        title="<b>Macro F1 Score: Unscrubbed vs. Scrubbed Condition Across All Architectures</b>",
        barmode="group",
        height=400,
        yaxis=dict(range=[0.6, 0.95], title="Macro F1-Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.markdown("### 🖼️ Publication Figures")
    fig_cols = st.columns(3)
    if os.path.exists(os.path.join(BASE, "fig1_stacked.png")):
        fig_cols[0].image(os.path.join(BASE, "fig1_stacked.png"), caption="Figure 1: Stacked Alignment Breakdown")
    if os.path.exists(os.path.join(BASE, "fig5_confusion.png")):
        fig_cols[1].image(os.path.join(BASE, "fig5_confusion.png"), caption="Figure 5: Scrubbed Confusion Matrix")
    if os.path.exists(os.path.join(BASE, "fig6_model_comparison.png")):
        fig_cols[2].image(os.path.join(BASE, "fig6_model_comparison.png"), caption="Figure 6: Cross-Model Comparison")


# ────────────────────────────────────────────────────────────────────────────
# 4. Clinical Lexicons Explorer
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "📚 Clinical Lexicons Explorer":
    st.markdown('<div class="main-header">📚 Validated Clinical Lexicons</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Official psychiatric survey keywords mapped to the diagnostic targets.</div>', unsafe_allow_html=True)
    
    scale_tabs = st.tabs(["PHQ-9 (Depression)", "GAD-7 (Anxiety)", "PCL-5 (PTSD)", "SPIN (Social Anxiety)", "Artifact Shortcut List"])
    
    with scale_tabs[0]:
        st.markdown("**Patient Health Questionnaire-9 (PHQ-9)** — 37 Clinical Symptom Terms:")
        st.write(", ".join(sorted(list(lexicons.get("PHQ9", [])))))
        
    with scale_tabs[1]:
        st.markdown("**Generalized Anxiety Disorder-7 (GAD-7)** — 25 Clinical Symptom Terms:")
        st.write(", ".join(sorted(list(lexicons.get("GAD7", [])))))
        
    with scale_tabs[2]:
        st.markdown("**PTSD Checklist for DSM-5 (PCL-5)** — 65 Clinical Symptom Terms:")
        st.write(", ".join(sorted(list(lexicons.get("PCL5", [])))))
        
    with scale_tabs[3]:
        st.markdown("**Social Phobia Inventory (SPIN)** — 37 Clinical Symptom Terms:")
        st.write(", ".join(sorted(list(lexicons.get("SPIN", [])))))
        
    with scale_tabs[4]:
        st.markdown("**Spurious Subreddit Shortcuts Stripped During Scrubbing:**")
        st.write(", ".join(sorted(list(ARTIFACT_TERMS + ARTIFACT_TERMS_WB))))
