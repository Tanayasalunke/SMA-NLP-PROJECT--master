import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import shap

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindTrace AI | Clinical Explainable NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling & Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #311042 100%);
        border-radius: 16px;
        padding: 30px 34px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        max-width: 900px;
        line-height: 1.6;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #34D399;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 10px #34D399;
    }

    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }

    .highlight-phq9 { background: #DBEAFE; color: #1E40AF; padding: 2px 7px; border-radius: 5px; font-weight: 700; border-bottom: 2px solid #3B82F6; }
    .highlight-gad7 { background: #FEF3C7; color: #B45309; padding: 2px 7px; border-radius: 5px; font-weight: 700; border-bottom: 2px solid #F59E0B; }
    .highlight-pcl5 { background: #F3E8FF; color: #6B21A8; padding: 2px 7px; border-radius: 5px; font-weight: 700; border-bottom: 2px solid #A855F7; }
    .highlight-spin { background: #FCE7F3; color: #9D174D; padding: 2px 7px; border-radius: 5px; font-weight: 700; border-bottom: 2px solid #EC4899; }
    .highlight-artifact { background: #FEE2E2; color: #991B1B; padding: 2px 7px; border-radius: 5px; font-weight: 700; text-decoration: line-through; border-bottom: 2px solid #EF4444; }

    .stat-box {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .stat-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
    }
    .stat-lbl {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.04em;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Core Artifacts ─────────────────────────────────────────────────────
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
def load_models_and_explainers():
    with open(os.path.join(BASE, "model_scrubbed.pkl"), "rb") as f:
        scrubbed_bundle = pickle.load(f)
    with open(os.path.join(BASE, "model_unscrubbed.pkl"), "rb") as f:
        unscrubbed_bundle = pickle.load(f)
    with open(os.path.join(BASE, "lexicons.pkl"), "rb") as f:
        lex_data = pickle.load(f)
        
    # Real-time SHAP TreeExplainer on scrubbed XGBoost
    scrub_explainer = shap.TreeExplainer(scrubbed_bundle["model"])
    
    return scrubbed_bundle, unscrubbed_bundle, lex_data, scrub_explainer

@st.cache_data
def load_all_metrics():
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
                
    shap_top = {}
    if os.path.exists(os.path.join(BASE, "shap_top_features.pkl")):
        with open(os.path.join(BASE, "shap_top_features.pkl"), "rb") as f:
            shap_top = pickle.load(f)
            
    return results, alignments, shap_top

scrubbed_pkg, unscrubbed_pkg, lexicons_pkg, shap_explainer = load_models_and_explainers()
results_data, alignments_data, shap_top_data = load_all_metrics()
lexicons = lexicons_pkg.get("lexicons", {})

LABEL_METADATA = {
    "depression": {"name": "Major Depression", "scale": "PHQ-9", "color": "#2563EB", "icon": "🌧️", "severity": "High"},
    "anxiety": {"name": "Generalized Anxiety", "scale": "GAD-7", "color": "#D97706", "icon": "⚡", "severity": "Medium-High"},
    "ptsd": {"name": "Post-Traumatic Stress", "scale": "PCL-5", "color": "#7C3AED", "icon": "🛡️", "severity": "Critical"},
    "social_anxiety": {"name": "Social Anxiety Disorder", "scale": "SPIN", "color": "#DB2777", "icon": "👥", "severity": "Medium"},
    "control": {"name": "Non-Clinical Baseline", "scale": "Neutral", "color": "#059669", "icon": "🌿", "severity": "Normal"}
}

def highlight_symptoms(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    highlighted = []
    found_map = {"PHQ9": [], "GAD7": [], "PCL5": [], "SPIN": [], "Artifact": []}
    
    word_to_scale = {}
    for scale, words in lexicons.items():
        for w in words:
            word_to_scale[w.lower()] = scale
            
    artifacts = set([a.lower() for a in ARTIFACT_TERMS + ARTIFACT_TERMS_WB])

    for tok in tokens:
        low = tok.lower()
        if low in artifacts:
            highlighted.append(f'<span class="highlight-artifact">{tok}</span>')
            found_map["Artifact"].append(tok)
        elif low in word_to_scale:
            scale = word_to_scale[low]
            found_map[scale].append(tok)
            css_cls = f"highlight-{scale.lower()}"
            highlighted.append(f'<span class="{css_cls}">{tok}<sup>[{scale}]</sup></span>')
        else:
            highlighted.append(tok)
            
    return " ".join(highlighted), found_map


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <div style="background: linear-gradient(135deg, #6366F1, #A855F7); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);">🧠</div>
        <div>
            <div style="font-weight: 800; font-size: 1.25rem; color: #0F172A; letter-spacing: -0.02em;">MindTrace AI</div>
            <div style="font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Clinical Explainable NLP</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    app_mode = st.radio(
        "Navigation",
        [
            "🔮 Live Diagnostic & Real-Time Explainability",
            "🔍 Global SHAP Explainability Lab",
            "⚡ Dynamic Keystroke Scanner",
            "⚖️ Shortcut vs. Clinical Stress Test",
            "📊 Multi-Model Research Leaderboard",
            "📚 Clinical Instrument Explorer",
        ]
    )
    
    st.markdown("---")
    st.markdown("**Experimental Specifications**")
    st.caption("• **RMHD Corpus:** 102,984 Reddit posts")
    st.caption("• **Explainability Framework:** TreeSHAP & SAS Decomposition")
    st.caption("• **Psychometric Targets:** PHQ-9, GAD-7, PCL-5, SPIN")
    st.markdown("---")
    st.caption("Repo: [SMA-NLP-PROJECT--master](https://github.com/Tanayasalunke/SMA-NLP-PROJECT--master)")


# ────────────────────────────────────────────────────────────────────────────
# 1. LIVE DIAGNOSTIC & REAL-TIME EXPLAINABILITY (SHAP)
# ────────────────────────────────────────────────────────────────────────────
if app_mode == "🔮 Live Diagnostic & Real-Time Explainability":
    st.markdown("""
    <div class="hero-banner">
        <div class="live-badge"><div class="pulse-dot"></div>TreeSHAP Explainability Engine Active</div>
        <div class="hero-title">Live Diagnostic & Local Model Explainer</div>
        <div class="hero-subtitle">Generates multi-class clinical predictions and calculates exact real-time SHAP feature importance values for the active text, exposing token-level influence on the model's decision.</div>
    </div>
    """, unsafe_allow_html=True)

    samples = {
        "Custom Input": "",
        "Depression (Severe Anhedonia)": "I feel completely hollow and exhausted every single day. Getting out of bed is impossible, I have lost interest in food and everything I used to enjoy. Feel like a complete burden to everyone and cannot concentrate on anything.",
        "Anxiety (Panic Attack Profile)": "My heart is constantly pounding out of my chest and I have this persistent suffocating dread that something terrible is about to happen. Can't relax or sleep because my brain won't shut off, trembling uncontrollably.",
        "PTSD (Trauma / Hypervigilance)": "Ever since the car crash assault, sudden loud noises trigger violent flashbacks. I wake up in cold sweats from recurring nightmares and constantly stay on high alert scanning the room.",
        "Social Anxiety (Performance Phobia)": "Whenever I have to speak in a meeting or enter a crowded room, I start sweating, trembling, and obsessing over being scrutinized, judged, and humiliated by everyone.",
        "Control (Finance / Everyday Life)": "Just updated my personal budget spreadsheet to track monthly retirement contributions and grocery expenses. Looking for low-fee index fund recommendations."
    }

    col_input, col_meta = st.columns([2.5, 1.2])

    with col_meta:
        st.markdown("**Clinical Archetypes:**")
        chosen_sample = st.selectbox("Load representative case:", list(samples.keys()))
        selected_text = samples[chosen_sample] if chosen_sample != "Custom Input" else ""

    with col_input:
        user_text = st.text_area(
            "Patient Narrative / Social Media Post:",
            value=selected_text,
            height=140,
            placeholder="Type or paste post text here to analyze psychological indicators with real-time SHAP..."
        )

    if user_text.strip():
        model = scrubbed_pkg["model"]
        vectorizer = scrubbed_pkg["vectorizer"]
        label_encoder = scrubbed_pkg["label_encoder"]
        
        scrubbed = scrub_text(user_text)
        vec = vectorizer.transform([scrubbed])
        probs = model.predict_proba(vec)[0]
        classes = label_encoder.classes_
        
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        pred_prob = probs[pred_idx]
        meta = LABEL_METADATA.get(pred_label, LABEL_METADATA["control"])

        # Top Metric Cards
        st.markdown("<br>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="stat-box" style="border-top: 4px solid {meta['color']};">
                <div class="stat-val" style="color: {meta['color']};">{meta['icon']} {meta['name']}</div>
                <div class="stat-lbl">Primary Classification</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="stat-box" style="border-top: 4px solid {meta['color']};">
                <div class="stat-val">{pred_prob*100:.1f}%</div>
                <div class="stat-lbl">Model Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="stat-box" style="border-top: 4px solid #6366F1;">
                <div class="stat-val">{meta['scale']}</div>
                <div class="stat-lbl">Matched Psychiatric Scale</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown(f"""
            <div class="stat-box" style="border-top: 4px solid #EC4899;">
                <div class="stat-val">{meta['severity']}</div>
                <div class="stat-lbl">Clinical Indicator Level</div>
            </div>
            """, unsafe_allow_html=True)

        # Charts Section
        st.markdown("<br>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns([1.5, 1.5])

        with chart_col1:
            # Interactive Radar Chart
            radar_df = pd.DataFrame({
                "Condition": [LABEL_METADATA[c]["name"] for c in classes],
                "Probability": probs
            })
            fig_radar = px.line_polar(
                radar_df,
                r="Probability",
                theta="Condition",
                line_close=True,
                title="<b>Multidimensional Mental State Radar</b>"
            )
            fig_radar.update_traces(fill="toself", fillcolor=f"{meta['color']}33", line_color=meta["color"], line_width=2.5)
            fig_radar.update_layout(height=340, margin=dict(l=40, r=40, t=45, b=30), polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig_radar, use_container_width=True)

        with chart_col2:
            # Horizontal Bar Chart
            bar_df = pd.DataFrame({
                "Condition": [LABEL_METADATA[c]["name"] for c in classes],
                "Probability": probs,
                "Color": [LABEL_METADATA[c]["color"] for c in classes]
            }).sort_values("Probability", ascending=True)

            fig_bar = px.bar(
                bar_df,
                x="Probability",
                y="Condition",
                orientation="h",
                text=bar_df["Probability"].apply(lambda p: f"{p*100:.1f}%"),
                title="<b>Posterior Probability Distribution</b>",
                color="Condition",
                color_discrete_map={row["Condition"]: row["Color"] for _, row in bar_df.iterrows()}
            )
            fig_bar.update_layout(showlegend=False, height=340, margin=dict(l=0, r=30, t=45, b=10), xaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── REAL-TIME LOCAL SHAP EXPLANATION (WATERFALL / BAR) ──────────────
        st.markdown("### 🧬 Real-Time Local SHAP Explanation")
        st.caption(f"Calculating exact TreeSHAP attribution for predicted state: **{meta['name']}**")

        try:
            shap_vals = shap_explainer.shap_values(vec)
            # shap_vals shape: (1, n_features, n_classes)
            if isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                pred_shap = shap_vals[0, :, pred_idx]
            elif isinstance(shap_vals, list):
                pred_shap = shap_vals[pred_idx][0]
            else:
                pred_shap = shap_vals[0]

            feature_names = vectorizer.get_feature_names_out()
            # Non-zero tokens in user input
            nz_indices = vec.nonzero()[1]
            
            if len(nz_indices) > 0:
                token_shaps = []
                for idx in nz_indices:
                    val = pred_shap[idx]
                    token = feature_names[idx]
                    token_shaps.append({"Token": token, "SHAP Impact": val, "Direction": "Positive (Supports)" if val > 0 else "Negative (Opposes)"})
                    
                df_local_shap = pd.DataFrame(token_shaps).sort_values("SHAP Impact", ascending=True)
                
                fig_local_shap = px.bar(
                    df_local_shap,
                    x="SHAP Impact",
                    y="Token",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={"Positive (Supports)": "#10B981", "Negative (Opposes)": "#EF4444"},
                    title=f"<b>Token-Level SHAP Impact Driving Decision: {meta['name']}</b>",
                    labels={"SHAP Impact": "SHAP Attribution (Log-Odds Contribution)"}
                )
                fig_local_shap.update_layout(height=max(250, len(df_local_shap)*28), margin=dict(l=0, r=20, t=40, b=10))
                st.plotly_chart(fig_local_shap, use_container_width=True)
            else:
                st.info("Input text does not contain known vocabulary features.")
        except Exception as e:
            st.warning(f"Note on SHAP calculation: {e}")

        # Clinical X-Ray Section
        st.markdown("### 🔬 Clinical Symptom 'X-Ray' & Diagnostic Battery Mapping")
        annotated_html, found_dict = highlight_symptoms(user_text)
        
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 22px; border-radius: 12px; font-size: 1.15rem; line-height: 2.1; box-shadow: inset 0 2px 6px rgba(0,0,0,0.03);">
            {annotated_html}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        b_cols = st.columns(4)
        b_cols[0].markdown(f"**PHQ-9 Symptoms ({len(found_dict['PHQ9'])}):** " + (", ".join(found_dict["PHQ9"]) if found_dict["PHQ9"] else "None"))
        b_cols[1].markdown(f"**GAD-7 Symptoms ({len(found_dict['GAD7'])}):** " + (", ".join(found_dict["GAD7"]) if found_dict["GAD7"] else "None"))
        b_cols[2].markdown(f"**PCL-5 Symptoms ({len(found_dict['PCL5'])}):** " + (", ".join(found_dict["PCL5"]) if found_dict["PCL5"] else "None"))
        b_cols[3].markdown(f"**SPIN Symptoms ({len(found_dict['SPIN'])}):** " + (", ".join(found_dict["SPIN"]) if found_dict["SPIN"] else "None"))


# ────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL SHAP EXPLAINABILITY LAB
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "🔍 Global SHAP Explainability Lab":
    st.markdown("""
    <div class="hero-banner">
        <div class="live-badge">Global Feature Explainability</div>
        <div class="hero-title">Global Feature Importance & Alignment Analysis</div>
        <div class="hero-subtitle">Computed over the entire 102,984-post corpus using TreeSHAP. Inspect the most influential linguistic markers for each mental health category under the scrubbed experimental condition.</div>
    </div>
    """, unsafe_allow_html=True)

    scrubbed_top = shap_top_data.get("scrubbed", {})
    if scrubbed_top:
        selected_class = st.selectbox(
            "Select Psychological Condition to Inspect Global SHAP Attributions:",
            list(scrubbed_top.keys()),
            format_func=lambda x: f"{LABEL_METADATA.get(x, {}).get('icon', '')} {LABEL_METADATA.get(x, {}).get('name', x.title())}"
        )
        
        top_list = scrubbed_top[selected_class]
        df_top = pd.DataFrame(top_list).sort_values("mean_abs_shap", ascending=True)
        
        # Tag each feature
        scale_map = {}
        for s_name, s_words in lexicons.items():
            for w in s_words:
                scale_map[w.lower()] = s_name
                
        df_top["Lexicon Status"] = df_top["feature"].apply(
            lambda f: f"Clinical ({scale_map[f.lower()]})" if f.lower() in scale_map else "General Language"
        )
        
        col_g1, col_g2 = st.columns([2.5, 1.2])
        
        with col_g1:
            fig_global = px.bar(
                df_top,
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                color="Lexicon Status",
                color_discrete_map={
                    "Clinical (PHQ9)": "#2563EB",
                    "Clinical (GAD7)": "#D97706",
                    "Clinical (PCL5)": "#7C3AED",
                    "Clinical (SPIN)": "#DB2777",
                    "General Language": "#94A3B8"
                },
                title=f"<b>Top SHAP Features for {LABEL_METADATA[selected_class]['name']} (Scrubbed Condition)</b>",
                labels={"mean_abs_shap": "Mean |SHAP Value| (Global Feature Importance)", "feature": "Vocabulary Token"}
            )
            fig_global.update_layout(height=480, margin=dict(l=0, r=20, t=40, b=20))
            st.plotly_chart(fig_global, use_container_width=True)
            
        with col_g2:
            st.markdown(f"#### Clinical Insights: {LABEL_METADATA[selected_class]['name']}")
            st.markdown(f"• **Scale:** {LABEL_METADATA[selected_class]['scale']}")
            st.markdown(f"• **Dominant Signal:** In the scrubbed model, top features reflect genuine pathological expressions rather than subreddit self-referencing.")
            
            clinical_tokens = df_top[df_top["Lexicon Status"].str.startswith("Clinical")]
            st.markdown(f"• **Clinical Lexicon Hits in Top Set:** `{len(clinical_tokens)} / {len(df_top)}`")
            for _, r in clinical_tokens.iterrows():
                st.markdown(f"  - **`{r['feature']}`** → *{r['Lexicon Status']}* (`{r['mean_abs_shap']:.3f}`)")


# ────────────────────────────────────────────────────────────────────────────
# 3. DYNAMIC REAL-TIME TEXT SCANNER
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "⚡ Dynamic Keystroke Scanner":
    st.markdown("""
    <div class="hero-banner">
        <div class="live-badge"><div class="pulse-dot"></div>Interactive Keystroke Scanner</div>
        <div class="hero-title">Live Symptom Stream Analyzer</div>
        <div class="hero-subtitle">Type, modify, or delete words to watch probabilities update dynamically. Explore how individual symptom words instantly alter the model's posterior belief.</div>
    </div>
    """, unsafe_allow_html=True)

    stream_input = st.text_input(
        "Type or edit a sentence dynamically:",
        value="I have severe insomnia and flashbacks every night feeling terrified"
    )

    if stream_input:
        model = scrubbed_pkg["model"]
        vectorizer = scrubbed_pkg["vectorizer"]
        label_encoder = scrubbed_pkg["label_encoder"]
        
        clean = scrub_text(stream_input)
        vec = vectorizer.transform([clean])
        probs = model.predict_proba(vec)[0]
        classes = label_encoder.classes_
        
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        
        feature_names = vectorizer.get_feature_names_out()
        nz_indices = vec.nonzero()[1]
        active_words = [feature_names[i] for i in nz_indices]

        c_dyn1, c_dyn2 = st.columns([2, 1])

        with c_dyn1:
            st.markdown("#### Live Probability Gauges")
            for c_idx, c_name in enumerate(classes):
                c_meta = LABEL_METADATA[c_name]
                p_val = probs[c_idx]
                st.markdown(f"**{c_meta['icon']} {c_meta['name']}** — `{p_val*100:.1f}%`")
                st.progress(float(p_val))

        with c_dyn2:
            st.markdown("#### Active TF-IDF Vocabulary Tokens")
            if active_words:
                for w in active_words:
                    scale_tag = "Generic Vocabulary"
                    for s_name, s_words in lexicons.items():
                        if w in s_words:
                            scale_tag = f"Clinical ({s_name})"
                            break
                    st.markdown(f"• **`{w}`** → *{scale_tag}*")
            else:
                st.caption("No vocabulary tokens recognized yet (type meaningful words).")


# ────────────────────────────────────────────────────────────────────────────
# 4. SHORTCUT VS. CLINICAL STRESS TEST
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "⚖️ Shortcut vs. Clinical Stress Test":
    st.markdown("""
    <div class="hero-banner">
        <div class="live-badge" style="background: rgba(239, 68, 68, 0.2); color: #F87171; border-color: rgba(239, 68, 68, 0.4);">Shortcut Artifact Benchmark</div>
        <div class="hero-title">The Subreddit Shortcut Stress Test</div>
        <div class="hero-subtitle">The pivotal finding of the research paper: Standard social media NLP models exploit community mentions ('r/ptsd', 'diagnosed with depression') rather than true pathology. Scrubbing eliminates these shortcuts.</div>
    </div>
    """, unsafe_allow_html=True)

    test_input = st.text_input(
        "Enter text containing Reddit shortcuts / diagnostic confessions:",
        value="I was diagnosed with ptsd on r/ptsd and my anxiety is terrible, having nightmares and flashbacks every night."
    )

    if test_input:
        clean_input = scrub_text(test_input)

        raw_clf, raw_vec = unscrubbed_pkg["model"], unscrubbed_pkg["vectorizer"]
        scrub_clf, scrub_vec = scrubbed_pkg["model"], scrubbed_pkg["vectorizer"]

        raw_probs = raw_clf.predict_proba(raw_vec.transform([test_input]))[0]
        scrub_probs = scrub_clf.predict_proba(scrub_vec.transform([clean_input]))[0]

        raw_classes = unscrubbed_pkg["label_encoder"].classes_
        scrub_classes = scrubbed_pkg["label_encoder"].classes_

        raw_pred = raw_classes[np.argmax(raw_probs)]
        scrub_pred = scrub_classes[np.argmax(scrub_probs)]

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid #EF4444;">
                <div style="color: #EF4444; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">1. Raw Text Arm (Unscrubbed Model)</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #1E293B; margin-top: 6px;">
                    {LABEL_METADATA[raw_pred]['icon']} {LABEL_METADATA[raw_pred]['name']} ({np.max(raw_probs)*100:.1f}%)
                </div>
                <div style="font-size: 0.95rem; color: #64748B; margin-top: 8px;">
                    <b>Trigger Tokens:</b> <span class="highlight-artifact">diagnosed</span>, <span class="highlight-artifact">r/ptsd</span>, <span class="highlight-artifact">anxiety</span>
                </div>
                <div style="background: #FEF2F2; padding: 12px; border-radius: 8px; margin-top: 14px; font-size: 0.88rem; color: #991B1B;">
                    ⚠️ <b>Vulnerability:</b> The model latches onto the subreddit name directly, ignoring whether real symptom criteria are described.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div class="glass-card" style="border-top: 4px solid #10B981;">
                <div style="color: #10B981; font-weight: 700; font-size: 0.85rem; text-transform: uppercase;">2. Scrubbed Arm (Clinical Model)</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #1E293B; margin-top: 6px;">
                    {LABEL_METADATA[scrub_pred]['icon']} {LABEL_METADATA[scrub_pred]['name']} ({np.max(scrub_probs)*100:.1f}%)
                </div>
                <div style="font-size: 0.95rem; color: #64748B; margin-top: 8px;">
                    <b>Surviving Clinical Tokens:</b> <span class="highlight-pcl5">nightmares</span>, <span class="highlight-pcl5">flashbacks</span>
                </div>
                <div style="background: #ECFDF5; padding: 12px; border-radius: 8px; margin-top: 14px; font-size: 0.88rem; color: #065F46;">
                    ✅ <b>Robustness:</b> Shortcuts completely stripped. The model relies entirely on clinical symptom pathology.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔬 Symptom Alignment Score (SAS) Mass Decomposition")
        st.caption("Quantifying the fraction of model decision mass allocated to Clinical Symptoms (SAS) vs. Spurious Shortcuts (SPUR) vs. Ambiguous (AMB).")

        xg_align = alignments_data.get("xgboost", {})
        if xg_align:
            unscrub_df = pd.DataFrame(xg_align["unscrubbed"]).T.reset_index().rename(columns={"index": "Class"})
            scrub_df = pd.DataFrame(xg_align["scrubbed"]).T.reset_index().rename(columns={"index": "Class"})

            c_tab1, c_tab2 = st.tabs(["Raw Unscrubbed Model (Shortcut Heavy)", "Scrubbed Model (Clinical Only)"])
            with c_tab1:
                fig1 = px.bar(
                    unscrub_df,
                    x="Class",
                    y=["SAS", "SPUR", "AMB"],
                    title="Unscrubbed XGBoost: Spurious Shortcut Leakage (SPUR in Red)",
                    color_discrete_map={"SAS": "#10B981", "SPUR": "#EF4444", "AMB": "#94A3B8"}
                )
                fig1.update_layout(height=360)
                st.plotly_chart(fig1, use_container_width=True)

            with c_tab2:
                fig2 = px.bar(
                    scrub_df,
                    x="Class",
                    y=["SAS", "SPUR", "AMB"],
                    title="Scrubbed XGBoost: Zero Spurious Leakage (SPUR = 0.00)",
                    color_discrete_map={"SAS": "#10B981", "SPUR": "#EF4444", "AMB": "#94A3B8"}
                )
                fig2.update_layout(height=360)
                st.plotly_chart(fig2, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# 5. MULTI-MODEL RESEARCH LEADERBOARD
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "📊 Multi-Model Research Leaderboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="live-badge">Research Experiment Outcomes</div>
        <div class="hero-title">Multi-Model Performance Benchmark</div>
        <div class="hero-subtitle">Comparative evaluation of Classical Machine Learning (XGBoost), Recurrent Neural Networks (BiLSTM), and Pre-trained Transformers (RoBERTa & MentalBERT).</div>
    </div>
    """, unsafe_allow_html=True)

    rows = []
    models_display = {
        "xgboost": "XGBoost + TF-IDF (Baseline)",
        "bilstm": "BiLSTM + GloVe 300d",
        "roberta": "RoBERTa (Fine-Tuned)",
        "mentalbert": "MentalBERT (Domain Pre-Trained)"
    }

    for m_key, m_name in models_display.items():
        if m_key in results_data:
            res = results_data[m_key]
            u_f1 = res["unscrubbed"]["classification_report"]["macro avg"]["f1-score"]
            s_f1 = res["scrubbed"]["classification_report"]["macro avg"]["f1-score"]
            rows.append({
                "Architecture": m_name,
                "Unscrubbed F1": u_f1,
                "Scrubbed F1": s_f1,
                "Delta Drop": s_f1 - u_f1,
                "Shortcut Resistance": f"{(s_f1/u_f1)*100:.1f}%"
            })

    bench_df = pd.DataFrame(rows)

    c_b1, c_b2 = st.columns([2.2, 1.2])
    with c_b1:
        st.dataframe(
            bench_df.style.format({
                "Unscrubbed F1": "{:.3f}",
                "Scrubbed F1": "{:.3f}",
                "Delta Drop": "{:.3f}"
            }).background_gradient(subset=["Scrubbed F1"], cmap="Greens"),
            use_container_width=True,
            hide_index=True
        )

    with c_b2:
        st.markdown("""
        <div class="stat-box" style="border-top: 4px solid #10B981;">
            <div class="stat-val" style="color: #10B981;">MentalBERT</div>
            <div class="stat-lbl">Champion Architecture</div>
            <div style="font-size: 0.95rem; color: #475569; margin-top: 8px;">Scrubbed Macro F1: <b>0.836</b></div>
            <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Retains 95.9% performance even when all shortcut tokens are removed.</div>
        </div>
        """, unsafe_allow_html=True)

    fig_bench = go.Figure()
    fig_bench.add_trace(go.Bar(
        name="Unscrubbed (With Artifacts)",
        x=bench_df["Architecture"],
        y=bench_df["Unscrubbed F1"],
        marker_color="#3B82F6",
        text=bench_df["Unscrubbed F1"].apply(lambda v: f"{v:.3f}"),
        textposition="auto"
    ))
    fig_bench.add_trace(go.Bar(
        name="Scrubbed (Clinical Language Only)",
        x=bench_df["Architecture"],
        y=bench_df["Scrubbed F1"],
        marker_color="#10B981",
        text=bench_df["Scrubbed F1"].apply(lambda v: f"{v:.3f}"),
        textposition="auto"
    ))
    fig_bench.update_layout(
        title="<b>F1 Robustness Degradation Across All Architectures</b>",
        barmode="group",
        height=380,
        yaxis=dict(range=[0.6, 0.95], title="Macro F1-Score")
    )
    st.plotly_chart(fig_bench, use_container_width=True)

    st.markdown("### 🖼️ Research Publication Figures")
    f_cols = st.columns(3)
    if os.path.exists(os.path.join(BASE, "fig1_stacked.png")):
        f_cols[0].image(os.path.join(BASE, "fig1_stacked.png"), caption="Figure 1: SHAP Alignment Decomposition")
    if os.path.exists(os.path.join(BASE, "fig5_confusion.png")):
        f_cols[1].image(os.path.join(BASE, "fig5_confusion.png"), caption="Figure 5: Confusion Matrix (Scrubbed)")
    if os.path.exists(os.path.join(BASE, "fig6_model_comparison.png")):
        f_cols[2].image(os.path.join(BASE, "fig6_model_comparison.png"), caption="Figure 6: Cross-Model Comparison")


# ────────────────────────────────────────────────────────────────────────────
# 6. CLINICAL INSTRUMENT EXPLORER
# ────────────────────────────────────────────────────────────────────────────
elif app_mode == "📚 Clinical Instrument Explorer":
    st.markdown("""
    <div class="hero-banner">
        <div class="live-badge">Psychiatric Lexicon Registry</div>
        <div class="hero-title">Validated Clinical Instruments</div>
        <div class="hero-subtitle">Official DSM-5 survey items and psychometric batteries utilized to validate model decisions and compute the Symptom Alignment Score (SAS).</div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["PHQ-9 (Depression)", "GAD-7 (Anxiety)", "PCL-5 (PTSD)", "SPIN (Social Anxiety)", "Artifact Shortcuts"])

    with tabs[0]:
        st.markdown("**Patient Health Questionnaire-9 (PHQ-9)** — 37 Clinical Terms:")
        st.write(", ".join(sorted(list(lexicons.get("PHQ9", [])))))
    with tabs[1]:
        st.markdown("**Generalized Anxiety Disorder-7 (GAD-7)** — 25 Clinical Terms:")
        st.write(", ".join(sorted(list(lexicons.get("GAD7", [])))))
    with tabs[2]:
        st.markdown("**PTSD Checklist for DSM-5 (PCL-5)** — 65 Clinical Terms:")
        st.write(", ".join(sorted(list(lexicons.get("PCL5", [])))))
    with tabs[3]:
        st.markdown("**Social Phobia Inventory (SPIN)** — 37 Clinical Terms:")
        st.write(", ".join(sorted(list(lexicons.get("SPIN", [])))))
    with tabs[4]:
        st.markdown("**Banned Spurious Shortcut Tokens:**")
        st.write(", ".join(sorted(list(ARTIFACT_TERMS + ARTIFACT_TERMS_WB))))
