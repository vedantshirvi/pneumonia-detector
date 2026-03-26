import streamlit as st
from fastai.vision.all import *
from PIL import Image
import plotly.graph_objects as go
import numpy as np
import time
 
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');
 
:root {
    --bg: #0e0e0e;
    --surface: #161616;
    --surface2: #1c1c1c;
    --border: rgba(255,255,255,0.07);
    --border-hover: rgba(255,255,255,0.14);
    --text: #e8e8e8;
    --text-muted: rgba(232,232,232,0.45);
    --accent: #c8a96e;
    --accent2: #7c6fcd;
    --red: #e05c5c;
    --green: #5cb87a;
}
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
 
.stApp { background: var(--bg); }
 
#particle-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
}
 
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 300px;
    background: radial-gradient(ellipse at 50% 0%, rgba(124,111,205,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
 
.hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.8rem;
}
.hero-rule {
    width: 40px; height: 2px;
    background: var(--accent);
    margin: 1.2rem auto 0;
    border-radius: 2px;
}
 
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin: 2rem 0;
}
.stat-card {
    background: var(--surface);
    padding: 1.4rem 1.2rem;
    transition: background 0.2s;
}
.stat-card:hover { background: var(--surface2); }
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.stat-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    border: 1px solid rgba(200,169,110,0.25);
    border-radius: 4px;
    padding: 1px 6px;
    margin-top: 0.4rem;
}
 
.sec-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
 
.result-wrap {
    border-radius: 10px;
    padding: 1.4rem;
    text-align: center;
    animation: fadeUp 0.4s ease-out;
}
.result-normal  { background: rgba(92,184,122,0.07);  border: 1px solid rgba(92,184,122,0.2); }
.result-pneumo  { background: rgba(224,92,92,0.07);   border: 1px solid rgba(224,92,92,0.2); }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(8px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
}
.result-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}
 
.icard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.25s, background 0.25s;
}
.icard:hover { border-color: var(--border-hover); background: var(--surface2); }
.icard-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.icard-text {
    font-size: 0.83rem;
    color: var(--text-muted);
    line-height: 1.65;
}
 
.badge-row { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem; }
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    color: rgba(232,232,232,0.6);
}
.badge-r { background:rgba(224,92,92,0.08);   border-color:rgba(224,92,92,0.2);   color:#e08888; }
.badge-g { background:rgba(92,184,122,0.08);  border-color:rgba(92,184,122,0.2);  color:#88c89a; }
.badge-a { background:rgba(200,169,110,0.08); border-color:rgba(200,169,110,0.2); color:#c8a96e; }
 
section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
 
.stFileUploader > div {
    background: var(--surface2) !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
div[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
 
.disclaimer {
    background: rgba(200,169,110,0.05);
    border: 1px solid rgba(200,169,110,0.15);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: rgba(200,169,110,0.55);
    letter-spacing: 0.03em;
    margin-top: 1rem;
}
 
.pulse-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #5cb87a;
    box-shadow: 0 0 0 0 rgba(92,184,122,0.4);
    animation: pulse 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(92,184,122,0.4); }
    70%  { box-shadow: 0 0 0 6px rgba(92,184,122,0); }
    100% { box-shadow: 0 0 0 0 rgba(92,184,122,0); }
}
 
.awaiting {
    height: 280px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border: 1px dashed rgba(255,255,255,0.07);
    border-radius: 10px;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-align: center;
    gap: 0.6rem;
}
</style>
 
<canvas id="particle-canvas"></canvas>
<script>
(function() {
  const canvas = document.getElementById('particle-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, dots, mouse = {x: -9999, y: -9999};
  const SPACING = 28, RADIUS = 1.1, INFLUENCE = 110;
 
  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
    buildDots();
  }
 
  function buildDots() {
    dots = [];
    const cols = Math.ceil(W / SPACING) + 1;
    const rows = Math.ceil(H / SPACING) + 1;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        dots.push({ x: c*SPACING, y: r*SPACING, ox: c*SPACING, oy: r*SPACING, vx:0, vy:0 });
  }
 
  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const d of dots) {
      const dx = mouse.x - d.x, dy = mouse.y - d.y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < INFLUENCE) {
        const force = (1 - dist/INFLUENCE) * 5;
        d.vx -= (dx/dist)*force;
        d.vy -= (dy/dist)*force;
      }
      d.vx += (d.ox - d.x)*0.12;
      d.vy += (d.oy - d.y)*0.12;
      d.vx *= 0.72; d.vy *= 0.72;
      d.x += d.vx;  d.y += d.vy;
      const nearness = Math.max(0, 1 - dist/INFLUENCE);
      const alpha = 0.06 + nearness * 0.3;
      const r = RADIUS + nearness * 1.4;
      ctx.beginPath();
      ctx.arc(d.x, d.y, r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }
 
  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  resize(); draw();
})();
</script>
""", unsafe_allow_html=True)
 
# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                    color:#e8e8e8;letter-spacing:-0.02em;white-space:nowrap;'>
            Pneumo<span style='color:#c8a96e;'>Scan</span>
        </div>
        <div style='font-family:DM Mono,monospace;font-size:0.65rem;
                    color:rgba(232,232,232,0.3);letter-spacing:0.2em;margin-top:0.3rem;'>
            AI DIAGNOSTIC · v2.1
        </div>
    </div>
    <div style='height:1px;background:rgba(255,255,255,0.06);margin-bottom:1.5rem;'></div>
 
    <div class='sec-header'>About Pneumonia</div>
                
    <div class='icard'>
        <div class='icard-text'>
            Pneumonia inflames the air sacs in one or both lungs, which may fill with fluid or pus.
            Caused by bacteria, viruses, or fungi — ranging from mild to life-threatening.
        </div>
    </div>
    <div class='icard'>
        <div class='icard-title'>Global Impact</div>
        <div class='icard-text'>
            Leading infectious killer of children under 5 —
            roughly <span style='color:#e8e8e8;'>700,000</span> deaths per year worldwide.
        </div>
    </div>
 
    <div class='disclaimer'>
        ⚠ EDUCATIONAL USE ONLY<br>
        Not a substitute for clinical diagnosis. Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)
 
# ── HERO ─────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-eyebrow'>Deep Learning · Pulmonary Analysis · ResNet-34</div>
    <div class='hero-title'>Pneumo<span>Scan</span> AI</div>
    <div class='hero-subtitle'>Upload a chest X-Ray and receive an AI-powered diagnostic prediction in seconds.</div>
    <div class='hero-rule'></div>
</div>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class='stat-grid'>
    <div class='stat-card'>
        <div class='stat-value'>81%</div>
        <div class='stat-label'>Validation Accuracy</div>
        <div class='stat-tag'>ResNet-34</div>
    </div>
    <div class='stat-card'>
        <div class='stat-value'>5,216</div>
        <div class='stat-label'>Training Images</div>
        <div class='stat-tag'>Kaggle Dataset</div>
    </div>
    <div class='stat-card'>
        <div class='stat-value'>~2s</div>
        <div class='stat-label'>Inference Time</div>
        <div class='stat-tag'>CPU</div>
    </div>
    <div class='stat-card'>
        <div class='stat-value'>2</div>
        <div class='stat-label'>Output Classes</div>
        <div class='stat-tag'>Binary</div>
    </div>
</div>
""", unsafe_allow_html=True)
 
# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬  Diagnostic Scanner", "📊  Model Analytics", "📚  Medical Reference"])
 
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")
 
    with col_upload:
        st.markdown("<div class='sec-header'>Upload X-Ray</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='icard-text' style='margin-bottom:1rem;font-size:0.83rem;'>
            Accepts JPG or PNG chest X-Ray images. Classifies into
            <span style='color:#e8e8e8;'>Normal</span> or
            <span style='color:#e8e8e8;'>Pneumonia</span> using a fine-tuned ResNet-34.
        </div>
        """, unsafe_allow_html=True)
 
        uploaded_file = st.file_uploader("Drop X-Ray here", type=["jpg","jpeg","png"], label_visibility="collapsed")
 
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
            st.markdown("""
            <div style='font-family:DM Mono,monospace;font-size:0.68rem;
                        color:rgba(232,232,232,0.3);margin-top:0.4rem;letter-spacing:0.05em;'>
                IMAGE LOADED · READY FOR ANALYSIS
            </div>
            """, unsafe_allow_html=True)
 
    with col_result:
        st.markdown("<div class='sec-header'>Analysis Result</div>", unsafe_allow_html=True)
 
        if uploaded_file is None:
            st.markdown("""
            <div class='awaiting'>
                <div style='font-size:2.5rem;opacity:0.15;'>🫁</div>
                <div>AWAITING IMAGE</div>
                <div style='font-size:0.65rem;opacity:0.5;'>Upload an X-Ray to begin analysis</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            @st.cache_resource
            def load_model():
                return load_learner('pneumonia_model.pkl')
            learn = load_model()
 
            with st.spinner("Running inference..."):
                time.sleep(0.4)
                pred, idx, probs = learn.predict(image)
                normal_prob = float(probs[0]) * 100
                pneumo_prob = float(probs[1]) * 100
 
            if pred == "NORMAL":
                st.markdown(f"""
                <div class='result-wrap result-normal'>
                    <div class='result-label' style='color:#5cb87a;'>✓ Normal</div>
                    <div class='result-sub'>No pneumonia detected · {normal_prob:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-wrap result-pneumo'>
                    <div class='result-label' style='color:#e05c5c;'>⚠ Pneumonia</div>
                    <div class='result-sub'>Pneumonia detected · {pneumo_prob:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=pneumo_prob,
                title={'text':"Pneumonia Probability",'font':{'color':'#666','size':12,'family':'DM Sans'}},
                number={'suffix':'%','font':{'color':'#e8e8e8','size':26,'family':'Syne'}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#333','tickfont':{'color':'#555','size':10}},
                    'bar':{'color':'#e05c5c' if pneumo_prob>50 else '#5cb87a','thickness':0.25},
                    'bgcolor':'rgba(22,22,22,0.8)',
                    'bordercolor':'rgba(255,255,255,0.06)',
                    'steps':[
                        {'range':[0,40],'color':'rgba(92,184,122,0.05)'},
                        {'range':[40,70],'color':'rgba(200,169,110,0.05)'},
                        {'range':[70,100],'color':'rgba(224,92,92,0.05)'}
                    ],
                    'threshold':{'line':{'color':'rgba(255,255,255,0.15)','width':2},'thickness':0.75,'value':50}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=210, margin=dict(l=20,r=20,t=40,b=10), font={'family':'DM Sans'}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
 
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=['Normal','Pneumonia'], y=[normal_prob, pneumo_prob],
                marker_color=['rgba(92,184,122,0.7)','rgba(224,92,92,0.7)'],
                marker_line_width=0,
                text=[f'{normal_prob:.1f}%', f'{pneumo_prob:.1f}%'],
                textposition='auto',
                textfont={'color':'#e8e8e8','size':12,'family':'DM Mono'}
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,22,22,0.5)',
                height=180, margin=dict(l=10,r=10,t=10,b=30),
                xaxis=dict(gridcolor='rgba(255,255,255,0.04)',tickfont=dict(color='#666',family='DM Sans')),
                yaxis=dict(gridcolor='rgba(255,255,255,0.04)',tickfont=dict(color='#666',family='DM Sans'),
                           range=[0,100],ticksuffix='%'),
                font={'family':'DM Sans'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
 
            m1, m2, m3 = st.columns(3)
            m1.metric("Normal",    f"{normal_prob:.1f}%")
            m2.metric("Pneumonia", f"{pneumo_prob:.1f}%")
            m3.metric("Prediction", pred)
 
with tab2:
    st.markdown("<div class='sec-header'>Training Metrics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
 
    epochs     = [0, 1]
    train_loss = [0.455, 0.197]
    valid_loss = [0.747, 0.721]
    accuracy   = [75.0,  81.25]
 
    def base_layout(title, h=260):
        return dict(
            title=dict(text=title, font=dict(color='#666',size=12,family='DM Sans')),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,22,22,0.5)',
            font={'color':'#666','family':'DM Sans'}, height=h,
            margin=dict(l=10,r=10,t=40,b=30),
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#666')),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)',tickfont=dict(color='#555'),title='Epoch'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)',tickfont=dict(color='#555'))
        )
 
    with c1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs,y=train_loss,name='Train Loss',
            line=dict(color='#c8a96e',width=2),
            fill='tozeroy',fillcolor='rgba(200,169,110,0.04)',
            mode='lines+markers',marker=dict(size=6,color='#c8a96e')))
        fig_loss.add_trace(go.Scatter(x=epochs,y=valid_loss,name='Valid Loss',
            line=dict(color='rgba(255,255,255,0.3)',width=2,dash='dot'),
            mode='lines+markers',marker=dict(size=6,color='rgba(255,255,255,0.3)')))
        fig_loss.update_layout(**base_layout('Loss per Epoch'))
        st.plotly_chart(fig_loss, use_container_width=True)
 
    with c2:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs,y=accuracy,name='Accuracy',
            line=dict(color='#5cb87a',width=2),
            fill='tozeroy',fillcolor='rgba(92,184,122,0.04)',
            mode='lines+markers',marker=dict(size=6,color='#5cb87a')))
        fig_acc.add_hline(y=80,line_dash='dot',line_color='rgba(255,255,255,0.08)',
                          annotation_text='80% target',
                          annotation_font_color='rgba(255,255,255,0.25)',
                          annotation_font_size=10)
        lay = base_layout('Accuracy per Epoch')
        lay['yaxis']['ticksuffix'] = '%'
        lay['yaxis']['range'] = [0,100]
        fig_acc.update_layout(**lay)
        st.plotly_chart(fig_acc, use_container_width=True)
 
    st.markdown("<div class='sec-header'>Dataset Composition</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
 
    with c3:
        fig_pie = go.Figure(go.Pie(
            labels=['Pneumonia','Normal'], values=[3875,1341], hole=0.55,
            marker=dict(colors=['rgba(224,92,92,0.75)','rgba(92,184,122,0.75)'],
                        line=dict(color='rgba(0,0,0,0)',width=0)),
            textfont=dict(color='#888',family='DM Mono',size=11)
        ))
        fig_pie.update_layout(
            title=dict(text='Training Set',font=dict(color='#666',size=12,family='DM Sans')),
            paper_bgcolor='rgba(0,0,0,0)',height=260,
            margin=dict(l=10,r=10,t=40,b=10),
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#666',family='DM Sans'))
        )
        st.plotly_chart(fig_pie, use_container_width=True)
 
    with c4:
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(name='Normal',x=['Train','Val'],y=[1341,8],
            marker_color='rgba(92,184,122,0.7)',marker_line_width=0))
        fig_stack.add_trace(go.Bar(name='Pneumonia',x=['Train','Val'],y=[3875,8],
            marker_color='rgba(224,92,92,0.7)',marker_line_width=0))
        lay2 = base_layout('Images per Split')
        lay2['barmode'] = 'stack'
        fig_stack.update_layout(**lay2)
        st.plotly_chart(fig_stack, use_container_width=True)
 
    st.markdown("<div class='sec-header'>Architecture Summary</div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    for col, (title, text) in zip([a1,a2,a3], [
        ("Backbone","ResNet-34 pretrained on ImageNet with 34 layers and residual skip connections to prevent vanishing gradients."),
        ("Transfer Learning","FastAI fine-tuning with discriminative learning rates — frozen base first, then full network unfrozen."),
        ("Output Layer","2-class softmax producing a probability distribution across Normal and Pneumonia categories."),
    ]):
        with col:
            st.markdown(f"""
            <div class='icard'>
                <div class='icard-title'>{title}</div>
                <div class='icard-text'>{text}</div>
            </div>
            """, unsafe_allow_html=True)
 
with tab3:
    st.markdown("<div class='sec-header'>Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='icard'>
        <div class='icard-text' style='font-size:0.87rem;line-height:1.75;'>
            Pneumonia is an acute respiratory infection. The alveoli — tiny air sacs responsible for
            gas exchange — fill with pus or fluid, making breathing painful and reducing oxygen uptake.
            Caused by bacteria, viruses, or fungi, with
            <span style='color:#e8e8e8;'>Streptococcus pneumoniae</span> the most common bacterial agent.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("<div class='sec-header'>Types</div>", unsafe_allow_html=True)
        for title, text in [
            ("Bacterial","Most common and serious. Lobar consolidation on X-Ray — dense white area in one lobe. Responds to antibiotics."),
            ("Viral","Caused by influenza, RSV, or COVID-19. Bilateral interstitial infiltrates on X-Ray. Serious in vulnerable groups."),
            ("Fungal","Affects immunocompromised patients. Caused by Pneumocystis jirovecii, Aspergillus, or Histoplasma."),
        ]:
            st.markdown(f"""
            <div class='icard'>
                <div class='icard-title'>{title}</div>
                <div class='icard-text'>{text}</div>
            </div>
            """, unsafe_allow_html=True)
 
    with t2:
        st.markdown("<div class='sec-header'>Symptoms & Risk Factors</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='icard'>
            <div class='icard-title'>Common Symptoms</div>
            <div class='badge-row'>
                <span class='badge badge-r'>Chest pain</span>
                <span class='badge badge-r'>High fever</span>
                <span class='badge badge-r'>Productive cough</span>
                <span class='badge badge-r'>Dyspnea</span>
                <span class='badge badge-r'>Fatigue</span>
                <span class='badge badge-r'>Chills</span>
            </div>
        </div>
        <div class='icard'>
            <div class='icard-title'>X-Ray Indicators</div>
            <div class='badge-row'>
                <span class='badge'>Consolidation</span>
                <span class='badge'>Ground-glass opacity</span>
                <span class='badge'>Air bronchograms</span>
                <span class='badge'>Pleural effusion</span>
            </div>
        </div>
        <div class='icard'>
            <div class='icard-title'>Risk Factors</div>
            <div class='badge-row'>
                <span class='badge badge-a'>Age &lt;5 or &gt;65</span>
                <span class='badge badge-a'>Smoking</span>
                <span class='badge badge-a'>Immunocompromised</span>
                <span class='badge badge-a'>Chronic illness</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<div class='sec-header'>Global Incidence by Region</div>", unsafe_allow_html=True)
    regions   = ['Sub-Saharan Africa','South Asia','East Asia','Latin America','Europe','North America']
    incidence = [28,22,18,12,8,5]
    mortality = [45,30,15,10,5,3]
 
    fig_g = go.Figure()
    fig_g.add_trace(go.Bar(name='Incidence per 1,000',x=regions,y=incidence,
        marker_color='rgba(124,111,205,0.65)',marker_line_width=0))
    fig_g.add_trace(go.Scatter(name='Mortality Index',x=regions,y=mortality,
        line=dict(color='rgba(224,92,92,0.8)',width=2),
        mode='lines+markers',marker=dict(size=7,color='rgba(224,92,92,0.8)'),yaxis='y2'))
    fig_g.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,22,22,0.5)',
        font={'color':'#666','family':'DM Sans'}, height=300,
        margin=dict(l=10,r=60,t=20,b=80),
        legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(color='#666')),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)',tickfont=dict(color='#555'),tickangle=-20),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)',tickfont=dict(color='#555')),
        yaxis2=dict(overlaying='y',side='right',tickfont=dict(color='rgba(224,92,92,0.5)'),
                    gridcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig_g, use_container_width=True)
 
    st.markdown("""
    <div class='disclaimer'>
        ⚠ DATA SOURCES: WHO Global Health Observatory · CDC · The Lancet ·
        Statistics are approximations for educational purposes only.
    </div>
    """, unsafe_allow_html=True)
 
