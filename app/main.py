"""Interactive recommendation signal room.

Bold visual redesign: high-contrast instrument panel, three skins, live model
inference, custom ranking scoreboard, test-set explorer and comparison matrix.

Run:
    D:\\Envs\\recbole-gnn-py39\\Scripts\\python.exe -m streamlit run app/main.py --server.port 8503
"""
import json
from pathlib import Path
import random
import re
import sys
import time
import html as _html

import altair as alt
import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
CODE = PROJECT / '03_代码与实验'
sys.path.insert(0, str(PROJECT / 'app'))
from predictor import SessionPredictor  # noqa: E402

LABELS = {'SRGNN': 'SR-GNN', 'GCSAN': 'GC-SAN', 'GCEGNN': 'GCE-GNN'}
MODELS = ['SRGNN', 'GCSAN', 'GCEGNN']
PAGES = ['推荐工作台', '测试集检索', '模型对比', '实验说明']

THEMES = {
    'ACID': {
        'BG': '#0A0A0A', 'PANEL': '#111110', 'PANEL2': '#191917', 'INK': '#F3F0E6',
        'MUTED': '#8B887B', 'LINE': '#2B2B27', 'ACCENT': '#D8FF36', 'ACCENT2': '#FF4D00',
        'ACCENT3': '#6C7CFF', 'SIDEBAR': '#050505', 'SIDEBAR_INK': '#F3F0E6',
        'GRID': 'rgba(243,240,230,.055)', 'SHADOW': '#D8FF36',
    },
    'BLUEPRINT': {
        'BG': '#061826', 'PANEL': '#0A2233', 'PANEL2': '#0E2C40', 'INK': '#D8F6FF',
        'MUTED': '#6F9AAE', 'LINE': '#1D3C4E', 'ACCENT': '#5EEBFF', 'ACCENT2': '#FFB000',
        'ACCENT3': '#7CFFB2', 'SIDEBAR': '#03101A', 'SIDEBAR_INK': '#D8F6FF',
        'GRID': 'rgba(94,235,255,.07)', 'SHADOW': '#5EEBFF',
    },
    'PAPER': {
        'BG': '#F2EFE6', 'PANEL': '#FFFDF8', 'PANEL2': '#E8E2D4', 'INK': '#0A0A0A',
        'MUTED': '#6A675C', 'LINE': '#0A0A0A', 'ACCENT': '#FF4D00', 'ACCENT2': '#1F3BFF',
        'ACCENT3': '#0A0A0A', 'SIDEBAR': '#0A0A0A', 'SIDEBAR_INK': '#F2EFE6',
        'GRID': 'rgba(10,10,10,.06)', 'SHADOW': '#0A0A0A',
    },
}

CSS_TEMPLATE = """
<style>
:root {
  --bg: __BG__; --panel: __PANEL__; --panel2: __PANEL2__; --ink: __INK__;
  --muted: __MUTED__; --line: __LINE__; --accent: __ACCENT__; --accent2: __ACCENT2__;
  --accent3: __ACCENT3__; --sidebar: __SIDEBAR__; --sidebar-ink: __SIDEBAR_INK__;
  --grid: __GRID__; --shadow: __SHADOW__;
  --display: "Arial Black", "Microsoft YaHei", "PingFang SC", sans-serif;
  --mono: "Cascadia Code", "Consolas", "SFMono-Regular", monospace;
  --sans: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stAppViewContainer"], .stApp {background: transparent !important;}
body {
  background-color: var(--bg);
  background-image: linear-gradient(var(--grid) 1px, transparent 1px),
                    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 34px 34px;
  color: var(--ink);
  font-family: var(--sans);
}
.block-container {padding: 1.2rem 2.2rem 4rem; max-width: 1500px;}
[data-testid="stSidebar"] {
  background: var(--sidebar); border-right: 1px solid var(--line); min-width: 268px;
}
[data-testid="stSidebar"] * {color: var(--sidebar-ink);}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color: var(--sidebar-ink);}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
  color: var(--ink) !important; background: var(--panel) !important;
  -webkit-text-fill-color: var(--ink) !important; border: 1px solid var(--line) !important;
  border-radius: 0 !important; font-family: var(--mono) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"], [data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--panel) !important; border-radius: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {color: var(--ink) !important;}
[data-testid="stSidebar"] [data-baseweb="select"] svg {fill: var(--ink) !important;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 8px 10px; border-left: 2px solid transparent; margin: 0;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(255,255,255,.06);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  border-left-color: var(--accent); background: rgba(255,255,255,.07);
}
[data-testid="stSidebar"] [role="radiogroup"] label p {font-family: var(--mono); font-size: 11px; letter-spacing: .06em;}
[data-testid="stTextArea"] textarea {
  background: var(--panel) !important; color: var(--ink) !important;
  -webkit-text-fill-color: var(--ink) !important; border: 1px solid var(--line) !important;
  border-radius: 0 !important; font-family: var(--mono) !important; font-size: 13px !important;
  line-height: 1.7 !important; padding: 12px !important;
}
[data-testid="stButton"] button {
  background: var(--accent) !important; color: var(--bg) !important; border: 0 !important;
  border-radius: 0 !important; font-family: var(--mono) !important; font-size: 10.5px !important;
  letter-spacing: .1em !important; text-transform: uppercase !important;
  box-shadow: 4px 4px 0 var(--accent2) !important; padding: 10px 12px !important;
  transition: transform .12s ease, box-shadow .12s ease !important;
}
[data-testid="stButton"] button:hover {
  transform: translate(2px, 2px) !important; box-shadow: 2px 2px 0 var(--accent2) !important;
  color: var(--bg) !important;
}
[data-testid="stButton"] button p {color: var(--bg) !important; font-family: var(--mono) !important;}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background: var(--panel) !important; border: 1px solid var(--line) !important;
  border-radius: 0 !important; color: var(--ink) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * {color: var(--ink) !important;}
[data-testid="stSelectbox"] [data-baseweb="select"] svg {fill: var(--ink) !important;}
[data-testid="stSlider"] [role="slider"] {background: var(--accent) !important;}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="presentation"] {background: var(--accent) !important;}
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important; background: var(--panel) !important;
  border-radius: 0 !important;
}
[data-testid="stExpander"] summary {font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: .08em;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 2px; border-bottom: 1px solid var(--line);}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: var(--panel) !important; border: 1px solid var(--line) !important;
  border-bottom: 0 !important; font-family: var(--mono) !important; font-size: 10px !important;
  padding: 8px 12px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {background: var(--accent) !important; color: var(--bg) !important;}
[data-testid="stAlert"] {
  background: var(--panel2) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 0 !important;
}
[data-testid="stAlert"] * {color: var(--ink) !important;}
[data-testid="stDataFrame"] {border: 1px solid var(--line) !important;}
[data-testid="stSpinner"] * {color: var(--accent) !important;}
hr {border-color: var(--line);}
.ticker {
  overflow: hidden; background: var(--accent); color: var(--bg);
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  margin: 10px 0 28px;
}
.ticker-track {
  display: flex; width: max-content; white-space: nowrap;
  animation: ticker 34s linear infinite; font-family: var(--mono);
  font-size: 10px; letter-spacing: .12em; text-transform: uppercase; padding: 6px 0;
}
.ticker-track span {padding-right: 36px;}
@keyframes ticker {from {transform: translateX(0);} to {transform: translateX(-50%);}}
.hero {margin: 8px 0 6px;}
.hero-top {
  display: flex; justify-content: space-between; gap: 20px;
  font-family: var(--mono); font-size: 9px; letter-spacing: .16em; color: var(--muted);
  border-bottom: 1px solid var(--line); padding-bottom: 10px; text-transform: uppercase;
}
.hero-grid {display: grid; grid-template-columns: 1.35fr .65fr; gap: 30px; align-items: end; padding-top: 18px;}
.hero h1 {
  font-family: var(--display); font-size: clamp(38px, 4.6vw, 64px);
  line-height: .86; letter-spacing: -.055em; margin: 0; font-weight: 900;
  text-transform: uppercase; color: var(--ink); white-space: nowrap;
}
.hero h1 em {font-style: normal; color: var(--accent); -webkit-text-stroke: 2px var(--accent);}
.hero-cn {
  font-family: var(--display); font-size: clamp(20px, 2.6vw, 34px);
  letter-spacing: -.02em; margin-top: 14px; color: var(--ink);
}
.hero-lede {max-width: 700px; color: var(--muted); font-size: 13px; line-height: 1.9; margin: 14px 0 0;}
.hero-panel {
  border: 1px solid var(--line); background: var(--panel); padding: 16px;
  box-shadow: 8px 8px 0 var(--shadow);
}
.hero-stat {border-bottom: 1px solid var(--line); padding: 10px 0 12px;}
.hero-stat:last-child {border-bottom: 0; padding-bottom: 0;}
.hero-stat span, .hero-stat small {display: block; font-family: var(--mono); font-size: 8.5px; letter-spacing: .14em; color: var(--muted);}
.hero-stat b {display: block; font-family: var(--display); font-size: 44px; line-height: 1; letter-spacing: -.04em; color: var(--accent); margin: 4px 0 2px;}
.hero-stat b em {font-size: 18px; color: var(--muted); font-style: normal;}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:has(.panel-head) {
  background: var(--panel); border: 1px solid var(--line);
  box-shadow: 6px 6px 0 var(--shadow); padding: 16px;
}
.panel {
  border: 1px solid var(--line); background: var(--panel); padding: 16px;
  box-shadow: 6px 6px 0 var(--shadow); margin-bottom: 18px;
}
.panel-head {
  display: flex; justify-content: space-between; gap: 16px; align-items: baseline;
  border-bottom: 1px solid var(--line); padding-bottom: 10px; margin-bottom: 14px;
}
.panel-num {font-family: var(--display); font-size: 28px; line-height: 1; color: var(--accent);}
.panel-title {font-family: var(--mono); font-size: 10px; letter-spacing: .14em; text-transform: uppercase; color: var(--ink);}
.panel-meta {font-family: var(--mono); font-size: 8.5px; letter-spacing: .12em; color: var(--muted); text-align: right;}
.micro {font-family: var(--mono); font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted);}
.chips {margin: 12px 0 10px;}
.chip {
  display: inline-block; font-family: var(--mono); font-size: 11px;
  border: 1px solid var(--line); padding: 4px 7px; margin: 0 4px 4px 0;
  background: var(--panel2); color: var(--ink);
}
.chip.target {background: var(--accent); color: var(--bg); border-color: var(--accent);}
.chip.ghost {border-style: dashed; color: var(--muted);}
.readout {border-bottom: 1px solid var(--line); padding: 2px 0 14px; margin-bottom: 8px;}
.readout-label {font-family: var(--mono); font-size: 9px; letter-spacing: .18em; color: var(--muted);}
.readout-value {
  font-family: var(--display); font-size: clamp(64px, 8vw, 110px);
  line-height: .82; letter-spacing: -.06em; color: var(--accent); margin: 6px 0 4px;
}
.readout-value.miss {color: var(--accent2);}
.readout-value.idle {color: var(--muted);}
.readout-meta {font-family: var(--mono); font-size: 9px; letter-spacing: .1em; color: var(--muted); line-height: 1.8;}
.lb {border-top: 1px solid var(--line);}
.lb-row {
  display: grid; grid-template-columns: 54px minmax(90px, .8fr) 2.2fr 68px 88px;
  gap: 10px; align-items: center; padding: 8px 6px; border-bottom: 1px solid var(--line);
}
.lb-row.target {background: var(--accent); color: var(--bg);}
.lb-row.target .lb-id, .lb-row.target .lb-score, .lb-row.target .lb-rank, .lb-row.target .lb-tag {color: var(--bg);}
.lb-rank {font-family: var(--display); font-size: 20px; line-height: 1; color: var(--muted);}
.lb-id {font-family: var(--mono); font-size: 11px; color: var(--ink);}
.lb-bar {height: 9px; background: var(--line); position: relative; overflow: hidden;}
.lb-bar span {display: block; height: 100%; background: var(--accent); width: var(--w, 0%);}
.lb-row.target .lb-bar {background: rgba(0,0,0,.18);}
.lb-row.target .lb-bar span {background: var(--bg);}
.lb-score {font-family: var(--mono); font-size: 10px; text-align: right; color: var(--muted);}
.lb-tag {font-family: var(--mono); font-size: 7.5px; letter-spacing: .08em; text-align: right; color: var(--muted);}
.miss-banner {
  margin-top: 8px; border: 1px solid var(--accent2); color: var(--accent2);
  font-family: var(--mono); font-size: 9.5px; letter-spacing: .1em; padding: 8px 10px;
  text-transform: uppercase;
}
.stat-strip {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 12px;}
.stat-strip div {border: 1px solid var(--line); padding: 8px 10px; background: var(--panel2);}
.stat-strip b {display: block; font-family: var(--display); font-size: 24px; line-height: 1; color: var(--ink);}
.stat-strip span {display: block; font-family: var(--mono); font-size: 8px; letter-spacing: .1em; color: var(--muted); margin-top: 4px;}
.score-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 14px;}
.score-card {border: 1px solid var(--line); background: var(--panel2); padding: 12px;}
.score-card.best {border-color: var(--accent); box-shadow: 4px 4px 0 var(--accent);}
.score-name {font-family: var(--mono); font-size: 9.5px; letter-spacing: .1em; color: var(--muted);}
.score-value {font-family: var(--display); font-size: 42px; line-height: .9; letter-spacing: -.05em; color: var(--ink); margin: 8px 0 2px;}
.score-card.best .score-value {color: var(--accent);}
.score-value em {font-size: 15px; color: var(--muted); font-style: normal;}
.score-row {display: flex; justify-content: space-between; font-family: var(--mono); font-size: 8.5px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 5px; margin-top: 5px;}
.score-time {font-family: var(--mono); font-size: 8px; color: var(--muted); margin-top: 6px;}
.data-table {width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 10px;}
.data-table th {
  text-align: left; font-size: 8px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); border-bottom: 1px solid var(--line); padding: 7px 6px; font-weight: 400;
}
.data-table td {border-bottom: 1px solid var(--line); padding: 7px 6px; color: var(--ink);}
.data-table tr.hit td {background: var(--accent); color: var(--bg);}
.data-table tr.miss td {color: var(--accent2);}
.data-scroll {max-height: 380px; overflow: auto; border: 1px solid var(--line); background: var(--panel);}
.manifesto {font-family: var(--display); font-size: clamp(22px, 3vw, 40px); line-height: 1.2; letter-spacing: -.03em; max-width: 1000px; color: var(--ink);}
.manifesto em {font-style: normal; color: var(--accent);}
.note-block {border-left: 2px solid var(--accent); padding: 2px 0 2px 14px; color: var(--muted); font-size: 12px; line-height: 1.8; margin: 12px 0;}
.awaiting {
  border: 1px dashed var(--line); padding: 44px 18px; text-align: center;
  font-family: var(--mono); font-size: 10px; letter-spacing: .16em; color: var(--muted);
  background: var(--panel2);
}
.graph-wrap {border: 1px solid var(--line); background: var(--panel); overflow: hidden;}
@media (max-width: 900px) {
  .block-container {padding: 1rem 1rem 3rem;}
  .hero-grid {grid-template-columns: 1fr; gap: 20px;}
  .hero h1 {font-size: clamp(34px, 10vw, 56px); white-space: normal;}
  .score-grid {grid-template-columns: 1fr;}
  .lb-row {grid-template-columns: 42px 1fr 1.5fr 60px; }
  .lb-tag {display: none;}
  .stat-strip {grid-template-columns: 1fr 1fr;}
}
@media (prefers-reduced-motion: reduce) {
  .ticker-track {animation: none;}
  * {transition: none !important;}
}
</style>
"""


def theme_css(theme):
    css = CSS_TEMPLATE
    for key, value in theme.items():
        css = css.replace('__' + key + '__', value)
    return css


@st.cache_data(show_spinner=False)
def load_web():
    return json.loads((PROJECT / 'web/data.json').read_text(encoding='utf-8'))


@st.cache_data(show_spinner=False)
def load_predictions(model):
    rows = []
    path = CODE / 'outputs/formal/predictions' / f'{model}.jsonl'
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            row['hit'] = row['target_rank'] > 0
            row['top1'] = row['recommendations'][0]['item'] if row['recommendations'] else ''
            row['top1_score'] = row['recommendations'][0]['score'] if row['recommendations'] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_comparison():
    return pd.read_csv(CODE / 'outputs/formal/comparison.csv')


@st.cache_resource(show_spinner=False)
def get_predictor(model):
    return SessionPredictor(model)


WEB = load_web()
CASES = WEB['cases']
METRICS = WEB['metrics']
CURVES = WEB['curves']
BEST = max(METRICS, key=lambda m: m['recall'])
BEST_METRIC = next(m for m in METRICS if m['key'] == BEST['key'])
SRGNN_METRIC = next(m for m in METRICS if m['key'] == 'SRGNN')
GAIN = (BEST['recall'] - SRGNN_METRIC['recall']) / SRGNN_METRIC['recall'] * 100
TIME_RATIO = BEST['time'] / SRGNN_METRIC['time']

st.set_page_config(page_title='RECOMMENDATION SIGNAL', page_icon='◑', layout='wide',
                   initial_sidebar_state='expanded')

if 'theme' not in st.session_state:
    st.session_state.theme = 'ACID'
skin_q = st.query_params.get('skin', '')
if skin_q in THEMES:
    st.session_state.theme = skin_q
if 'model' not in st.session_state:
    st.session_state.model = 'GCEGNN'
if 'history_text' not in st.session_state:
    first = CASES[0]
    st.session_state.history_text = ', '.join(first['history'])
    st.session_state.target = first['target']
    st.session_state.source_id = first['id']
    st.session_state.source_history = tuple(first['history'])
    st.session_state.preds = None
    st.session_state.pred_history = None
    st.session_state.compare = False
    st.session_state.latency = {}
    st.session_state.editor_version = 0


def esc(value):
    return _html.escape(str(value))


def load_case(case):
    st.session_state.history_text = ', '.join(case['history'])
    st.session_state.target = case['target']
    st.session_state.source_id = case['id']
    st.session_state.source_history = tuple(case['history'])
    st.session_state.preds = None
    st.session_state.pred_history = None
    st.session_state.compare = False
    st.session_state.editor_version += 1
    st.rerun()


def load_by_id(sid):
    sid = str(sid).strip()
    if not sid:
        return
    for model in [st.session_state.model, 'GCEGNN', 'GCSAN', 'SRGNN']:
        df = load_predictions(model)
        row = df[df['session_id'] == sid]
        if not row.empty:
            r = row.iloc[0]
            st.session_state.history_text = ', '.join(r['history'])
            st.session_state.target = r['target']
            st.session_state.source_id = sid
            st.session_state.source_history = tuple(r['history'])
            st.session_state.preds = None
            st.session_state.pred_history = None
            st.session_state.compare = False
            st.session_state.editor_version += 1
            st.rerun()
    st.error(f'TEST SET 中没有会话 {sid}')


def parse_items(text):
    return [x for x in re.split(r'[,\s，]+', text or '') if x]


def run_models(models, items):
    results, latency = {}, {}
    for model in models:
        with st.spinner(f'LOADING / {LABELS[model]} …'):
            predictor = get_predictor(model)
            bad = [x for x in items if not predictor.valid_item(x)]
            if bad:
                raise ValueError(f'商品不在训练词表中：{", ".join(bad[:5])}')
            start = time.perf_counter()
            results[model] = predictor.predict(items)
            latency[model] = (time.perf_counter() - start) * 1000
    st.session_state.preds = results
    st.session_state.latency = latency
    st.session_state.pred_history = tuple(items)
    st.session_state.compare = len(models) > 1
    st.rerun()


def rank_of(recs, target):
    if not target:
        return None
    return next((r['rank'] for r in recs if r['item'] == str(target)), 0)


def ticker_html():
    text = (f'20,000 TEST SESSIONS  /  32,968 ITEMS  /  {LABELS[BEST["key"]]} RECALL@10 '
            f'{BEST["recall"] * 100:.2f}%  /  {LABELS[BEST["key"]]} GAIN vs SR-GNN '
            f'{GAIN:.1f}%  /  TIME {TIME_RATIO:.2f}×  /  SEED 42  /  20 EPOCHS  /  SINGLE RUN  /  ')
    return f'<div class="ticker"><div class="ticker-track"><span>{esc(text)}</span><span>{esc(text)}</span></div></div>'


def hero_html():
    return f"""
<div class="hero">
  <div class="hero-top">
    <span>ALGORITHM ANALYSIS / FINAL PROJECT / 2026</span>
    <span>RUN 2026-11-02 / SEED 42 / CPU</span>
  </div>
  <div class="hero-grid">
    <div>
      <h1>RECOMMENDATION<br><em>SIGNAL</em></h1>
      <div class="hero-cn">电商会话推荐 · 实时实验台</div>
      <p class="hero-lede">把一段浏览序列送进三个图神经网络，实时看下一个点击会落在哪里。不是回放截图：输入序列后直接调用正式 checkpoint 推理。</p>
    </div>
    <div class="hero-panel">
      <div class="hero-stat"><span>BEST RECALL@10</span><b>{BEST['recall'] * 100:.2f}<em>%</em></b><small>{LABELS[BEST['key']]}</small></div>
      <div class="hero-stat"><span>GAIN vs SR-GNN</span><b>{GAIN:.1f}<em>%</em></b><small>TIME {TIME_RATIO:.2f}×</small></div>
      <div class="hero-stat"><span>TEST SESSIONS</span><b>20,000</b><small>32,968 ITEMS / SINGLE SEED</small></div>
    </div>
  </div>
</div>
"""


def panel_head(num, title, meta=''):
    return f'<div class="panel-head"><span class="panel-num">{num}</span><span class="panel-title">{esc(title)}</span><span class="panel-meta">{esc(meta)}</span></div>'


def chips_html(items, target):
    parts = ''.join(f'<span class="chip">{esc(x)}</span>' for x in items)
    if target:
        parts += '<span class="chip ghost">→</span>'
        parts += f'<span class="chip target">{esc(target)}</span>'
    return f'<div class="chips">{parts}</div>'


def readout_html(rank, target, latency, length, model):
    if not target:
        value, cls, meta = '--', 'idle', 'CUSTOM SEQUENCE / NO GROUND TRUTH'
    elif rank == 0:
        value, cls, meta = 'MISS', 'miss', f'TARGET {target} / NOT IN TOP-10'
    else:
        value, cls, meta = f'#{rank:02d}', '', f'TARGET {target} / TRUE NEXT'
    return f"""
<div class="readout">
  <div class="readout-label">HIT RANK / {LABELS[model]}</div>
  <div class="readout-value {cls}">{value}</div>
  <div class="readout-meta">{esc(meta)}<br>LATENCY {latency:.1f} MS / HISTORY {length} CLICKS</div>
</div>"""


def leaderboard_html(recs, target):
    max_score = max((r['score'] for r in recs), default=1.0) or 1.0
    rows = []
    hit_any = False
    for r in recs:
        hit = bool(target and r['item'] == str(target))
        hit_any = hit_any or hit
        width = r['score'] / max_score * 100
        rows.append(
            f'<div class="lb-row {"target" if hit else ""}">'
            f'<div class="lb-rank">{r["rank"]:02d}</div>'
            f'<div class="lb-id">{esc(r["item"])}</div>'
            f'<div class="lb-bar"><span style="width:{width:.1f}%"></span></div>'
            f'<div class="lb-score">{r["score"]:.3f}</div>'
            f'<div class="lb-tag">{"TRUE NEXT" if hit else ""}</div>'
            f'</div>'
        )
    miss = ''
    if target and not hit_any:
        miss = f'<div class="miss-banner">TARGET {esc(target)} / NOT IN TOP-10</div>'
    return '<div class="lb">' + ''.join(rows) + '</div>' + miss


def scorecards_html():
    cards = []
    for m in METRICS:
        best = m['key'] == BEST['key']
        cards.append(f"""
<div class="score-card {'best' if best else ''}">
  <div class="score-name">{LABELS[m['key']]}{' / BEST' if best else ''}</div>
  <div class="score-value">{m['recall'] * 100:.2f}<em>%</em></div>
  <div class="score-row"><span>MRR</span><b>{m['mrr'] * 100:.2f}</b></div>
  <div class="score-row"><span>NDCG</span><b>{m['ndcg'] * 100:.2f}</b></div>
  <div class="score-time">TRAIN+VAL {m['time']:.1f}s / PARAMS {m['params']:,}</div>
</div>""")
    return '<div class="score-grid">' + ''.join(cards) + '</div>'


def table_html(headers, rows, hit_flags=None):
    head = ''.join(f'<th>{esc(h)}</th>' for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ''
        if hit_flags and i < len(hit_flags):
            cls = ' class="hit"' if hit_flags[i] else ''
        body.append('<tr' + cls + '>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>')
    return f'<div class="data-scroll"><table class="data-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def graph_svg(nodes, edges, target, theme):
    import networkx as nx
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n['id'])
    for e in edges:
        G.add_edge(e['source'], e['target'])
    pos = nx.spring_layout(G, seed=42, k=1.2)
    W, H = 900, 370
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    def sx(x):
        return 60 + (x - min(xs)) / max(1e-9, max(xs) - min(xs)) * (W - 120)
    def sy(y):
        return 40 + (y - min(ys)) / max(1e-9, max(ys) - min(ys)) * (H - 80)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="background:{theme["PANEL"]}">']
    parts.append(f'<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="{theme["MUTED"]}"/></marker></defs>')
    for e in edges:
        x1, y1 = sx(pos[e['source']][0]), sy(pos[e['source']][1])
        x2, y2 = sx(pos[e['target']][0]), sy(pos[e['target']][1])
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{theme["MUTED"]}" stroke-width="1.2" marker-end="url(#arrow)" opacity=".7"/>')
    for n in nodes:
        x, y = sx(pos[n['id']][0]), sy(pos[n['id']][1])
        r = 15 + min(n.get('count', 1), 5) * 2.6
        color = theme['ACCENT2'] if target and n['id'] == str(target) else theme['ACCENT']
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="{theme["BG"]}" stroke-width="3"/>')
        parts.append(f'<text x="{x:.1f}" y="{y + r + 13:.1f}" text-anchor="middle" font-family="Consolas,monospace" font-size="10" fill="{theme["INK"]}">{esc(n["id"])}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def curve_chart(theme):
    rows = []
    for key, points in CURVES.items():
        for p in points:
            rows.append({'model': LABELS[key], 'epoch': p['epoch'], 'mrr': p['valid_mrr']})
    df = pd.DataFrame(rows)
    order = [LABELS[k] for k in MODELS]
    return alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=40, filled=True)).encode(
        x=alt.X('epoch:Q', title='EPOCH', axis=alt.Axis(labelColor=theme['MUTED'], titleColor=theme['MUTED'], grid=False)),
        y=alt.Y('mrr:Q', title='VALID MRR@10', scale=alt.Scale(zero=False),
                axis=alt.Axis(labelColor=theme['MUTED'], titleColor=theme['MUTED'], gridColor=theme['LINE'])),
        color=alt.Color('model:N', sort=order,
                        scale=alt.Scale(domain=order, range=[theme['ACCENT'], theme['ACCENT2'], theme['INK']])),
        tooltip=['model', 'epoch', alt.Tooltip('mrr:Q', format='.4f')],
    ).properties(height=320, background='transparent').configure_view(stroke=None)


# ── theme first, then CSS ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="micro" style="color:#8B887B;">SIGNAL ROOM / 2026</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--display);font-size:42px;line-height:.9;letter-spacing:-.04em;margin:2px 0 4px;">R/S</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:var(--mono);font-size:8px;letter-spacing:.16em;color:#8B887B;margin-bottom:14px;">RECOMMENDATION<br>SIGNAL ROOM</div>', unsafe_allow_html=True)
    theme_name = st.selectbox('SKIN', list(THEMES.keys()), index=list(THEMES).index(st.session_state.theme))
    st.session_state.theme = theme_name

THEME = THEMES[st.session_state.theme]
st.markdown(theme_css(THEME), unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="height:1px;background:#2B2B27;margin:12px 0;"></div>', unsafe_allow_html=True)
    page = st.radio('NAV', PAGES, label_visibility='collapsed')
    st.markdown('<div style="height:1px;background:#2B2B27;margin:12px 0;"></div>', unsafe_allow_html=True)
    model = st.selectbox('MODEL', MODELS, index=MODELS.index(st.session_state.model), format_func=lambda x: LABELS[x])
    st.session_state.model = model
    st.markdown(f'<div class="micro" style="color:#8B887B;line-height:1.8;">20,000 TEST SESSIONS<br>32,968 ITEMS<br>SEED 42 / 20 EPOCHS</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:#2B2B27;margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="micro" style="color:#8B887B;margin-bottom:4px;">LOAD SESSION</div>', unsafe_allow_html=True)
    sid = st.text_input('SESSION ID', key='sid_input', label_visibility='collapsed', placeholder='10114')
    if st.button('LOAD / 载入', use_container_width=True):
        load_by_id(sid)
    st.markdown('<div class="micro" style="color:#8B887B;margin-top:10px;margin-bottom:4px;">CURATED CASES</div>', unsafe_allow_html=True)
    for kind, label in [('success', 'SUCCESS / 成功'), ('failure', 'FAILURE / 失败'), ('disagreement', 'DISAGREE / 分歧')]:
        pool = [c for c in CASES if c['kind'] == kind]
        if pool and st.button(label, use_container_width=True):
            load_case(random.choice(pool))

# ── header ───────────────────────────────────────────────────────────────
st.markdown(hero_html(), unsafe_allow_html=True)
st.markdown(ticker_html(), unsafe_allow_html=True)

# ── workbench ────────────────────────────────────────────────────────────
if page == '推荐工作台':
    left, right = st.columns([0.88, 1.12], gap='large')
    with left:
        st.markdown(panel_head('01', 'INPUT / SESSION', 'MAX 20 CLICKS'), unsafe_allow_html=True)
        raw = st.text_area('浏览序列', value=st.session_state.history_text,
                           key=f"editor_{st.session_state.editor_version}",
                           height=110, label_visibility='collapsed')
        st.session_state.history_text = raw
        items = parse_items(raw)
        if len(items) > 20:
            st.warning('序列超过 20 个，只使用最后 20 个。')
            items = items[-20:]
        source_history = st.session_state.get('source_history')
        if source_history and tuple(items) != tuple(source_history):
            st.session_state.target = None
            st.session_state.source_id = None
        st.markdown(chips_html(items, st.session_state.target), unsafe_allow_html=True)
        meta = []
        if st.session_state.source_id:
            meta.append(f'SOURCE {st.session_state.source_id}')
        meta.append(f'CLICKS {len(items)}')
        if st.session_state.target:
            meta.append(f'TARGET {st.session_state.target}')
        st.markdown(f'<div class="micro">{esc(" / ".join(meta))}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            run_one = st.button('RUN MODEL / 运行', use_container_width=True)
        with c2:
            run_all = st.button('3-MODEL COMPARE / 对比', use_container_width=True)
        if run_one or run_all:
            if not items:
                st.error('请先输入至少一个商品 ID。')
            else:
                try:
                    run_models([st.session_state.model] if run_one else MODELS, items)
                except Exception as exc:
                    st.error(str(exc))

        if st.session_state.preds and st.session_state.pred_history == tuple(items):
            target = st.session_state.target
            if st.session_state.compare:
                rows = []
                for mk in MODELS:
                    recs = st.session_state.preds.get(mk)
                    if not recs:
                        continue
                    r = rank_of(recs, target)
                    rows.append([LABELS[mk], '—' if not target else (r if r else 'MISS'),
                                 recs[0]['item'], f"{recs[0]['score']:.4f}"])
                st.markdown('<div class="micro" style="margin-top:14px;">SAME INPUT / THREE MODELS</div>', unsafe_allow_html=True)
                st.markdown(table_html(['MODEL', 'TARGET RANK', 'TOP-1', 'SCORE'], rows), unsafe_allow_html=True)
            else:
                lat = st.session_state.latency.get(st.session_state.model, 0)
                st.markdown(f'<div class="stat-strip"><div><b>{len(items)}</b><span>CLICKS</span></div>'
                            f'<div><b>{lat:.1f}</b><span>MS / INFERENCE</span></div>'
                            f'<div><b>{len(st.session_state.preds.get(st.session_state.model, []))}</b><span>TOP-K RETURNED</span></div></div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="awaiting">AWAITING INPUT / RUN THE MODEL</div>', unsafe_allow_html=True)

    with right:
        st.markdown(panel_head('02', 'RANKING / TOP-10', LABELS[st.session_state.model]), unsafe_allow_html=True)
        if st.session_state.preds and st.session_state.pred_history == tuple(items):
            target = st.session_state.target
            if st.session_state.compare:
                st.markdown(scorecards_html(), unsafe_allow_html=True)
                tabs = st.tabs([LABELS[m] for m in MODELS])
                for tab, mk in zip(tabs, MODELS):
                    with tab:
                        recs = st.session_state.preds.get(mk)
                        if not recs:
                            continue
                        r = rank_of(recs, target)
                        st.markdown(readout_html(r, target, st.session_state.latency.get(mk, 0), len(items), mk), unsafe_allow_html=True)
                        st.markdown(leaderboard_html(recs, target), unsafe_allow_html=True)
            else:
                recs = st.session_state.preds.get(st.session_state.model)
                if recs:
                    r = rank_of(recs, target)
                    st.markdown(readout_html(r, target, st.session_state.latency.get(st.session_state.model, 0), len(items), st.session_state.model), unsafe_allow_html=True)
                    st.markdown(leaderboard_html(recs, target), unsafe_allow_html=True)
        else:
            st.markdown('<div class="awaiting">NO SIGNAL YET<br>RUN THE MODEL TO GENERATE A RANKING</div>', unsafe_allow_html=True)

        with st.expander('SIGNAL GRAPH / 会话图'):
            if st.session_state.preds and st.session_state.pred_history == tuple(items):
                try:
                    predictor = get_predictor(st.session_state.model)
                    graph = predictor.graph(items)
                    st.markdown('<div class="graph-wrap">' + graph_svg(graph['nodes'], graph['edges'], st.session_state.target, THEME) + '</div>', unsafe_allow_html=True)
                    if predictor.model_name == 'GCEGNN' and items:
                        neighbours = predictor.global_neighbours(items[-1], limit=10)
                        if neighbours:
                            st.markdown('<div class="micro" style="margin-top:8px;">GLOBAL NEIGHBOURS / LAST ITEM</div>', unsafe_allow_html=True)
                            st.markdown(''.join(f'<span class="chip">{esc(n["id"])} · {n["weight"]:.1f}</span>' for n in neighbours), unsafe_allow_html=True)
                except Exception as exc:
                    st.info(f'图暂时不可用：{exc}')
            else:
                st.markdown('<div class="micro">先运行一次推荐，再查看该会话的局部图。</div>', unsafe_allow_html=True)

# ── test browser ─────────────────────────────────────────────────────────
elif page == '测试集检索':
    st.markdown('<div class="hero-top"><span>TEST SET / 20,000 SESSIONS</span><span>FILTER / INSPECT / LOAD</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="manifesto">FIND THE <em>FAILURES</em>, NOT JUST THE WINS.</div>', unsafe_allow_html=True)
    st.markdown('<div class="note-block">按真实命中结果、历史长度和重复点击筛选测试会话；点击载入后回到工作台运行模型。</div>', unsafe_allow_html=True)
    df = load_predictions(st.session_state.model)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c1:
        outcome = st.selectbox('OUTCOME', ['ALL / 全部', 'HIT / 命中', 'MISS / 未命中'])
    with c2:
        lo, hi = st.slider('HISTORY LENGTH', 4, 19, (4, 19))
    with c3:
        repeat_only = st.checkbox('REPEAT CLICKS ONLY')
    view = df[(df['history_len'] >= lo) & (df['history_len'] <= hi)]
    if outcome.startswith('HIT'):
        view = view[view['hit']]
    elif outcome.startswith('MISS'):
        view = view[~view['hit']]
    if repeat_only:
        view = view[view['repeat_count'] > 0]
    st.markdown(f'<div class="stat-strip"><div><b>{len(view):,}</b><span>FILTERED SESSIONS</span></div>'
                f'<div><b>{view["hit"].mean() * 100 if len(view) else 0:.2f}%</b><span>RECALL@10</span></div>'
                f'<div><b>{view["history_len"].mean() if len(view) else 0:.1f}</b><span>AVG CLICKS</span></div></div>',
                unsafe_allow_html=True)
    if view.empty:
        st.info('当前筛选没有会话。')
    else:
        rows = []
        flags = []
        for _, r in view.head(40).iterrows():
            rows.append([r['session_id'], r['history_len'], r['repeat_count'], r['target'],
                         r['target_rank'] if r['target_rank'] else 'MISS', r['top1'], f"{r['top1_score']:.3f}"])
            flags.append(bool(r['hit']))
        st.markdown(table_html(['SESSION', 'CLICKS', 'REPEAT', 'TARGET', 'RANK', 'TOP-1', 'SCORE'], rows, flags), unsafe_allow_html=True)
        pick = st.selectbox('SELECT SESSION / 选择会话', view['session_id'].head(300).tolist())
        if st.button('LOAD SELECTED / 载入所选'):
            load_by_id(pick)
        st.markdown('<div class="manifesto" style="font-size:24px;margin-top:26px;">GROUPED <em>SIGNAL</em></div>', unsafe_allow_html=True)
        g = view.copy()
        g['长度分组'] = pd.cut(g['history_len'], bins=[3, 7, 11, 15, 19], labels=['4–7', '8–11', '12–15', '16–19'])
        by_len = g.groupby('长度分组', observed=True).agg(会话数=('hit', 'size'), Recall=('hit', 'mean')).reset_index()
        by_rep = g.groupby(g['repeat_count'] > 0).agg(会话数=('hit', 'size'), Recall=('hit', 'mean')).reset_index()
        by_rep['重复点击'] = by_rep['repeat_count'].map({True: 'YES', False: 'NO'})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="micro">BY SESSION LENGTH</div>', unsafe_allow_html=True)
            st.markdown(table_html(['BIN', 'N', 'RECALL'], [[r['长度分组'], r['会话数'], f"{r['Recall']:.4f}"] for _, r in by_len.iterrows()]), unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="micro">BY REPEAT CLICKS</div>', unsafe_allow_html=True)
            st.markdown(table_html(['REPEAT', 'N', 'RECALL'], [[r['重复点击'], r['会话数'], f"{r['Recall']:.4f}"] for _, r in by_rep.iterrows()]), unsafe_allow_html=True)

# ── model matrix ─────────────────────────────────────────────────────────
elif page == '模型对比':
    st.markdown('<div class="hero-top"><span>MODEL MATRIX / 20K FORMAL RUN</span><span>QUALITY / COST / CURVE</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="manifesto">THREE MODELS.<br>ONE <em>RANKING</em>.</div>', unsafe_allow_html=True)
    st.markdown(scorecards_html(), unsafe_allow_html=True)
    comp = load_comparison()
    rows = []
    for _, r in comp.iterrows():
        rows.append([LABELS[r['model']], f"{r['recall@10']:.4f}", f"{r['mrr@10']:.4f}", f"{r['ndcg@10']:.4f}",
                     f"{r['hit@10']:.4f}", f"{r['precision@10']:.4f}", f"{r['training_and_validation_seconds']:.1f}",
                     f"{int(r['parameters']):,}"])
    st.markdown(table_html(['MODEL', 'RECALL', 'MRR', 'NDCG', 'HIT', 'PRECISION', 'TIME/S', 'PARAMS'], rows), unsafe_allow_html=True)
    st.markdown('<div class="manifesto" style="font-size:24px;margin-top:30px;">LEARNING <em>CURVE</em></div>', unsafe_allow_html=True)
    st.altair_chart(curve_chart(THEME), use_container_width=True)
    st.markdown('<div class="manifesto" style="font-size:24px;margin-top:30px;">RECALL BY <em>LENGTH</em></div>', unsafe_allow_html=True)
    frames = []
    for key in MODELS:
        d = load_predictions(key)[['session_id', 'history_len', 'hit']].copy()
        d['MODEL'] = LABELS[key]
        frames.append(d)
    all_df = pd.concat(frames, ignore_index=True)
    all_df['BIN'] = pd.cut(all_df['history_len'], bins=[3, 7, 11, 15, 19], labels=['4–7', '8–11', '12–15', '16–19'])
    pivot = all_df.groupby(['BIN', 'MODEL'], observed=True)['hit'].mean().unstack()
    rows = []
    for idx, r in pivot.iterrows():
        rows.append([idx] + [f'{r[m]:.4f}' for m in pivot.columns])
    st.markdown(table_html(['BIN'] + list(pivot.columns), rows), unsafe_allow_html=True)
    st.markdown('<div class="note-block">单种子、单次实验；数值为描述性结果。没有多种子、简单基线与 GCE 去全局分支消融。</div>', unsafe_allow_html=True)

# ── notes ────────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="hero-top"><span>NOTES / METHOD & BOUNDARY</span><span>WHAT THIS IS / WHAT IT IS NOT</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="manifesto">A REAL <em>INFERENCE</em> DEMO,<br>NOT A SLIDE DECK.</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="note-block">
<strong>DATA</strong><br>
完整 Diginetica 先做 item 5-core，再抽取 {WEB['meta']['counts']['sessions']:,} 个 5–20 次点击会话；排除 smoke。
正式子集 {WEB['meta']['counts']['items']:,} 个商品，训练 / 验证 / 测试 {WEB['meta']['counts']['train']:,} / {WEB['meta']['counts']['valid']:,} / {WEB['meta']['counts']['test']:,}。
</div>
<div class="note-block">
<strong>PROTOCOL</strong><br>
会话级留一：最后一项测试、倒数第二项验证；全候选排序；Recall@10、MRR@10、NDCG@10。
三模型统一 batch 128、20 轮上限、早停 5、embedding/hidden 64、CE、CPU、seed=42。
</div>
<div class="note-block">
<strong>LIVE INFERENCE</strong><br>
本页不是回放预存结果。`app/predictor.py` 把输入序列映射成内部 ID、构造对应会话图，再调用正式 checkpoint 的 `full_sort_predict`。
已与离线预测校验：Top-10 商品与排名一致，得分最大差约 5×10⁻⁶。
</div>
<div class="note-block">
<strong>BOUNDARY</strong><br>
单种子、单次实验；没有多种子、最后点击/热门基线、GCE 去全局分支消融。
不能宣称统计显著，也不能把三模型差异单独归因于全局分支。
商品 ID 为 Diginetica 匿名编码；得分为模型分数，不是购买概率。
</div>
""", unsafe_allow_html=True)
