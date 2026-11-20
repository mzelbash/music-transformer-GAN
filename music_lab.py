# -*- coding: utf-8 -*-
"""
Chapter 11 Code Lab (Streamlit).
Live music21 demos for the lecture + full copy-paste Colab code for every section.

Run locally:   streamlit run music_lab.py
Deploy:        push this folder to Streamlit Community Cloud, main file = music_lab.py
"""
import io
import os
import shutil
import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Music Gen Lab · Chapter 11", page_icon="🎵", layout="wide")

AMBER, INDIGO, GREEN, GREY, INK = "#E8952F", "#3E5BD6", "#2E9E52", "#7A8194", "#1B2333"

# ============================================================ theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{ --brand:#3E5BD6; --brand-d:#2f47b5; --amber:#E8952F; --green:#2E9E52; --ink:#1B2333;
       --muted:#5b6474; --line:#e4e8f1; --soft:#f5f6fb; }
html, body, [class*="css"], .stApp { font-family:'Inter', system-ui, -apple-system, sans-serif; }
.stApp { background:var(--soft); }
.block-container { padding-top:1.2rem; padding-bottom:3rem; max-width:1140px; }
p, li { line-height:1.65; }

/* ---------- page header ---------- */
.crumb { display:flex; align-items:center; flex-wrap:wrap; gap:8px; font-size:.82rem;
         font-weight:600; color:var(--muted); letter-spacing:.01em; margin:4px 0 8px; }
.crumb .part { color:var(--brand); font-weight:700; }
.crumb .sep { color:#b3bacb; }
.crumb .badge { margin-left:auto; background:#eef1f8; color:#3d4a72; border:1px solid #dfe4f0;
                border-radius:6px; padding:2px 10px; font-size:.7rem; font-weight:700;
                letter-spacing:.07em; }
.prog { height:4px; border-radius:4px; background:#e6e9f2; margin:0 0 18px; overflow:hidden; }
.prog > div { height:100%; background:var(--brand); border-radius:4px; }
.ptitle { font-size:2.15rem; font-weight:800; color:var(--ink); line-height:1.15;
          letter-spacing:-.015em; margin:0 0 8px; }
.psub { font-size:1.12rem; color:var(--muted); line-height:1.5; max-width:860px; }
.phr { height:1px; background:var(--line); border:0; margin:20px 0 6px; }

/* ---------- section heading ---------- */
.sec { font-size:1.18rem; font-weight:700; color:var(--ink); margin:30px 0 8px;
       letter-spacing:-.005em; }
.sec-sub { color:var(--muted); font-size:.96rem; margin:-2px 0 12px; }

/* ---------- lead paragraph ---------- */
.lead { font-size:1.1rem; color:#363d4d; line-height:1.7; margin:14px 0 6px; max-width:900px; }
.lead b { color:var(--ink); }

/* ---------- key points ---------- */
.pts { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 20px 8px;
       margin:14px 0; }
.pts-t { font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
         color:var(--muted); margin-bottom:8px; }
.pts ul { margin:0; padding-left:1.1rem; }
.pts li { margin:0 0 7px; font-size:1rem; color:#2a3040; }
.pts li::marker { color:var(--brand); }

/* ---------- callouts ---------- */
.cal { border-radius:12px; padding:13px 18px; margin:12px 0; border:1px solid; }
.cal .cal-label { display:block; font-size:.72rem; font-weight:700; letter-spacing:.09em;
                  text-transform:uppercase; margin-bottom:4px; }
.cal .cal-body { font-size:1rem; color:#2a3040; line-height:1.65; }
.cal.example { border-color:#f1dcbd; background:#fffaf2; }
.cal.example .cal-label { color:#b26a12; }
.cal.concept { border-color:#d9e0f7; background:#f5f7fe; }
.cal.concept .cal-label { color:var(--brand); }
.cal.key { border-color:#d6dbe6; background:#f6f7fa; }
.cal.key .cal-label { color:#3d4a72; }
code { background:rgba(62,91,214,.08); color:#2a3a8f; padding:1px 6px; border-radius:5px;
       font-size:.9em; }

/* ---------- in / out boxes ---------- */
.io { display:flex; gap:14px; align-items:stretch; margin:10px 0; flex-wrap:wrap; }
.io-box { flex:1; min-width:220px; background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:14px 16px; box-shadow:0 1px 5px rgba(20,30,60,.04); }
.io-lab { font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
          color:var(--muted); margin-bottom:4px; }
.io-val { font-size:1.02rem; color:var(--ink); font-weight:600; }
.io-arrow { align-self:center; font-size:1.6rem; color:var(--brand); font-weight:800; }

/* ---------- home cards ---------- */
.feat { background:#fff; border:1px solid var(--line); border-radius:14px; padding:20px 20px;
        height:100%; }
.feat-i { font-size:1.6rem; }
.feat-t { font-weight:800; font-size:1.08rem; color:var(--ink); margin:6px 0 6px; }
.feat-b { font-size:.96rem; color:var(--muted); line-height:1.55; }
.path { background:#fff; border:1px solid var(--line); border-radius:16px; padding:20px 22px;
        box-shadow:0 2px 10px rgba(20,30,60,.05); height:100%; border-top:6px solid var(--brand); }
.path.gan { border-top-color:#3d4a72; }
.path-k { font-size:.74rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
          color:var(--brand); }
.path.gan .path-k { color:#3d4a72; }
.path-t { font-size:1.3rem; font-weight:800; color:var(--ink); margin:4px 0 4px; }
.path-s { color:var(--muted); font-size:.96rem; margin-bottom:10px; }
.path ol { margin:0; padding-left:1.3rem; }
.path li { margin:0 0 4px; color:#2a3040; font-size:.97rem; }
.step { background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px;
        text-align:center; height:100%; box-shadow:0 2px 10px rgba(20,30,60,.05); }
.step-n { width:38px; height:38px; border-radius:50%; background:var(--brand); color:#fff;
          font-weight:800; display:inline-flex; align-items:center; justify-content:center;
          font-size:1.05rem; }
.step-t { font-weight:800; color:var(--ink); margin:10px 0 4px; font-size:1.04rem; }
.step-b { color:var(--muted); font-size:.93rem; line-height:1.5; }

/* ---------- widgets ---------- */
.stButton>button, .stDownloadButton>button { background:var(--brand); color:#fff; border:0;
        border-radius:10px; padding:9px 20px; font-weight:700; }
.stButton>button:hover, .stDownloadButton>button:hover { background:var(--brand-d); color:#fff; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px;
        padding:14px 16px; box-shadow:0 1px 4px rgba(20,30,60,.04); }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:#fff;
        box-shadow:0 1px 5px rgba(20,30,60,.04); }
[data-testid="stExpander"] summary { font-weight:700; color:var(--ink); font-size:1rem; }
[data-testid="stExpander"] summary:hover { color:var(--brand); }
hr { margin:1.3rem 0; }

/* ---------- sidebar menu ---------- */
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
.sb-brand { background:linear-gradient(120deg,#3E5BD6,#6d84ec); color:#fff; border-radius:14px;
            padding:14px 16px; margin:0 0 6px; }
.sb-brand-t { font-weight:800; font-size:1.08rem; }
.sb-brand-s { font-size:.8rem; opacity:.92; }
[data-testid="stNavSectionHeader"] { color:#c67614 !important; font-weight:800 !important;
        font-size:.74rem !important; letter-spacing:.08em; text-transform:uppercase;
        margin-top:14px !important; }
[data-testid="stSidebarNavLink"] { border-radius:10px; padding:6px 10px !important; margin:1px 0; }
[data-testid="stSidebarNavLink"] span { font-size:.98rem !important; font-weight:600; color:#2a3040; }
[data-testid="stSidebarNavLink"]:hover { background:#eef2ff; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background:var(--brand); }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color:#fff !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:var(--brand); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color:#fff !important; }
</style>
""", unsafe_allow_html=True)

SR = 22050


# ============================================================ music helpers
@st.cache_resource(show_spinner=False)
def load_score():
    """A four-voice Bach chorale that ships with music21 (no download needed)."""
    from music21 import corpus
    return corpus.parse("bach/bwv66.6")


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def synth_wav(seq, spq=0.5):
    """seq: list of (midi or None, duration_in_beats). Returns WAV bytes."""
    buf = []
    for m, d in seq:
        n = int(max(d * spq, 0.12) * SR)
        t = np.linspace(0, n / SR, n, endpoint=False)
        if m is None:
            w = np.zeros(n)
        else:
            f = midi_hz(m)
            w = sum(a * np.sin(2 * np.pi * f * k * t) for k, a in enumerate([1, .5, .28, .14], 1))
            env = np.exp(-2.2 * t)
            a = int(0.02 * SR)
            env[:a] *= np.linspace(0, 1, a)
            w = w * env
        buf.append(w)
    sig = np.concatenate(buf) if buf else np.zeros(1)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
    data = (sig * 32767).astype(np.int16)
    b = io.BytesIO()
    with wave.open(b, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(data.tobytes())
    return b.getvalue()


def top_line(score, limit=64):
    from music21 import note, chord
    seq = []
    for el in score.chordify().recurse():
        if isinstance(el, note.Note):
            seq.append((el.pitch.midi, float(el.duration.quarterLength)))
        elif isinstance(el, chord.Chord):
            seq.append((el.pitches[-1].midi, float(el.duration.quarterLength)))
        if len(seq) >= limit:
            break
    return seq


def to_streams(score, limit=None):
    from music21 import note, chord, meter
    notes, durs = ["START"], ["0.0"]
    try:
        k = score.analyze("key")
        notes.append(f"{k.tonic.name}:{k.mode}"); durs.append("0.0")
    except Exception:
        pass
    ts = score.recurse().getElementsByClass(meter.TimeSignature)
    notes.append((ts[0].ratioString if ts else "4/4") + "TS"); durs.append("0.0")
    for el in score.chordify().recurse():
        if isinstance(el, note.Note):
            notes.append(str(el.pitch)); durs.append(str(el.duration.quarterLength))
        elif isinstance(el, chord.Chord):
            notes.append(str(el.pitches[-1])); durs.append(str(el.duration.quarterLength))
        elif isinstance(el, note.Rest):
            notes.append("REST"); durs.append(str(el.duration.quarterLength))
        if limit and len(notes) >= limit:
            break
    return notes, durs


def piano_roll_fig(score):
    colors = ["#E23B3B", AMBER, GREEN, INDIGO]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    max_off = 0
    for pi, part in enumerate(score.parts):
        for n in part.recurse().notes:
            pitches = n.pitches if n.isChord else [n.pitch]
            for p in pitches:
                ax.add_patch(Rectangle((float(n.offset), p.midi - 0.4),
                             float(n.duration.quarterLength), 0.8,
                             color=colors[pi % 4], ec="white", lw=0.5))
            max_off = max(max_off, float(n.offset) + float(n.duration.quarterLength))
    ax.set_xlim(0, min(max_off, 24)); ax.set_ylim(48, 82)
    ax.set_xlabel("time (beats)"); ax.set_ylabel("pitch (MIDI number)")
    ax.set_title("Piano roll: pitch x time, one color per voice")
    fig.tight_layout()
    return fig


def guess_mscore():
    candidates = [
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
        "/Applications/MuseScore 3.app/Contents/MacOS/mscore",
        "/usr/bin/mscore4", "/usr/bin/mscore3", "/usr/bin/musescore3",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for name in ["MuseScore4", "MuseScore3", "mscore", "mscore3", "musescore3"]:
        p = shutil.which(name)
        if p:
            return p
    return ""


# ============================================================ UI components
import re as _re
import html as _html


def _inline(md):
    """Tiny markdown for HTML cards: **bold**, `code`, and the text itself escaped."""
    s = _html.escape(md, quote=False)
    s = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = _re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


BADGES = {"LIVE DEMO", "MODEL", "TRAINING", "MODEL + TRAINING", "THE PAYOFF", "COLAB SETUP",
          "OVERVIEW", "FULL CODE"}
TOTAL_STEPS = 15


def hero(title, subtitle, tags, accent=None):
    """Page header: breadcrumb and progress on top, then the title and subtitle."""
    tags = [tags] if isinstance(tags, str) else list(tags)
    badge = tags.pop() if tags and tags[-1] in BADGES else ""
    crumb = ""
    for i, t in enumerate(tags):
        if i:
            crumb += '<span class="sep">›</span>'
        crumb += f'<span class="{"part" if i == 0 else ""}">{t}</span>'
    if badge:
        crumb += f'<span class="badge">{badge}</span>'
    prog = ""
    m = _re.search(r"Step (\d+) of (\d+)", " ".join(tags))
    if m:
        pct = int(int(m.group(1)) / int(m.group(2)) * 100)
        prog = f'<div class="prog"><div style="width:{pct}%"></div></div>'
    st.markdown(f'<div class="crumb">{crumb}</div>{prog}<div class="ptitle">{title}</div>'
                f'<div class="psub">{subtitle}</div><div class="phr"></div>',
                unsafe_allow_html=True)


def section(title, sub=None):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{_inline(sub)}</div>', unsafe_allow_html=True)


def lead(md):
    """The opening idea of a page, in one clean card."""
    st.markdown(f'<div class="lead">{_inline(md)}</div>', unsafe_allow_html=True)


def points(items, title="Key ideas", tone=""):
    lis = "".join(f"<li>{_inline(i)}</li>" for i in items)
    st.markdown(f'<div class="pts"><div class="pts-t">{title}</div><ul>{lis}</ul></div>',
                unsafe_allow_html=True)


def callout(kind, label, body_html):
    st.markdown(f'<div class="cal {kind}"><span class="cal-label">{label}</span>'
                f'<div class="cal-body">{body_html}</div></div>', unsafe_allow_html=True)


def io(pairs):
    """In -> out boxes. pairs = [(label, value), ...]."""
    parts = []
    for i, (lab, val) in enumerate(pairs):
        if i:
            parts.append('<div class="io-arrow">&rarr;</div>')
        parts.append(f'<div class="io-box"><div class="io-lab">{lab}</div>'
                     f'<div class="io-val">{_inline(val)}</div></div>')
    st.markdown(f'<div class="io">{"".join(parts)}</div>', unsafe_allow_html=True)


def feat(icon, title, body):
    st.markdown(f'<div class="feat"><div class="feat-i">{icon}</div><div class="feat-t">{title}</div>'
                f'<div class="feat-b">{body}</div></div>', unsafe_allow_html=True)


def colab_code(block, title="Colab code"):
    """Code in a dark editor card with a Copy button, folded inside an expander."""
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    src = block.strip("\n")
    n = src.count("\n") + 1
    with st.expander(f"💻  {title}  ·  {n} lines  ·  click to open", expanded=False):
        hl = highlight(src, PythonLexer(), HtmlFormatter(style="monokai", noclasses=True))
        esc = _html.escape(src)
        height = min(48 + n * 20 + 18, 660)
        doc = (
            "<!doctype html><html><head><meta charset='utf-8'><style>"
            "*{box-sizing:border-box;} body{margin:0;font-family:Consolas,'SF Mono',monospace;}"
            ".ed{border-radius:11px;overflow:hidden;background:#272822;border:1px solid #14150e;"
            "box-shadow:0 6px 18px rgba(0,0,0,.22);}"
            ".bar{background:#1d1e17;padding:8px 12px;display:flex;align-items:center;gap:7px;}"
            ".dot{width:11px;height:11px;border-radius:50%;}"
            ".r{background:#ff5f56}.y{background:#ffbd2e}.g{background:#27c93f}"
            ".name{color:#9aa0a6;font-size:12px;margin-left:8px;letter-spacing:.02em;}"
            ".copy{margin-left:auto;background:#3a3b32;color:#dfe2d6;border:0;border-radius:6px;"
            "padding:4px 13px;font-size:12px;cursor:pointer;font-family:inherit;}"
            ".copy:hover{background:#4b4c41;}"
            ".highlight{background:transparent!important;margin:0;}"
            ".highlight pre{margin:0!important;padding:14px 16px!important;overflow:auto;"
            "font-size:13.5px;line-height:1.55;background:transparent!important;}"
            "</style></head><body>"
            "<div class='ed'><div class='bar'>"
            "<span class='dot r'></span><span class='dot y'></span><span class='dot g'></span>"
            "<span class='name'>python &nbsp;&middot;&nbsp; paste into Colab</span>"
            "<button class='copy' onclick='cp()'>Copy</button></div>"
            + hl +
            "</div><textarea id='src' style='position:absolute;left:-9999px;top:0;'>" + esc + "</textarea>"
            "<script>function cp(){var t=document.getElementById('src');t.select();"
            "try{document.execCommand('copy');}catch(e){}"
            "var b=document.querySelector('.copy');b.textContent='Copied!';"
            "setTimeout(function(){b.textContent='Copy';},1200);}</script>"
            "</body></html>")
        components.html(doc, height=height, scrolling=True)


def footer_nav(func):
    try:
        i = ORDER_FUNCS.index(func)
    except (ValueError, NameError):
        return
    st.divider()
    c1, c2 = st.columns(2)
    try:
        if i > 0:
            with c1:
                st.page_link(ORDER_PAGES[i - 1], label=f"Back: {ORDER_PAGES[i - 1].title}",
                             icon=":material/arrow_back:")
        if i < len(ORDER_PAGES) - 1:
            with c2:
                st.page_link(ORDER_PAGES[i + 1], label=f"Next: {ORDER_PAGES[i + 1].title}",
                             icon=":material/arrow_forward:")
    except Exception:
        pass


# ============================================================ CODE (Colab)
C = {}

C["setup"] = '''
# --- One-time setup in Colab (use a GPU runtime for training) ---
!pip -q install music21 keras-hub

import glob, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import music21
from music21 import corpus, note, chord, meter
print("TensorFlow", tf.__version__)
'''

C["download"] = '''
# music21 ships hundreds of Bach scores, so this runs in Colab with NO download.
from music21 import corpus
bach_paths = corpus.getComposer("bach")      # bundled Bach file paths (~430)
print(len(bach_paths), "Bach scores available offline (no download needed)")

# Note: the book itself uses the J.S. Bach *cello suites*, fetched by a separate
# Kaggle download script. The pipeline below is identical, so we use the built-in
# corpus here so everything just runs. To use your own MIDI folder instead:
#     bach_paths = sorted(glob.glob("your_midi_folder/*.mid"))
'''

C["musescore"] = '''
# --- Optional: graphical notation so score.show() draws real sheet music ---
# music21 renders notation by calling MuseScore. Two things that trip people up in Colab:
#   1) run `apt-get update` FIRST, or you get 404 "Failed to fetch" errors (stale package cache)
#   2) only point music21 at the binary AFTER confirming it installed, or music21 raises.

!apt-get -qq update
!apt-get -qq -y install musescore3 xvfb 2>/dev/null || apt-get -qq -y --fix-missing install musescore3 xvfb

import os, shutil
from music21 import environment

# find the MuseScore binary wherever apt put it
mscore = (shutil.which("mscore3") or shutil.which("musescore3")
          or shutil.which("mscore") or "/usr/bin/mscore3")

if os.path.exists(mscore):
    env = environment.UserSettings()
    env["musescoreDirectPNGPath"] = mscore
    env["musicxmlPath"] = mscore
    os.environ["QT_QPA_PLATFORM"] = "offscreen"      # MuseScore needs a (virtual) display
    print("MuseScore ready:", mscore)
    # score.show()                  # now renders the whole score as an image
    # score.measures(1, 4).show()   # just the first few bars
else:
    print("MuseScore did not install. score.show('text') still works with no install.")
    print("If needed, retry:  !apt-get update && apt-get install -y musescore3 xvfb")
'''

C["parse"] = '''
from music21 import corpus

# Parse one score and collapse to a single line (keep the top note -> monophonic)
score = corpus.parse(bach_paths[0]).chordify()

score.show("text")     # the same music as a structured event list
# score.show()         # graphical notation (needs MuseScore installed)
'''

C["streams"] = '''
from music21 import corpus, note, chord, meter

def parse_to_streams(score):
    notes, durations = ["START"], ["0.0"]
    try:
        key = score.analyze("key")
        notes.append(f"{key.tonic.name}:{key.mode}"); durations.append("0.0")
    except Exception:
        pass
    ts = score.recurse().getElementsByClass(meter.TimeSignature)
    notes.append((ts[0].ratioString if ts else "4/4") + "TS"); durations.append("0.0")
    for el in score.chordify().recurse():
        if isinstance(el, note.Note):
            notes.append(str(el.pitch));       durations.append(str(el.duration.quarterLength))
        elif isinstance(el, chord.Chord):
            notes.append(str(el.pitches[-1])); durations.append(str(el.duration.quarterLength))
        elif isinstance(el, note.Rest):
            notes.append("REST");              durations.append(str(el.duration.quarterLength))
    return notes, durations

# Build the corpus: one (notes, durations) string pair per score
notes_corpus, durations_corpus = [], []
for path in bach_paths[:120]:          # a subset keeps parsing quick; use all if you like
    try:
        n, d = parse_to_streams(corpus.parse(path))
        notes_corpus.append(" ".join(n)); durations_corpus.append(" ".join(d))
    except Exception:
        pass
print(len(notes_corpus), "scores parsed into streams")
'''

C["tokenize"] = '''
from tensorflow.keras import layers

# One vocabulary for notes, one for durations. TextVectorization gives every distinct
# token an integer id, most frequent first. Ids 0 and 1 are reserved for padding and [UNK].
def make_vectorizer(texts):
    vec = layers.TextVectorization(standardize=None, output_mode="int")
    vec.adapt(tf.data.Dataset.from_tensor_slices(texts).batch(128))
    return vec, vec.get_vocabulary()

notes_vec, notes_vocab = make_vectorizer(notes_corpus)
durs_vec,  durs_vocab  = make_vectorizer(durations_corpus)

print("note vocab:", len(notes_vocab), " duration vocab:", len(durs_vocab))
print("first note tokens:", notes_vocab[:10])
'''

C["window"] = '''
SEQ_LEN, BATCH_SIZE = 50, 256

# 1. Cut every piece into overlapping windows of SEQ_LEN + 1 tokens (slide by one token)
notes_seqs, durs_seqs = [], []
for n, d in zip(notes_corpus, durations_corpus):
    n, d = n.split(), d.split()
    for i in range(len(n) - SEQ_LEN):
        notes_seqs.append(" ".join(n[i:i + SEQ_LEN + 1]))
        durs_seqs.append(" ".join(d[i:i + SEQ_LEN + 1]))
print(len(notes_seqs), "training windows")

# 2. Turn text into ids. Input = first SEQ_LEN tokens, target = the same window shifted by one
def split_xy(n, d):
    n, d = notes_vec(n), durs_vec(d)
    return (n[:, :-1], d[:, :-1]), (n[:, 1:], d[:, 1:])

train_ds = (tf.data.Dataset.from_tensor_slices((notes_seqs, durs_seqs))
            .shuffle(10000)
            .batch(BATCH_SIZE)
            .map(split_xy)
            .prefetch(tf.data.AUTOTUNE))

(x_notes, x_durs), (y_notes, y_durs) = next(iter(train_ds))
print("input shape:", x_notes.shape, " target shape:", y_notes.shape)   # (256, 50) each
'''

C["posenc"] = '''
from tensorflow.keras import layers
try:
    import keras_hub as kh          # current name of the library
except ImportError:
    import keras_nlp as kh          # older name, same layer

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = kh.layers.SinePositionEncoding()   # deterministic, any length
    def call(self, x):
        embedding = self.token_emb(x)
        return embedding + self.pos_emb(embedding)
'''

C["model"] = '''
from tensorflow.keras import layers, models

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads, key_dim, output_shape=2 * key_dim)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = models.Sequential([layers.Dense(ff_dim, activation="relu"),
                                      layers.Dense(2 * key_dim)])
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)
    def call(self, x):
        n = tf.shape(x)[1]
        mask = tf.linalg.band_part(tf.ones((n, n)), -1, 0)  # causal
        a, scores = self.attn(x, x, attention_mask=mask, return_attention_scores=True)
        x = self.ln1(x + self.drop(a))
        f = self.ffn(x)
        return self.ln2(x + self.drop(f)), scores

EMBED_DIM, KEY_DIM, N_HEADS, FF_DIM = 128, 128, 5, 256
note_in = layers.Input(shape=(None,)); dur_in = layers.Input(shape=(None,))
note_emb = TokenAndPositionEmbedding(len(notes_vocab), EMBED_DIM)(note_in)
dur_emb  = TokenAndPositionEmbedding(len(durs_vocab),  EMBED_DIM)(dur_in)
x = layers.Concatenate()([note_emb, dur_emb])                 # [b, l, 256]
x, attention_scores = TransformerBlock(N_HEADS, KEY_DIM, FF_DIM)(x)
note_out = layers.Dense(len(notes_vocab), activation="softmax", name="note")(x)
dur_out  = layers.Dense(len(durs_vocab),  activation="softmax", name="duration")(x)

model = models.Model([note_in, dur_in], [note_out, dur_out])
model.compile(optimizer="adam",
              loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"])
model.fit(train_ds, epochs=20)
'''

C["generate"] = '''
def sample_with_temp(probs, temperature=0.5):
    probs = np.log(probs + 1e-9) / temperature
    probs = np.exp(probs); probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)

def generate(max_tokens=250, temperature=0.5):
    notes, durs = ["START"], ["0.0"]
    for _ in range(max_tokens):
        x1 = notes_vec([" ".join(notes)])
        x2 = durs_vec([" ".join(durs)])
        p_note, p_dur = model.predict([x1, x2], verbose=0)
        n = sample_with_temp(p_note[0][-1], temperature)
        d = sample_with_temp(p_dur[0][-1],  temperature)
        notes.append(notes_vocab[n]); durs.append(durs_vocab[d])
    return notes, durs

gen_notes, gen_durs = generate(temperature=0.5)
print(" ".join(gen_notes[:40]))
'''

C["analysis"] = '''
import matplotlib.pyplot as plt

# A sub-model that returns the attention scores from the Transformer block
att_model = models.Model(inputs=[note_in, dur_in], outputs=attention_scores)

L = min(SEQ_LEN, len(gen_notes))
x1 = notes_vec([" ".join(gen_notes[:L])])
x2 = durs_vec([" ".join(gen_durs[:L])])

att = att_model.predict([x1, x2], verbose=0)[0].mean(axis=0)   # [L, L], averaged over heads
p_note, _ = model.predict([x1, x2], verbose=0)                  # [1, L, note_vocab]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. Attention: row = the position making a prediction, column = where it looked
a1.imshow(att, cmap="Greens")
a1.set_xticks(range(L)); a1.set_xticklabels(gen_notes[:L], rotation=90, fontsize=6)
a1.set_yticks(range(L)); a1.set_yticklabels(gen_notes[:L], fontsize=6)
a1.set_title("Attention (darker = looked there more)")

# 2. Next-note probabilities at every position: blank rows = notes the model avoids
a2.imshow(p_note[0].T, aspect="auto", cmap="Greys", origin="lower")
a2.set_yticks(range(len(notes_vocab))); a2.set_yticklabels(notes_vocab, fontsize=5)
a2.set_xlabel("position in the sequence"); a2.set_title("P(next note) at each position")

plt.tight_layout(); plt.show()
'''

C["polyphony"] = '''
# GRID tokenization: at each timestep list the four voices in order S, A, T, B
# EVENT tokenization: a recipe of NOTE_ON / NOTE_OFF / TIME_SHIFT instructions
# See the Music Transformer paper (Huang et al., 2018) for both schemes.

# grid example (one row per timestep, four voice pitches):
grid = [[74, 70, 65, 58], [74, 70, 65, 55], [75, 70, 60, 55]]

# event example:
events = ["NOTE_ON<74>", "NOTE_ON<70>", "NOTE_ON<65>", "NOTE_ON<58>",
          "TIME_SHIFT<1.0>", "NOTE_OFF<58>", "NOTE_ON<55>", "TIME_SHIFT<0.5>"]
'''

C["musegan_data"] = '''
# MuseGAN treats the score as an image: [batch, N_BARS, N_STEPS, N_PITCHES, N_TRACKS]
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from music21 import corpus, meter, chord

N_BARS, N_STEPS, N_PITCHES, N_TRACKS = 2, 16, 84, 4
STEPS_PER_BEAT = N_STEPS // 4        # 16 steps per 4/4 bar = 4 steps per beat
LOWEST = 24                          # pitch row 0 = MIDI 24 (C1), so rows cover MIDI 24 to 107

def chorale_to_roll(score):
    """One chorale -> piano roll [time_steps, 84, 4], or None if it does not fit."""
    if len(score.parts) != N_TRACKS:
        return None
    ts = score.recurse().getElementsByClass(meter.TimeSignature)
    if ts and ts[0].ratioString != "4/4":
        return None
    n_steps = int(round(float(score.highestTime) * STEPS_PER_BEAT))
    roll = np.zeros((n_steps, N_PITCHES, N_TRACKS), dtype=np.float32)
    for t, part in enumerate(score.parts):                  # S, A, T, B
        for n in part.flatten().notes:
            p = n.pitches[-1] if isinstance(n, chord.Chord) else n.pitch
            row = p.midi - LOWEST
            start = int(round(float(n.offset) * STEPS_PER_BEAT))
            end = int(round(float(n.offset + n.duration.quarterLength) * STEPS_PER_BEAT))
            if 0 <= row < N_PITCHES and end > start:
                roll[start:end, row, t] = 1.0
    return roll

# The book uses 229 four-voice Bach chorales. music21 ships Bach chorales too,
# so we can build the training tensor offline, with no download.
chunks = []
for score in corpus.chorales.Iterator():
    roll = chorale_to_roll(score)
    if roll is None:
        continue
    chunk_len = N_BARS * N_STEPS                             # 2 bars = 32 steps
    for s in range(0, len(roll) - chunk_len + 1, chunk_len):
        chunks.append(roll[s:s + chunk_len].reshape(N_BARS, N_STEPS, N_PITCHES, N_TRACKS))

data = np.stack(chunks) * 2.0 - 1.0     # 0/1 -> -1/+1 to match the generator's tanh output
print("training tensor:", data.shape)   # (n_chunks, 2, 16, 84, 4)
'''

C["temporal"] = '''
Z_DIM = 32
initializer = tf.keras.initializers.RandomNormal(stddev=0.02)

def conv_t(x, f, k, s, a="relu", p="same", bn=True):
    x = layers.Conv2DTranspose(f, k, strides=s, padding=p, kernel_initializer=initializer)(x)
    if bn: x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation(a)(x)

def TemporalNetwork():
    z = layers.Input(shape=(Z_DIM,))
    x = layers.Reshape([1, 1, Z_DIM])(z)
    x = conv_t(x, 1024, (2, 1), (1, 1), p="valid")
    x = conv_t(x, Z_DIM, (N_BARS - 1, 1), (1, 1), p="valid")
    x = layers.Reshape([N_BARS, Z_DIM])(x)              # one latent vector per bar
    return models.Model(z, x)
'''

C["bargen"] = '''
def BarGenerator():
    z = layers.Input(shape=(Z_DIM * 4,))               # chords | style | melody | groove
    x = layers.Dense(1024)(z)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.Reshape([2, 1, 512])(x)
    x = conv_t(x, 512, (2, 1), (2, 1))                 # expand TIME
    x = conv_t(x, 256, (2, 1), (2, 1))
    x = conv_t(x, 256, (2, 1), (2, 1))
    x = conv_t(x, 256, (1, 7), (1, 7))                 # expand PITCH
    x = conv_t(x, 1,   (1, 12), (1, 12), a="tanh", bn=False)
    x = layers.Reshape([1, N_STEPS, N_PITCHES, 1])(x)
    return models.Model(z, x)                          # one bar for one voice
'''

C["generator"] = '''
def Generator():
    chords = layers.Input(shape=(Z_DIM,),           name="chords")
    style  = layers.Input(shape=(Z_DIM,),           name="style")
    melody = layers.Input(shape=(N_TRACKS, Z_DIM),  name="melody")
    groove = layers.Input(shape=(N_TRACKS, Z_DIM),  name="groove")

    chords_over_time = TemporalNetwork()(chords)                       # per bar
    melody_over_time = [TemporalNetwork()(
        layers.Lambda(lambda v, t=t: v[:, t, :])(melody)) for t in range(N_TRACKS)]
    barGen = [BarGenerator() for _ in range(N_TRACKS)]

    bars = []
    for bar in range(N_BARS):
        tracks = []
        c = layers.Lambda(lambda v, b=bar: v[:, b, :])(chords_over_time)
        for t in range(N_TRACKS):
            m = layers.Lambda(lambda v, b=bar: v[:, b, :])(melody_over_time[t])
            g = layers.Lambda(lambda v, t=t: v[:, t, :])(groove)
            z = layers.Concatenate(axis=1)([c, style, m, g])
            tracks.append(barGen[t](z))
        bars.append(layers.Concatenate(axis=-1)(tracks))
    out = layers.Concatenate(axis=1)(bars)
    return models.Model([chords, style, melody, groove], out)
'''

C["critic"] = '''
def conv(x, f, k, s, p="same"):
    x = layers.Conv3D(f, k, strides=s, padding=p, kernel_initializer=initializer)(x)
    return layers.LeakyReLU()(x)

def Critic():
    inp = layers.Input(shape=(N_BARS, N_STEPS, N_PITCHES, N_TRACKS))
    x = conv(inp, 128, (2, 1, 1), (1, 1, 1), "valid")          # collapse BARS
    x = conv(x,   128, (N_BARS - 1, 1, 1), (1, 1, 1), "valid")
    x = conv(x,   128, (1, 1, 12), (1, 1, 12))                 # collapse PITCH
    x = conv(x,   128, (1, 1, 7),  (1, 1, 7))
    x = conv(x,   256, (1, 4, 1),  (1, 2, 1))                  # collapse TIME
    x = conv(x,   512, (1, 3, 1),  (1, 2, 1))
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x); x = layers.LeakyReLU()(x)
    out = layers.Dense(1, activation=None)(x)                  # single realness score
    return models.Model(inp, out)                              # Conv3D, no BatchNorm
'''

C["wgan"] = '''
# WGAN-GP training loop (same idea as your GAN lecture)
generator, critic = Generator(), Critic()
g_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
c_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
CRITIC_STEPS, GP_WEIGHT, BATCH, EPOCHS = 5, 10.0, 64, 100

def gradient_penalty(real, fake):
    eps = tf.random.uniform([tf.shape(real)[0], 1, 1, 1, 1])
    mix = eps * real + (1 - eps) * fake
    with tf.GradientTape() as gt:
        gt.watch(mix); pred = critic(mix, training=True)
    grad = gt.gradient(pred, mix)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3, 4]) + 1e-12)
    return tf.reduce_mean((norm - 1.0) ** 2)

def rand_z(n):
    return [tf.random.normal((n, Z_DIM)), tf.random.normal((n, Z_DIM)),
            tf.random.normal((n, N_TRACKS, Z_DIM)), tf.random.normal((n, N_TRACKS, Z_DIM))]

@tf.function
def train_step(real):
    n = tf.shape(real)[0]
    # 1. Critic: score real above fake, plus the gradient penalty (several steps)
    for _ in range(CRITIC_STEPS):
        with tf.GradientTape() as tape:
            fake = generator(rand_z(n), training=True)
            c_loss = (tf.reduce_mean(critic(fake, training=True))
                      - tf.reduce_mean(critic(real, training=True))
                      + GP_WEIGHT * gradient_penalty(real, fake))
        grads = tape.gradient(c_loss, critic.trainable_variables)
        c_opt.apply_gradients(zip(grads, critic.trainable_variables))
    # 2. Generator: make the critic score its fakes higher
    with tf.GradientTape() as tape:
        fake = generator(rand_z(n), training=True)
        g_loss = -tf.reduce_mean(critic(fake, training=True))
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads, generator.trainable_variables))
    return c_loss, g_loss

train_data = (tf.data.Dataset.from_tensor_slices(data.astype("float32"))
              .shuffle(2000).batch(BATCH, drop_remainder=True))

for epoch in range(EPOCHS):
    for real in train_data:
        c_loss, g_loss = train_step(real)
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}   critic loss {float(c_loss):8.3f}   generator loss {float(g_loss):8.3f}")

# Generate: for every step and voice, the brightest pitch row is the note
sample = generator(rand_z(1), training=False)[0].numpy()     # [2, 16, 84, 4]
midi_notes = sample.argmax(axis=2) + LOWEST                    # [2, 16, 4]: S A T B per step
print(midi_notes[0, :8])                                       # first 8 steps of bar 1
'''

FULL_NOTEBOOK = "\n\n".join(f"# ===== {k} =====\n{v.strip()}" for k, v in C.items())


# ============================================================ pages
P1, P2 = "Part 1 · Transformer", "Part 2 · MuseGAN"


def page_home():
    hero("Music Generation Code Lab",
         "Teach a computer to write Bach, two ways. Every step has a live demo you can try in "
         "the browser and the full Python code to run in Colab.",
         ["Start here", "Chapter 11 · Generative Deep Learning", "OVERVIEW"])
    c1, c2, c3 = st.columns(3)
    with c1:
        feat("🎼", "Live demos", "Parse real Bach, build tokens, draw the piano roll, and listen, "
             "all in the browser. Nothing to install.")
    with c2:
        feat("💻", "Full Colab code", "Every step has runnable Python with a copy button. Open it "
             "when you need it, paste it into Colab, and run it on a GPU.")
    with c3:
        feat("🎛️", "Two ways to see music", "Part 1 reads music as a <b>sequence</b> of tokens. "
             "Part 2 draws music as an <b>image</b>.")

    section("Two models, two ways to see music")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown(
            '<div class="path"><div class="path-k">Part 1</div>'
            '<div class="path-t">Transformer: music as a sequence</div>'
            '<div class="path-s">Like a language model that predicts the next word, it predicts '
            'the next note and how long it lasts.</div><ol>'
            '<li>Parse MIDI with music21</li><li>Split into two streams: notes and durations</li>'
            '<li>Turn symbols into token IDs</li><li>Cut sliding training windows</li>'
            '<li>Add position encoding</li><li>Build and train the model</li>'
            '<li>Generate new music</li><li>Look inside: attention and keys</li>'
            '<li>Polyphony: several notes at once</li></ol></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(
            '<div class="path gan"><div class="path-k">Part 2</div>'
            '<div class="path-t">MuseGAN: music as an image</div>'
            '<div class="path-s">A GAN draws a whole piano roll at once: pitch up the side, time '
            'across, one layer per voice.</div><ol start="10">'
            '<li>Build the piano-roll tensor</li><li>Temporal network: one plan per bar</li>'
            '<li>Bar generator: paint one bar of one voice</li>'
            '<li>Full generator: every bar, every voice</li><li>Critic: score how real it looks</li>'
            '<li>WGAN-GP training</li></ol></div>', unsafe_allow_html=True)

    section("How each page works")
    s1, s2, s3 = st.columns(3)
    for col, n, t, b in [
        (s1, 1, "Read the idea", "A short explanation, the key points, and a small example."),
        (s2, 2, "Try the live demo", "Real Bach data, processed right here. Change a setting and "
                                     "watch what happens."),
        (s3, 3, "Open the Colab code", "Click the code panel to unfold it, copy it, and run it "
                                       "in Colab with a GPU.")]:
        with col:
            st.markdown(f'<div class="step"><div class="step-n">{n}</div><div class="step-t">{t}</div>'
                        f'<div class="step-b">{b}</div></div>', unsafe_allow_html=True)

    section("Before you start")
    points(["Open a new **Colab** notebook and switch the runtime to a **GPU**.",
            "Run the **Setup and data** page first. It installs music21 and loads the Bach scores.",
            "Then follow the pages in order with the menu on the left, or the **Back / Next** "
            "buttons at the bottom of each page."], "Checklist", "green")
    st.download_button("⬇  Download the whole notebook (.py)", FULL_NOTEBOOK,
                       file_name="chapter11_music.py", mime="text/x-python")
    footer_nav(page_home)


def page_setup():
    hero("Setup and data",
         "Install the libraries and load the Bach scores. Run these cells first in Colab.",
         ["Start here", "Setup", "COLAB SETUP"])
    points(["**music21** reads and writes music files and ships hundreds of Bach scores.",
            "**TensorFlow / Keras** builds and trains both models.",
            "Use a **GPU runtime** in Colab: Runtime → Change runtime type → GPU."],
           "What you need")
    section("1. Install and import")
    colab_code(C["setup"], "Install and import")
    section("2. Get the data", "music21 already includes the Bach scores, so there is nothing to "
                               "download.")
    colab_code(C["download"], "Load the Bach scores")
    section("3. Optional: sheet music with MuseScore",
            "music21 draws real notation by calling MuseScore. Skip this if `show('text')` is enough.")
    colab_code(C["musescore"], "Install MuseScore in Colab")
    callout("concept", "If apt shows 404 errors",
            "That is Colab's stale package cache. The <code>apt-get update</code> line in the cell "
            "fixes it. And <code>score.show('text')</code> always works with no install at all.")
    footer_nav(page_setup)


def page_parse():
    hero("Parse MIDI with music21",
         "Turn a music file into a list of timed events: the bridge from music to data.",
         [P1, "Step 1 of 15", "LIVE DEMO"])
    lead("A MIDI file is a list of instructions: which note, when it starts, how long it lasts. "
         "`music21` reads those instructions and gives us back **objects we can loop over in "
         "Python**.")
    points(["Each object has a **type** (Note, Rest, Chord, KeySignature, ...) and an **offset**: "
            "when it happens, in beats.",
            "A note knows its **pitch** (like `G3`) and its **duration** (like `1.0` beat).",
            "From here on, the music is just data."])
    try:
        score = load_score()
        st.info(f"Loaded a real Bach chorale with {len(score.parts)} voices "
                   f"({', '.join(p.partName or '?' for p in score.parts)}).")
        section("Live: the first events in the score")
        rows = []
        for el in list(score.recurse())[:24]:
            rows.append({"offset (beat)": round(float(el.offset), 2),
                         "type": el.__class__.__name__, "value": str(el)[:44]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)
        with st.expander("How to read this table"):
            st.markdown(
                "- **offset (beat):** when the object happens, in beats from the start. `0.0` is "
                "the very beginning.\n"
                "- **type:** what kind of object it is. The first rows are setup (`Instrument`, "
                "`Clef`, `KeySignature`, `TimeSignature`). After that, the real `Note`, `Chord` "
                "and `Rest` objects begin.\n"
                "- **value:** the content, such as a pitch `C5`, a key `f# minor`, or a time "
                "signature `4/4`.")
        callout("concept", "Which Bach?",
                "The book trains on the J.S. Bach <b>cello suites</b>. This app uses the Bach "
                "<b>chorales</b> that ship with music21, so everything runs with no download. "
                "The pipeline is exactly the same; only the data changes.")
        section("Live: hear the top line", "At each moment we keep only the highest note: one "
                                           "melody, one note at a time.")
        st.audio(synth_wav(top_line(score, 48)))
        with st.expander("What exactly is playing?"):
            st.markdown(
                "- The chorale has **four voices** singing at once.\n"
                "- We keep just the **top note at each step**, which gives one melodic line "
                "(**monophonic**).\n"
                "- That is on purpose: the Part 1 Transformer learns a single line first, so this "
                "is the signal it trains on.\n"
                "- The plain tone can sound beepy. We are showing the *notes*, not a real "
                "instrument.")
        section("Optional: sheet music",
                "Drawing notation needs MuseScore installed on the machine running this app. "
                "Everything else works without it.")
        mpath = st.text_input("MuseScore executable path (optional)", value=guess_mscore(),
                              placeholder=r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")
        if st.button("Render the first bars"):
            if not mpath:
                st.info("MuseScore was not found, so sheet music is off. Install it from "
                        "musescore.org and enter its path, or use the MuseScore cell in Colab.")
            else:
                with st.spinner("Rendering with MuseScore..."):
                    try:
                        from music21 import environment
                        us = environment.UserSettings()
                        us["musescoreDirectPNGPath"] = mpath
                        us["musicxmlPath"] = mpath
                        fp = score.measures(1, 4).write("musicxml.png")
                        st.image(str(fp), caption="First four bars, rendered by MuseScore")
                    except Exception as ex:
                        st.warning(f"Could not render notation ({ex}). Check the path, or install "
                                   "MuseScore from musescore.org.")
    except Exception as e:
        st.warning(f"Live parse unavailable here ({e}). The Colab code below still works.")
    section("Colab code")
    colab_code(C["parse"], "Parse every score into notes and durations")
    footer_nav(page_parse)


def page_streams():
    hero("Two streams: notes and durations",
         "Split the music into two lists that line up: what plays, and for how long.",
         [P1, "Step 2 of 15", "LIVE DEMO"])
    lead("Every musical event carries **two facts**: which note, and how long it lasts. So we keep "
         "**two lists that line up** position by position.")
    points(["**Notes:** the pitch at each step, or `REST` for silence.",
            "**Durations:** how many beats each event lasts.",
            "The key and time signature go at the front with duration `0.0`, because they are "
            "labels, not sounds."])
    callout("example", "Example",
            "G3 for 1 beat, then silence for 1 beat, then A3 for 2 beats:<br><br>"
            "<b>notes</b> = [G3, REST, A3] &nbsp;&nbsp; <b>durations</b> = [1.0, 1.0, 2.0]"
            "<br><br>Read the two lists column by column and you get the music back.")
    try:
        score = load_score()
        notes, durs = to_streams(score, limit=40)
        section("Live: the two streams from a real chorale")
        st.markdown("**notes**"); st.code("  ".join(notes[:36]) + " ...", language=None)
        st.markdown("**durations**"); st.code("  ".join(durs[:36]) + " ...", language=None)
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    section("Colab code")
    colab_code(C["streams"], "Build the two streams")
    footer_nav(page_streams)


def page_tokenize():
    hero("Tokenization: two vocabularies",
         "Give every symbol an integer ID: one dictionary for notes, one for durations.",
         [P1, "Step 3 of 15", "LIVE DEMO"])
    lead("A neural network only understands **numbers**, not text like `G3`. So we build a "
         "dictionary that gives every distinct symbol its own integer ID.")
    points(["We build it **twice**: one vocabulary for notes, one for durations.",
            "IDs `0` and `1` are reserved for padding and unknown tokens (`[UNK]`).",
            "The rest are ordered by **frequency**: the most common token gets ID `2`.",
            "The numbers are just labels. The model learns what they mean from the data."])
    callout("example", "Example",
            "A text tokenizer turns 'the cat sat' into [5, 8, 2]. Our music tokenizer turns "
            "[G3, REST, A3] into something like [17, 6, 25].")
    try:
        score = load_score()
        notes, durs = to_streams(score)
        from collections import Counter
        # same order as TextVectorization: padding, [UNK], then most frequent first
        nvocab = ["", "[UNK]"] + [t for t, _ in Counter(notes).most_common()]
        dvocab = ["", "[UNK]"] + [t for t, _ in Counter(durs).most_common()]
        nid = {t: i for i, t in enumerate(nvocab)}
        section("Live: note IDs from a real chorale")
        cc1, cc2 = st.columns(2)
        cc1.metric("note vocabulary", len(nvocab))
        cc2.metric("duration vocabulary", len(dvocab))
        demo = notes[:12]
        st.dataframe(pd.DataFrame({"note token": demo, "id": [nid[t] for t in demo]}),
                     use_container_width=True, height=300)
        st.caption("This table uses one chorale. Colab builds the vocabulary from the whole "
                   "corpus, so the IDs there will differ.")
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    section("Colab code")
    colab_code(C["tokenize"], "Build the two vocabularies")
    footer_nav(page_tokenize)


def page_window():
    hero("Sliding windows",
         "Cut each piece into many small 'predict the next event' examples.",
         [P1, "Step 4 of 15", "LIVE DEMO"])
    lead("Pieces have different lengths, but the model needs inputs of **one fixed size**. So we "
         "slide a window across each piece and cut out many training examples.")
    points(["**Window size:** how many events the model sees at once. The book uses 50.",
            "**Stride:** how far the window moves each time. Here, 1 event.",
            "**Overlap:** events shared by neighboring windows = window size − stride.",
            "**Target:** the same window shifted one step to the right, so the model practices a "
            "next-event guess at every position."], "Three choices you make in any sequence project")
    callout("example", "Example",
            "From G3 D3 B3 A3 B3 with a window of 3:<br><br>"
            "input <b>[G3 D3 B3]</b> &rarr; target <b>[D3 B3 A3]</b><br>"
            "slide by one: input <b>[D3 B3 A3]</b> &rarr; target <b>[B3 A3 B3]</b><br><br>"
            "The note list and the duration list slide together, so they stay aligned.")
    try:
        score = load_score()
        notes, durs = to_streams(score)
        rows = [{"window (input)": " ".join(notes[i:i + 6]),
                 "target (shifted by one)": " ".join(notes[i + 1:i + 7]),
                 "next event": notes[i + 6]} for i in range(3)]
        section("Live: windows from a real chorale")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    section("Colab code")
    colab_code(C["window"], "Cut windows and build the dataset")
    footer_nav(page_window)


def page_posenc():
    hero("Position encoding",
         "Stamp every slot with a 'where am I' signal, so order matters.",
         [P1, "Step 5 of 15", "LIVE DEMO"])
    lead("A Transformer looks at every token in the window **at once**, so on its own it has no "
         "sense of order. Position encoding adds a signal that says **where** each token sits.")
    points(["G3 B3 D3 and D3 B3 G3 use the same notes. Only the **order** differs.",
            "**Sinusoidal:** sine and cosine waves of different speeds give each slot its own "
            "pattern. Nothing extra to train.",
            "It is **added** to each stream's embedding, so it has the same size: 128 numbers.",
            "The embedding says **what** the note is. The position says **where** it is."])
    section("Live: the sinusoidal position signal",
            "Each column is one position, each row one number of the signal. Pick a row to see "
            "its wave.")
    dim = st.slider("highlight embedding dimension", 0, 31, 4)
    pos, D = 64, 32
    pe = np.zeros((pos, D))
    for p in range(pos):
        for i in range(D):
            denom = np.power(10000, (2 * (i // 2)) / D)
            pe[p, i] = np.sin(p / denom) if i % 2 == 0 else np.cos(p / denom)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.imshow(pe.T, aspect="auto", cmap="RdBu_r", origin="lower")
        ax.axhline(dim, color="black", lw=2)
        ax.set_xlabel("position"); ax.set_ylabel("embedding dimension")
        st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.4))
        ax2.plot(pe[:, dim], color=INDIGO, lw=2)
        ax2.set_title(f"dimension {dim}"); ax2.set_xlabel("position")
        st.pyplot(fig2)
    st.caption("Low dimensions wiggle fast, high dimensions change slowly. Together they give "
               "every position a unique fingerprint.")
    section("Colab code")
    colab_code(C["posenc"], "Token + position embedding layer")
    footer_nav(page_posenc)


def page_model():
    hero("The model: what goes in, what comes out",
         "Two token streams in, two 'what comes next' predictions out.",
         [P1, "Step 6 of 15", "MODEL + TRAINING"])
    lead("This is the heart of Part 1. The most important thing to be clear about is the "
         "**input** and the **output**.")
    io([("Input", "two windows of IDs: notes and durations, `[B, 50]` each"),
        ("Decoder", "embed, add position, join to `[B, 50, 256]`, causal attention"),
        ("Output", "next note `[B, 50, 59]` and next duration `[B, 50, 24]`")])
    ci, co = st.columns(2)
    with ci:
        section("Input")
        st.markdown("Two aligned windows of token **IDs**:")
        st.code("notes: [START, G3, D3,  B3,  A3]   ->  ids [2, 17, 12, 9, 25]\n"
                "durs:  [0.0,   1.0, 0.5, 1.0, 2.0] ->  ids [7, 4, 2, 4, 5]", language=None)
        st.caption("Example IDs, matching the lecture slides. Real IDs come from the vocabulary "
                   "built in Colab.")
    with co:
        section("Output")
        st.markdown("At every position, **two probability lists**: next note and next duration.")
        fig, ax = plt.subplots(figsize=(4.2, 1.9))
        ax.bar(["G3", "A3", "B3", "C4", "D4"], [0.08, 0.55, 0.12, 0.18, 0.07], color=INDIGO)
        ax.set_title("P(next note)", fontsize=10); ax.set_ylim(0, 0.7)
        ax.tick_params(labelsize=8); fig.tight_layout()
        st.pyplot(fig)
        st.caption("Here the model is fairly sure the next note is A3.")
    points(["Each stream gets its own **embedding** (128 numbers) plus **position**.",
            "The two are **joined**: 128 + 128 = 256 numbers per event.",
            "A **causal** Transformer block mixes information, looking only at the past.",
            "The same 256 numbers go to **two output heads**: one picks the next note, one the "
            "next duration. The loss is note loss + duration loss."], "Inside the model")
    section("The architecture")
    st.graphviz_chart("""
    digraph { rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef2ff",
              color="#3E5BD6", fontname=Helvetica];
      n[label="Note IDs [B,50]"]; d[label="Duration IDs [B,50]"];
      ne[label="Note embedding + position\\n[B,50,128]"]; de[label="Duration embedding + position\\n[B,50,128]"];
      cat[label="Concatenate [B,50,256]", fillcolor="#f3ecfd", color="#7B3FB0"];
      tb[label="Causal Transformer block", fillcolor="#fff6dd", color="#E8952F"];
      no[label="Note head -> [B,50,59]"]; do[label="Duration head -> [B,50,24]", fillcolor="#fff1e2", color="#E8952F"];
      n->ne; d->de; ne->cat; de->cat; cat->tb; tb->no; tb->do; }
    """)
    callout("concept", "Where training happens",
            "The last line of the code, <code>model.fit(train_ds, epochs=20)</code>, is the "
            "training step. It adjusts the weights so the predicted next note and duration match "
            "the real next event in Bach. Run it on a GPU in Colab.")
    section("Colab code")
    colab_code(C["model"], "Build and train the Transformer")
    footer_nav(page_model)


def page_generate():
    hero("Generate new music",
         "Feed the music so far, sample the next event, append it, and repeat.",
         [P1, "Step 7 of 15", "LIVE DEMO"])
    lead("Training is done. Now we **use** the model to compose. This is called **inference**.")
    points(["Feed the events so far, starting from `START`.",
            "The model returns the next-note and next-duration probabilities.",
            "**Sample** one note and one duration, append them, and repeat.",
            "Stop when the piece is long enough."], "The generation loop")
    section("Live: temperature, the creativity dial",
            "Low temperature plays it safe and repeats itself. High temperature takes risks. Drag "
            "the slider and watch the probabilities change.")
    labels = ["G", "A", "Bb", "C", "D", "Eb", "F", "F#"]
    logits = np.array([3.1, 1.2, 0.4, 2.6, 1.9, 0.7, 1.4, -1.5])
    temp = st.slider("temperature", 0.1, 2.0, 1.0, 0.05)
    probs = np.exp(logits / temp); probs /= probs.sum()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(labels, probs, color=INDIGO)
    ax.set_ylabel("probability"); ax.set_ylim(0, 1); ax.set_title("next-note probabilities")
    st.pyplot(fig)
    if st.button("Sample a note"):
        pick = int(np.random.choice(len(labels), p=probs))
        st.info(f"sampled: {labels[pick]}")
    st.caption("Real generation from the trained model runs in Colab with the code below.")
    section("Colab code")
    colab_code(C["generate"], "Generate with the trained model")
    footer_nav(page_generate)


def page_analysis():
    hero("Look inside: attention and keys",
         "Trained only to predict the next event, the model learns music theory by itself.",
         [P1, "Step 8 of 15", "THE PAYOFF"])
    lead("Nobody told the model about keys or scales. It only learned to **predict the next "
         "event**. Yet two heatmaps show it learned real musical structure.")
    callout("key", "It discovered the key",
            "In the next-note heatmap, whole pitch rows stay blank: the model avoids notes "
            "outside the key, a rule it was never told.")
    callout("key", "It learned where to look",
            "In the attention map, early tokens lean on the key and time signature, and later "
            "tokens watch recent notes and repeated patterns.")
    points(["Run **Generate** first, so there is a sequence to analyze.",
            "The code draws both heatmaps side by side."], "How to use the code", "green")
    section("Colab code")
    colab_code(C["analysis"], "Draw the attention and next-note heatmaps")
    footer_nav(page_analysis)


def page_polyphony():
    hero("Polyphony: several notes at once",
         "Real music stacks notes. Two ways to flatten several voices into one sequence.",
         [P1, "Step 9 of 15", "LIVE DEMO"])
    lead("So far the music was **one line** (monophonic). Real music often has **several notes at "
         "once** (polyphonic). A Transformer reads one list of tokens, so we must **flatten** "
         "the voices into one sequence, like flattening an image into one long vector.")
    points(["**Grid:** chop time into fixed steps and write every voice at each step.",
            "**Event:** write instructions: note on, note off, move the clock forward.",
            "Either way, one musical moment can take **several tokens**."], "Two ways to flatten")
    callout("example", "Tiny example",
            "Two voices, C and E, both held for one beat, then both stop.<br><br>"
            "<b>Grid:</b> &nbsp; [C, E] , [C, E]<br>"
            "<b>Event:</b> &nbsp; NOTE_ON C, NOTE_ON E, TIME_SHIFT 1 beat, NOTE_OFF C, NOTE_OFF E")
    section("Live: the same music, two tokenizations")
    tab1, tab2 = st.tabs(["Grid tokenization", "Event tokenization"])
    with tab1:
        st.markdown("Chop time into fixed steps. At each step, write every voice in a fixed order "
                    "(soprano, alto, tenor, bass). Read it like a spreadsheet, one row at a time.")
        st.dataframe(pd.DataFrame(
            {"S": [74, 74, 75], "A": [70, 70, 70], "T": [65, 65, 60], "B": [58, 55, 55]}),
            use_container_width=True)
        st.caption("Simple and regular, but rigid about fine timing.")
    with tab2:
        st.markdown("Write the music as instructions: turn a note on, turn it off, wait a bit. "
                    "Like the messages a keyboard sends.")
        st.code("NOTE_ON<74>  NOTE_ON<70>  NOTE_ON<65>  NOTE_ON<58>\n"
                "TIME_SHIFT<1.0>\nNOTE_OFF<58>  NOTE_ON<55>\nTIME_SHIFT<0.5>", language=None)
        st.caption("Flexible (exact timing, triplets, loudness), but sequences get longer.")
    callout("concept", "What changes in the model",
            "Every event type shares <b>one vocabulary</b>, so the model needs only <b>one output "
            "head</b>: Linear 256 &rarr; V, where V is the number of event types.")
    section("Colab code")
    colab_code(C["polyphony"], "Event tokenization")
    footer_nav(page_polyphony)


def page_musegan_data():
    hero("The piano-roll tensor",
         "A new view: draw the score as an image, then use convolution tools.",
         [P2, "Step 10 of 15", "LIVE DEMO"])
    lead("MuseGAN does not read tokens. It draws the music on a **grid**: pitch up the side, time "
         "across, and a 1 wherever a note sounds. That grid is an **image**, so convolutions work "
         "on it.")
    points(["**Each cell is 0 or 1**, not a brightness from 0 to 255 like a photo.",
            "A note held for one beat = **four 1s in a row** (4 steps per beat).",
            "**Each voice is its own layer**, like the R, G, B layers of a photo.",
            "Before training, 0 becomes −1 and 1 becomes +1, to match the generator's tanh output."],
           "How to read the grid")
    section("Live: a real chorale as a piano roll")
    try:
        st.pyplot(piano_roll_fig(load_score()))
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    callout("concept", "One training example",
            "2 bars &times; 16 steps per bar &times; 84 pitches &times; 4 voices "
            "&nbsp;&rarr;&nbsp; <code>[batch, 2, 16, 84, 4]</code>")
    section("Colab code")
    colab_code(C["musegan_data"], "Build the piano-roll training tensor")
    footer_nav(page_musegan_data)


def page_temporal():
    hero("The temporal network",
         "Turn one random vector into one plan per bar, so the bars belong together.",
         [P2, "Step 11 of 15", "MODEL"])
    lead("A plain GAN turns one random vector into one output. A piece has **several bars** that "
         "should relate. The temporal network turns **one seed** into **one vector per bar**.")
    points(["It is a **small part of the generator**, not a separate model, and draws no music.",
            "Built from **transposed convolutions** that stretch the seed along time.",
            "Used for **chords** (one network) and **melody** (one per voice).",
            "Style and groove skip it, because they stay the same in every bar.",
            "It has **no loss of its own**: it learns with the rest of the generator."])
    callout("example", "Example (4 numbers instead of the book's 32)",
            "chords seed [0.8, −1.2, 0.3, 0.5] &rarr; bar 1 [0.9, −0.7, 0.2, 1.1] and "
            "bar 2 [0.4, −1.0, 0.6, −0.3]. Different, but related, because both come from one "
            "seed. Heard as music: G major, then C major.")
    section("Colab code")
    colab_code(C["temporal"], "Temporal network")
    footer_nav(page_temporal)


def page_bargen():
    hero("The bar generator",
         "Paint one bar for one voice, then repeat it everywhere.",
         [P2, "Step 12 of 15", "MODEL"])
    lead("The bar generator does one small job: **paint a single bar for a single voice**.")
    points(["Its input joins **four pieces**: chords for this bar, style, this voice's melody for "
            "this bar, and this voice's groove.",
            "4 × 32 = **128 numbers** in the book.",
            "It grows them into a grid, first along **time** (2 → 16 steps), then along **pitch** "
            "(→ 84 rows).",
            "There is **one bar generator per voice**, each with its own weights."],
           "What goes in and out")
    callout("example", "Analogy",
            "Like a printer filling in one tile of a mosaic. Run it for every bar and every voice "
            "to build the whole score.")
    section("Colab code")
    colab_code(C["bargen"], "Bar generator")
    footer_nav(page_bargen)


def page_generator():
    hero("The full generator",
         "Run the bar generator for every bar and every voice, then stack the results.",
         [P2, "Step 13 of 15", "MODEL"])
    lead("Now we put the pieces together. The full generator is the **temporal networks plus the "
         "four bar generators**.")
    points(["**Style:** one vector, copied to every voice and every bar.",
            "**Chords:** one per bar from the temporal network, shared by all voices.",
            "**Groove:** one per voice, used in every bar of that voice.",
            "**Melody:** one per voice per bar, from that voice's temporal network.",
            "Output: one full piano roll, `[2, 16, 84, 4]`."], "The four noise inputs")
    callout("concept", "Why keep four separate inputs?",
            "They are all random noise. What makes them different is <b>where each one is "
            "sent</b>. That wiring forces each input to take a job, and lets you change one "
            "thing at a time, like a new harmony with the same voices.")
    section("Colab code")
    colab_code(C["generator"], "Full generator")
    footer_nav(page_generator)


def page_critic():
    hero("The critic",
         "The judge: one realism score for a whole piano roll.",
         [P2, "Step 14 of 15", "MODEL"])
    lead("The critic looks at a whole piano roll, real Bach or a fake, and outputs **one number**: "
         "how real does this look?")
    points(["It is a **CNN** that reads the full `[bars, steps, pitches, voices]` block.",
            "It outputs a **score**, not a yes/no probability (WGAN-GP).",
            "It uses **no batch normalization**, which WGAN-GP requires.",
            "The generator learns only from this score."])
    callout("example", "Analogy",
            "An art expert scoring how convincingly a forgery imitates the master.")
    section("Colab code")
    colab_code(C["critic"], "Critic")
    footer_nav(page_critic)


def page_wgan():
    hero("WGAN-GP training",
         "The adversarial game from your GAN lecture, now with music.",
         [P2, "Step 15 of 15", "TRAINING"])
    lead("The critic and the generator **take turns**. The critic learns to tell real from fake; "
         "the generator learns to fool it.")
    points(["**Critic:** scores real music high and fakes low. It updates its weights **5 times**; "
            "the generator is frozen.",
            "**Generator:** changes so its fakes score higher. **1 update**, with its gradient "
            "flowing back through the frozen critic.",
            "**Gradient penalty:** keeps the critic smooth, so its advice stays useful.",
            "**Repeat.** The fakes slowly start to look like Bach."], "One training round")
    callout("key", "At the end",
            "The code samples a new 2-bar chorale and reads the notes back out of the piano roll.")
    section("Colab code")
    colab_code(C["wgan"], "WGAN-GP training loop")
    footer_nav(page_wgan)


def page_notebook():
    hero("The whole notebook",
         "Every code cell from every page, in order, in one file.",
         ["Take it to Colab", "FULL CODE"])
    points(["Download the file, or open the panel below and copy it.",
            "Paste it into Colab, cell by cell, with a **GPU** runtime.",
            "Run Part 1 first, then Part 2."], "How to use it", "green")
    st.download_button("⬇  Download the whole notebook (.py)", FULL_NOTEBOOK,
                       file_name="chapter11_music.py", mime="text/x-python")
    n = FULL_NOTEBOOK.count("\n") + 1
    with st.expander(f"💻  Full notebook  ·  {n} lines  ·  click to open", expanded=False):
        st.code(FULL_NOTEBOOK, language="python")
    footer_nav(page_notebook)


# ============================================================ navigation
try:
    st.logo("logo.png", size="large")
except Exception:
    pass

with st.sidebar:
    st.markdown('<div class="sb-brand"><div class="sb-brand-t">🎵 Music Gen Lab</div>'
                '<div class="sb-brand-s">Transformer + MuseGAN · Chapter 11</div></div>',
                unsafe_allow_html=True)

p_home = st.Page(page_home, title="Home", icon=":material/home:", default=True)
p_setup = st.Page(page_setup, title="Setup and data", icon=":material/download:")
p_parse = st.Page(page_parse, title="1. Parse MIDI", icon=":material/music_note:")
p_streams = st.Page(page_streams, title="2. Two streams", icon=":material/list:")
p_tokenize = st.Page(page_tokenize, title="3. Tokenization", icon=":material/tag:")
p_window = st.Page(page_window, title="4. Sliding windows", icon=":material/view_column:")
p_posenc = st.Page(page_posenc, title="5. Position encoding", icon=":material/waves:")
p_model = st.Page(page_model, title="6. Model + training", icon=":material/account_tree:")
p_generate = st.Page(page_generate, title="7. Generate", icon=":material/casino:")
p_analysis = st.Page(page_analysis, title="8. Look inside", icon=":material/insights:")
p_polyphony = st.Page(page_polyphony, title="9. Polyphony", icon=":material/queue_music:")
p_mdata = st.Page(page_musegan_data, title="10. Piano-roll tensor", icon=":material/grid_on:")
p_temporal = st.Page(page_temporal, title="11. Temporal network", icon=":material/schedule:")
p_bargen = st.Page(page_bargen, title="12. Bar generator", icon=":material/brush:")
p_gen = st.Page(page_generator, title="13. Full generator", icon=":material/layers:")
p_critic = st.Page(page_critic, title="14. Critic", icon=":material/gavel:")
p_wgan = st.Page(page_wgan, title="15. WGAN-GP training", icon=":material/model_training:")
p_notebook = st.Page(page_notebook, title="Full notebook", icon=":material/menu_book:")

ORDER_PAGES = [p_home, p_setup, p_parse, p_streams, p_tokenize, p_window, p_posenc, p_model,
               p_generate, p_analysis, p_polyphony, p_mdata, p_temporal, p_bargen, p_gen,
               p_critic, p_wgan, p_notebook]
ORDER_FUNCS = [page_home, page_setup, page_parse, page_streams, page_tokenize, page_window,
               page_posenc, page_model, page_generate, page_analysis, page_polyphony,
               page_musegan_data, page_temporal, page_bargen, page_generator, page_critic,
               page_wgan, page_notebook]

nav = st.navigation({
    "Start here": [p_home, p_setup],
    "Part 1 · Transformer": [p_parse, p_streams, p_tokenize, p_window, p_posenc, p_model,
                             p_generate, p_analysis, p_polyphony],
    "Part 2 · MuseGAN": [p_mdata, p_temporal, p_bargen, p_gen, p_critic, p_wgan],
    "Take it to Colab": [p_notebook],
})
nav.run()
