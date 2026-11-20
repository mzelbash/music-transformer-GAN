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

/* ---------- home ---------- */
.home-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:24px 26px;
             border-top:6px solid var(--brand); margin-bottom:10px; }
.home-card.gan { border-top-color:#E8952F; }
.home-card.gan .path-k { color:#c67614; }
.mini-seq { display:flex; flex-wrap:wrap; gap:6px; margin-top:16px; }
.mini-chip { border:1.5px solid var(--brand); background:#eef2ff; border-radius:8px; padding:4px 9px;
             font-weight:700; font-size:.85rem; color:var(--ink); }
.mini-grid { display:grid; grid-template-columns:repeat(16, 1fr); gap:3px; margin-top:16px; }
.mini-cell { height:16px; border-radius:3px; background:#eef0f5; }
.mini-cell.on1 { background:#43A047; } .mini-cell.on2 { background:#F0851A; }
.mini-cell.on3 { background:#7B4FB8; } .mini-cell.on4 { background:#2F7FD6; }
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


# ============================================================ demo helpers
import base64 as _b64
import copy as _copy
import hashlib as _hashlib
import json as _json
import tempfile as _tempfile
from pathlib import Path as _Path

ASSETS = _Path(__file__).resolve().parent / "assets"
VOICE_COLORS = ["#43A047", "#F0851A", "#7B4FB8", "#2F7FD6", "#E23B3B", "#00897B"]
DEFAULT_PIECE = "bwv66.6"


@st.cache_resource(show_spinner=False)
def bach_catalog():
    """Bach scores that ship with music21: {name: path}."""
    try:
        from music21 import corpus
        out = {}
        for p in corpus.getComposer("bach"):
            stem = _Path(str(p)).stem
            if stem.startswith("bwv"):
                out[stem] = str(p)
        return dict(sorted(out.items(), key=lambda kv: [int(x) if x.isdigit() else x
                                                        for x in _re.split(r"(\d+)", kv[0])]))
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def parse_corpus(path):
    from music21 import corpus
    return corpus.parse(path)


@st.cache_resource(show_spinner=False)
def parse_upload(data, suffix):
    from music21 import converter
    with _tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        name = f.name
    return converter.parse(name)


def current_piece():
    """(label, score) for the piece chosen in the sidebar."""
    up = st.session_state.get("upload")
    if up is not None:
        data = up.getvalue()
        return up.name, parse_upload(data, _Path(up.name).suffix or ".mid")
    cat = bach_catalog()
    name = st.session_state.get("piece", DEFAULT_PIECE)
    if name in cat:
        return f"Bach {name.upper().replace('BWV', 'BWV ')}", parse_corpus(cat[name])
    return "Bach BWV 66.6", load_score()


def part_names(score):
    return [p.partName or f"Voice {i + 1}" for i, p in enumerate(score.parts)]


def score_midi(s):
    from music21 import midi
    return midi.translate.streamToMidiFile(s).writestr()


def parts_midi(score, idx):
    from music21 import stream
    sc = stream.Score()
    for i in idx:
        sc.insert(0, _copy.deepcopy(score.parts[i]))
    return score_midi(sc)


def tokens_midi(notes, durs):
    from fractions import Fraction
    from music21 import stream, note
    s = stream.Stream()
    for n, d in zip(notes, durs):
        try:
            ql = float(Fraction(d))
        except (ValueError, ZeroDivisionError):
            continue
        if ql <= 0:
            continue
        try:
            s.append(note.Rest(quarterLength=ql) if n == "REST" else note.Note(n, quarterLength=ql))
        except Exception:
            continue
    return score_midi(s)


def midi_player(data, key, roll=True):
    """Piano sound plus a scrolling piano roll, played in the browser (html-midi-player)."""
    b64 = _b64.b64encode(data).decode()
    vid = "v" + _hashlib.md5((key + b64[:64]).encode()).hexdigest()[:8]
    vis = (f'<midi-visualizer type="piano-roll" id="{vid}" '
           f'style="display:block;background:#fff;border:1px solid #e4e8f1;border-radius:10px;'
           f'margin-top:8px;overflow:auto;"></midi-visualizer>') if roll else ""
    vattr = f'visualizer="#{vid}"' if roll else ""
    doc = (
        '<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/'
        'es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>'
        '<style>body{margin:0;font-family:Inter,sans-serif;} midi-player{display:block;width:100%;}'
        'midi-player::part(control-panel){background:#eef2ff;border-radius:10px;}</style>'
        f'<midi-player src="data:audio/midi;base64,{b64}" sound-font {vattr}></midi-player>{vis}')
    components.html(doc, height=300 if roll else 70, scrolling=False)


def chips(tokens, color="#3E5BD6", fill="#eef2ff", sub=None, sub_color="#E8952F", hl=None):
    """A row of token chips; optional second line under each chip."""
    cells = []
    for i, t in enumerate(tokens):
        on = hl is None or i in hl
        style = (f"border:1.5px solid {color};background:{fill};color:#1B2333;" if on else
                 "border:1.5px dashed #d0d5e2;background:#fff;color:#b0b6c4;")
        top = (f'<div style="{style}border-radius:8px;padding:5px 6px;font-weight:700;'
               f'font-size:.86rem;text-align:center;white-space:nowrap;">{_html.escape(str(t))}</div>')
        bot = ""
        if sub is not None:
            bot = (f'<div style="border:1.5px solid {sub_color};border-radius:8px;padding:4px 6px;'
                   f'margin-top:5px;font-size:.84rem;text-align:center;font-weight:600;'
                   f'color:#1B2333;background:#fff;">{_html.escape(str(sub[i]))}</div>')
        cells.append(f'<div style="min-width:58px;">{top}{bot}</div>')
    st.markdown(f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 10px;">'
                f'{"".join(cells)}</div>', unsafe_allow_html=True)


def note_roll(score, idx=None, beats=None, title=None):
    """Matplotlib piano roll of the chosen voices."""
    idx = list(range(len(score.parts))) if idx is None else idx
    fig, ax = plt.subplots(figsize=(10, 3.4))
    lo, hi, end = 127, 0, 0
    for i in idx:
        for n in score.parts[i].flatten().notes:
            on, ql = float(n.offset), float(n.duration.quarterLength)
            if beats and on >= beats:
                continue
            for p in n.pitches:
                ax.add_patch(Rectangle((on, p.midi - 0.42), ql, 0.84,
                                       color=VOICE_COLORS[i % 6], ec="white", lw=0.4))
                lo, hi = min(lo, p.midi), max(hi, p.midi)
            end = max(end, on + ql)
    if hi:
        ax.set_ylim(lo - 2, hi + 2)
    ax.set_xlim(0, beats or end or 1)
    ax.set_xlabel("time (beats)"); ax.set_ylabel("pitch (MIDI number)")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def asset(name):
    p = ASSETS / name
    return p if p.exists() else None


def need_assets(what):
    st.info(f"{what} appear here after you run the **export** cell in Colab and upload the "
            "`assets` folder to the app's GitHub repo.", icon=":material/upload:")


def note_events(score):
    """[(part, on, off, pitch_name)] for every note in every part."""
    ev = []
    for i, part in enumerate(score.parts):
        for n in part.flatten().notes:
            on = float(n.offset); off = on + float(n.duration.quarterLength)
            ev.append((i, on, off, n.pitches[-1].nameWithOctave))
    return ev


def grid_and_events(score, start, beats=4.0, steps_per_beat=4):
    ev = note_events(score)
    nparts = len(score.parts)
    grid = []
    for k in range(int(beats * steps_per_beat)):
        t = start + k / steps_per_beat
        row = []
        for i in range(nparts):
            hit = [p for (pi, on, off, p) in ev if pi == i and on <= t < off]
            row.append(hit[0] if hit else "–")
        grid.append(row)
    end = start + beats
    win = [(on if on > start else start, off if off < end else end, p) for (pi, on, off, p) in ev
           if off > start and on < end]
    times = sorted({t for a, b, _ in win for t in (a, b)})
    toks = []
    for j, t in enumerate(times):
        toks += [f"NOTE_OFF<{p}>" for a, b, p in win if b == t]
        toks += [f"NOTE_ON<{p}>" for a, b, p in win if a == t]
        if j + 1 < len(times):
            toks.append(f"TIME_SHIFT<{times[j + 1] - t:g}>")
    return grid, toks


def roll_array(score, bars=2, steps=16, lowest=24, pitches=84):
    """[time, pitch, voice] 0/1 array for the first bars, like the MuseGAN input."""
    T = bars * steps
    nv = len(score.parts)
    arr = np.zeros((T, pitches, nv), dtype=int)
    spb = steps / 4
    for i, part in enumerate(score.parts):
        for n in part.flatten().notes:
            a = int(round(float(n.offset) * spb))
            b = int(round((float(n.offset) + float(n.duration.quarterLength)) * spb))
            row = n.pitches[-1].midi - lowest
            if 0 <= row < pitches and a < T:
                arr[a:min(b, T), row, i] = 1
    return arr


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
history = model.fit(train_ds, epochs=20)
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

C["export_transformer"] = '''
# Save outputs for the lab app: generated MIDI, attention weights and the loss curve
import os, json
from fractions import Fraction
from music21 import stream, note
os.makedirs("assets", exist_ok=True)

def tokens_to_midi(notes, durs, path):
    s = stream.Stream()
    for n, d in zip(notes, durs):
        try:
            ql = float(Fraction(d))
        except (ValueError, ZeroDivisionError):
            continue
        if ql <= 0:
            continue                                  # START, key and time-signature labels
        if n == "REST":
            s.append(note.Rest(quarterLength=ql))
        else:
            try:
                s.append(note.Note(n, quarterLength=ql))
            except Exception:
                continue                              # skip [UNK] and other labels
    s.write("midi", fp=path)

for t in [0.5, 1.0, 1.5]:
    n, d = generate(max_tokens=120, temperature=t)
    tokens_to_midi(n, d, f"assets/transformer_t{t}.mid")

L = min(SEQ_LEN, len(gen_notes))
att = att_model.predict([notes_vec([" ".join(gen_notes[:L])]),
                         durs_vec([" ".join(gen_durs[:L])])], verbose=0)[0].mean(axis=0)
np.savez("assets/attention.npz", att=att, tokens=np.array(gen_notes[:L]))
json.dump({"loss": [float(v) for v in history.history["loss"]]},
          open("assets/transformer_loss.json", "w"))

!zip -qr assets.zip assets
from google.colab import files
files.download("assets.zip")   # unzip and upload the assets folder to the app's GitHub repo
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

c_hist, g_hist = [], []
for epoch in range(EPOCHS):
    for real in train_data:
        c_loss, g_loss = train_step(real)
    c_hist.append(float(c_loss)); g_hist.append(float(g_loss))
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}   critic loss {float(c_loss):8.3f}   generator loss {float(g_loss):8.3f}")

# Generate: for every step and voice, the brightest pitch row is the note
sample = generator(rand_z(1), training=False)[0].numpy()     # [2, 16, 84, 4]
midi_notes = sample.argmax(axis=2) + LOWEST                    # [2, 16, 4]: S A T B per step
print(midi_notes[0, :8])                                       # first 8 steps of bar 1
'''

C["export_musegan"] = '''
# Save outputs for the lab app: generated chorales, losses and critic scores
import os, json
from music21 import stream, note
os.makedirs("assets", exist_ok=True)
STEP = 4 / N_STEPS                       # 16 steps per 4-beat bar = 0.25 beat per step

def roll_to_midi(roll, path):
    """roll [bars, steps, pitches, voices] -> a four-part score. Repeated pitches merge."""
    score = stream.Score()
    pitches = roll.argmax(axis=2).reshape(-1, N_TRACKS)    # [time, voices]
    for v in range(N_TRACKS):
        part, seq, t = stream.Part(), pitches[:, v], 0
        while t < len(seq):
            u = t
            while u + 1 < len(seq) and seq[u + 1] == seq[t]:
                u += 1
            part.append(note.Note(int(seq[t]) + LOWEST, quarterLength=(u - t + 1) * STEP))
            t = u + 1
        score.insert(0, part)
    score.write("midi", fp=path)

samples = generator(rand_z(4), training=False).numpy()
for i, roll in enumerate(samples):
    roll_to_midi(roll, f"assets/musegan_sample_{i + 1}.mid")

real = data[:64].astype("float32")
fake = generator(rand_z(64), training=False)
json.dump({"critic_loss": c_hist, "generator_loss": g_hist,
           "critic_real": critic(real).numpy().ravel().tolist(),
           "critic_fake": critic(fake).numpy().ravel().tolist()},
          open("assets/musegan_stats.json", "w"))

!zip -qr assets.zip assets
from google.colab import files
files.download("assets.zip")   # unzip and upload the assets folder to the app's GitHub repo
'''

FULL_NOTEBOOK = "\n\n".join(f"# ===== {k} =====\n{v.strip()}" for k, v in C.items())


# ============================================================ pages
P1, P2 = "Part 1 · Transformer", "Part 2 · MuseGAN"


def piece_line(label, score):
    names = part_names(score)
    st.markdown(f'<div class="sec-sub">Now using <b>{_html.escape(label)}</b> · '
                f'{len(names)} voices · change it in the sidebar</div>', unsafe_allow_html=True)


def page_home():
    st.markdown('<div class="ptitle" style="margin-top:8px;">Music Generation Code Lab</div>'
                '<div class="psub">Two ways to teach a computer to write Bach.</div>'
                '<div class="phr"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    seq = "".join(f'<span class="mini-chip">{t}</span>' for t in
                  ["START", "G3", "D3", "B3", "A3", "B3", "D3", "…"])
    cells = "".join(f'<span class="mini-cell{" on" + str(v) if v else ""}"></span>'
                    for v in [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0])
    with c1:
        st.markdown('<div class="home-card"><div class="path-k">Part 1</div>'
                    '<div class="path-t">Transformer</div>'
                    '<div class="path-s">Music as a <b>sequence</b>: predict the next note.</div>'
                    f'<div class="mini-seq">{seq}</div></div>', unsafe_allow_html=True)
        st.page_link(p_parse, label="Start Part 1", icon=":material/arrow_forward:")
    with c2:
        st.markdown('<div class="home-card gan"><div class="path-k">Part 2</div>'
                    '<div class="path-t">MuseGAN</div>'
                    '<div class="path-s">Music as an <b>image</b>: draw the whole piano roll.</div>'
                    f'<div class="mini-grid">{cells}</div></div>', unsafe_allow_html=True)
        st.page_link(p_mdata, label="Start Part 2", icon=":material/arrow_forward:")
    footer_nav(page_home)


def page_setup():
    hero("Setup", "Run these cells first in a Colab notebook with a GPU.",
         ["Start here", "Setup", "COLAB SETUP"])
    colab_code(C["setup"], "1. Install and import")
    colab_code(C["download"], "2. Load the Bach scores (no download needed)")
    colab_code(C["musescore"], "3. Optional: sheet music with MuseScore")
    footer_nav(page_setup)


def page_parse():
    hero("Parse MIDI", "A music file becomes a list of timed events.",
         [P1, "Step 1 of 15", "LIVE DEMO"])
    try:
        label, score = current_piece()
        piece_line(label, score)
        names = part_names(score)
        pick = st.multiselect("Voices to play", names, default=names)
        idx = [names.index(n) for n in pick] or [0]
        midi_player(parts_midi(score, idx), "parse")
        st.pyplot(note_roll(score, idx))
        with st.expander("The events music21 sees"):
            n = st.slider("How many events", 10, 120, 30)
            rows = [{"offset (beat)": round(float(el.offset), 2), "type": el.__class__.__name__,
                     "value": str(el)[:44]} for el in list(score.recurse())[:n]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)
    except Exception as e:
        st.warning(f"Live demo unavailable here ({e}).")
    colab_code(C["parse"], "Parse every score into notes and durations")
    footer_nav(page_parse)


def page_streams():
    hero("Two streams", "Every event becomes a note and a duration, side by side.",
         [P1, "Step 2 of 15", "LIVE DEMO"])
    try:
        label, score = current_piece()
        piece_line(label, score)
        notes, durs = to_streams(score)
        n = st.slider("Events to show", 6, 30, 14)
        st.markdown("**notes** (top) and **durations** (bottom)")
        chips(notes[:n], sub=durs[:n])
        st.markdown("**Play the melody these two lists describe**")
        midi_player(tokens_midi(notes[:120], durs[:120]), "streams")
    except Exception as e:
        st.warning(f"Live demo unavailable here ({e}).")
    colab_code(C["streams"], "Build the two streams")
    footer_nav(page_streams)


def page_tokenize():
    hero("Tokenization", "Every symbol gets an integer ID. One vocabulary for notes, one for "
                         "durations.", [P1, "Step 3 of 15", "LIVE DEMO"])
    try:
        label, score = current_piece()
        piece_line(label, score)
        notes, durs = to_streams(score)
        from collections import Counter
        nvocab = ["", "[UNK]"] + [t for t, _ in Counter(notes).most_common()]
        dvocab = ["", "[UNK]"] + [t for t, _ in Counter(durs).most_common()]
        nid = {t: i for i, t in enumerate(nvocab)}
        did = {t: i for i, t in enumerate(dvocab)}
        m1, m2 = st.columns(2)
        m1.metric("note vocabulary", len(nvocab))
        m2.metric("duration vocabulary", len(dvocab))
        n = st.slider("Events to show", 6, 24, 12)
        st.markdown("**note → ID**")
        chips(notes[:n], sub=[nid[t] for t in notes[:n]])
        st.markdown("**duration → ID**")
        chips(durs[:n], color="#E8952F", fill="#fff4e6", sub=[did[t] for t in durs[:n]],
              sub_color="#3E5BD6")
        top = Counter(notes).most_common(15)
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.bar([t for t, _ in top], [c for _, c in top], color=INDIGO)
        ax.set_title("Most common note tokens (they get the smallest IDs)", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        fig.tight_layout(); st.pyplot(fig)
        st.caption("IDs 0 and 1 are reserved for padding and [UNK]. Colab builds the vocabulary "
                   "from all pieces, so its IDs differ from this one piece.")
    except Exception as e:
        st.warning(f"Live demo unavailable here ({e}).")
    colab_code(C["tokenize"], "Build the two vocabularies")
    footer_nav(page_tokenize)


def page_window():
    hero("Sliding windows", "Cut each piece into fixed-size training examples.",
         [P1, "Step 4 of 15", "LIVE DEMO"])
    try:
        label, score = current_piece()
        piece_line(label, score)
        notes, _ = to_streams(score)
        c1, c2 = st.columns(2)
        W = c1.slider("Window size", 2, 8, 4)
        S = c2.slider("Stride", 1, W, 1)
        n_win = max(0, (len(notes) - 1 - W) // S + 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("events in this piece", len(notes))
        m2.metric("training windows", n_win)
        m3.metric("overlap (window − stride)", max(W - S, 0))
        span = min(len(notes), 3 * S + W + 1)
        st.markdown("**The piece**")
        chips(notes[:span], color="#1B2333", fill="#fff")
        for k in range(min(3, n_win)):
            a = k * S
            st.markdown(f"**Window {k + 1}** · input (blue) and target (green, shifted by one)")
            chips(notes[:span], hl=set(range(a, a + W)))
            chips(notes[:span], color="#2E9E52", fill="#eafaf0", hl=set(range(a + 1, a + W + 1)))
    except Exception as e:
        st.warning(f"Live demo unavailable here ({e}).")
    colab_code(C["window"], "Cut windows and build the dataset")
    footer_nav(page_window)


def page_posenc():
    hero("Position encoding", "Each slot gets its own wave pattern, so order matters.",
         [P1, "Step 5 of 15", "LIVE DEMO"])
    pos, D = 64, 32
    pe = np.zeros((pos, D))
    for p in range(pos):
        for i in range(D):
            denom = np.power(10000, (2 * (i // 2)) / D)
            pe[p, i] = np.sin(p / denom) if i % 2 == 0 else np.cos(p / denom)
    c1, c2 = st.columns(2)
    a = c1.slider("Position A", 0, pos - 1, 2)
    b = c2.slider("Position B", 0, pos - 1, 38)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.imshow(pe.T, aspect="auto", cmap="RdBu_r", origin="lower")
    ax1.axvline(a, color="#E8952F", lw=2.5); ax1.axvline(b, color="#2E9E52", lw=2.5)
    ax1.set_xlabel("position"); ax1.set_ylabel("dimension"); ax1.set_title("the position table")
    ax2.plot(pe[a], color="#E8952F", lw=2, label=f"position {a}")
    ax2.plot(pe[b], color="#2E9E52", lw=2, label=f"position {b}")
    ax2.set_xlabel("dimension"); ax2.set_title("same note, two positions: different signals")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)
    colab_code(C["posenc"], "Token + position embedding layer")
    footer_nav(page_posenc)


def page_model():
    hero("The model", "Two token streams in, two predictions out.",
         [P1, "Step 6 of 15", "MODEL + TRAINING"])
    c1, c2 = st.columns(2)
    B = c1.slider("Batch size B", 1, 64, 32)
    L = c2.slider("Window length", 10, 100, 50)
    st.graphviz_chart(f"""
    digraph {{ rankdir=LR; node [shape=box, style="rounded,filled", fontname=Helvetica,
              fillcolor="#eef2ff", color="#3E5BD6"]; edge [color="#5b6474"];
      n [label="Note IDs\\n[{B}, {L}]"]; d [label="Duration IDs\\n[{B}, {L}]", fillcolor="#fff4e6", color="#E8952F"];
      ne [label="Embed + position\\n[{B}, {L}, 128]"]; de [label="Embed + position\\n[{B}, {L}, 128]", fillcolor="#fff4e6", color="#E8952F"];
      c [label="Concatenate\\n[{B}, {L}, 256]", fillcolor="#f3ecfd", color="#7B3FB0"];
      t [label="Causal decoder\\n[{B}, {L}, 256]", fillcolor="#fff6dd", color="#E8952F"];
      no [label="Note head\\n[{B}, {L}, 59]"]; do [label="Duration head\\n[{B}, {L}, 24]", fillcolor="#fff4e6", color="#E8952F"];
      n->ne; d->de; ne->c; de->c; c->t; t->no; t->do; }}""")
    loss = asset("transformer_loss.json")
    if loss:
        vals = _json.loads(loss.read_text())["loss"]
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(range(1, len(vals) + 1), vals, color=INDIGO, lw=2, marker="o")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training loss", fontsize=10)
        fig.tight_layout(); st.pyplot(fig)
    else:
        need_assets("The training-loss curve and generated music")
    colab_code(C["model"], "Build and train the Transformer")
    footer_nav(page_model)


def page_generate():
    hero("Generate", "Sample the next event, append it, repeat.",
         [P1, "Step 7 of 15", "LIVE DEMO"])
    labels = ["G", "A", "Bb", "C", "D", "Eb", "F", "F#"]
    logits = np.array([3.1, 1.2, 0.4, 2.6, 1.9, 0.7, 1.4, -1.5])
    temp = st.slider("Temperature", 0.1, 2.0, 1.0, 0.05)
    probs = np.exp(logits / temp); probs /= probs.sum()
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.bar(labels, probs, color=INDIGO)
    ax.set_ylim(0, 1); ax.set_title("next-note probabilities", fontsize=10)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)
    st.caption("Low temperature: safe and repetitive. High: adventurous.")
    section("Music the trained model wrote")
    found = False
    cols = st.columns(3)
    for col, t in zip(cols, ["0.5", "1.0", "1.5"]):
        f = asset(f"transformer_t{t}.mid")
        if f:
            found = True
            with col:
                st.markdown(f"**Temperature {t}**")
                midi_player(f.read_bytes(), f"gen{t}", roll=False)
    if not found:
        need_assets("Pieces written by the trained model")
    colab_code(C["generate"], "Generate with the trained model")
    colab_code(C["export_transformer"], "Export the results for this app")
    footer_nav(page_generate)


def page_analysis():
    hero("Look inside", "Where does each note look when it predicts the next one?",
         [P1, "Step 8 of 15", "THE PAYOFF"])
    f = asset("attention.npz")
    if f:
        z = np.load(f, allow_pickle=True)
        att, toks = z["att"], [str(t) for t in z["tokens"]]
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Heatmap(z=att, x=[f"{i}:{t}" for i, t in enumerate(toks)],
                                       y=[f"{i}:{t}" for i, t in enumerate(toks)],
                                       colorscale="Greens",
                                       hovertemplate="%{y} looks at %{x}<br>weight %{z:.2f}<extra></extra>"))
            fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            fig, ax = plt.subplots(figsize=(9, 8))
            ax.imshow(att, cmap="Greens")
            ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=90, fontsize=6)
            ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=6)
            st.pyplot(fig)
        st.caption("Row = the note making a prediction. Column = where it looked. Darker = more.")
    else:
        need_assets("The attention map")
    colab_code(C["analysis"], "Draw the attention and next-note heatmaps")
    footer_nav(page_analysis)


def page_polyphony():
    hero("Polyphony", "Several notes at once, flattened into one token sequence.",
         [P1, "Step 9 of 15", "LIVE DEMO"])
    try:
        label, score = current_piece()
        piece_line(label, score)
        total = float(score.highestTime)
        start = st.slider("Start beat (shows 4 beats)", 0, max(int(total) - 4, 0), 0, 1)
        grid, toks = grid_and_events(score, float(start))
        names = part_names(score)
        m1, m2 = st.columns(2)
        m1.metric("grid tokens", len(grid) * len(names))
        m2.metric("event tokens", len(toks))
        t1, t2 = st.tabs(["Grid: every voice at every step", "Event: on, off, wait"])
        with t1:
            st.dataframe(pd.DataFrame(grid, columns=names,
                                      index=[f"step {i + 1}" for i in range(len(grid))]),
                         use_container_width=True, height=420)
        with t2:
            def tone(t):
                return ("#3E5BD6", "#eef2ff") if t.startswith("NOTE_ON") else (
                    ("#E8952F", "#fff4e6") if t.startswith("NOTE_OFF") else ("#2E9E52", "#eafaf0"))
            for chunk in [toks[i:i + 8] for i in range(0, len(toks), 8)]:
                c = [tone(t) for t in chunk]
                st.markdown('<div style="display:flex;gap:6px;flex-wrap:wrap;margin:4px 0;">' +
                            "".join(f'<span style="border:1.5px solid {a};background:{b};'
                                    f'border-radius:8px;padding:4px 8px;font-size:.82rem;'
                                    f'font-weight:700;">{_html.escape(t)}</span>'
                                    for t, (a, b) in zip(chunk, c)) + "</div>",
                            unsafe_allow_html=True)
        midi_player(parts_midi(score, list(range(len(names)))), "poly", roll=False)
    except Exception as e:
        st.warning(f"Live demo unavailable here ({e}).")
    colab_code(C["polyphony"], "Event tokenization")
    footer_nav(page_polyphony)


def page_musegan_data():
    hero("The piano-roll tensor", "Each cell is 0 or 1. Each voice is its own layer.",
         [P2, "Step 10 of 15", "LIVE DEMO"])
    try:
        label, score = current_piece()
        piece_line(label, score)
        names = part_names(score)
        arr = roll_array(score)
        pick = st.multiselect("Voice layers", names, default=names)
        idx = [names.index(n) for n in pick] or [0]
        zoom = st.toggle("Zoom to the pitches used", value=True)
        show_n = st.toggle("Show the 0s and 1s", value=False)
        used = np.where(arr[:, :, idx].sum(axis=(0, 2)) > 0)[0]
        lo, hi = (used.min(), used.max()) if (zoom and len(used)) else (0, arr.shape[1] - 1)
        img = np.ones((hi - lo + 1, arr.shape[0], 3))
        from matplotlib.colors import to_rgb
        for i in idx:
            on = arr[:, lo:hi + 1, i].T > 0
            img[on] = to_rgb(VOICE_COLORS[i % 6])
        fig, ax = plt.subplots(figsize=(11, 4.2 if zoom else 6))
        ax.imshow(img[::-1], aspect="auto", interpolation="nearest",
                  extent=(0, arr.shape[0], lo + 24 - 0.5, hi + 24 + 0.5))
        ax.axvline(16, color="#1B2333", lw=2)
        ax.set_xticks(range(0, 33, 4)); ax.set_xlabel("time step (2 bars × 16)")
        ax.set_ylabel("pitch (MIDI number)")
        if show_n and zoom:
            for r in range(lo, hi + 1):
                for t in range(arr.shape[0]):
                    v = int(arr[t, r, idx].max())
                    ax.text(t + 0.5, r + 24, str(v), ha="center", va="center", fontsize=5.5,
                            color="white" if v else "#b0b6c4")
        fig.tight_layout(); st.pyplot(fig)
        m1, m2, m3 = st.columns(3)
        m1.metric("tensor shape", f"[2, 16, 84, {len(names)}]")
        m2.metric("cells that are 1", int(arr.sum()))
        m3.metric("share of 1s", f"{arr.sum() / arr.size:.1%}")
    except Exception as e:
        st.warning(f"Live demo unavailable here ({e}).")
    colab_code(C["musegan_data"], "Build the piano-roll training tensor")
    footer_nav(page_musegan_data)


def shape_flow(rows, color="#3E5BD6", fill="#eef2ff"):
    nodes = "".join(f'n{i} [label="{a}\\n{b}"];' for i, (a, b) in enumerate(rows))
    edges = "->".join(f"n{i}" for i in range(len(rows)))
    st.graphviz_chart(f'digraph {{ rankdir=LR; node [shape=box, style="rounded,filled", '
                      f'fontname=Helvetica, fillcolor="{fill}", color="{color}"]; '
                      f'edge [color="#5b6474"]; {nodes} {edges}; }}')


def samples_block():
    found = False
    cols = st.columns(4)
    for i, col in enumerate(cols, 1):
        f = asset(f"musegan_sample_{i}.mid")
        if f:
            found = True
            with col:
                st.markdown(f"**Sample {i}**")
                midi_player(f.read_bytes(), f"mg{i}", roll=False)
    if not found:
        need_assets("Chorales written by the trained MuseGAN")


def page_temporal():
    hero("Temporal network", "One random seed becomes one plan per bar.",
         [P2, "Step 11 of 15", "MODEL"])
    shape_flow([("one seed", "[32]"), ("reshape", "[1, 1, 32]"),
                ("transposed conv", "[2, 1, 1024]"), ("transposed conv", "[2, 1, 32]"),
                ("one plan per bar", "[2, 32]")], "#7B3FB0", "#f3ecfd")
    seed = st.slider("Random seed", 0, 50, 7)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=8)
    bars = np.stack([np.tanh(z * 0.9 + rng.normal(scale=.35, size=8)) for _ in range(2)])
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.2), sharey=True)
    for ax, v, t, c in zip(axes, [z, bars[0], bars[1]], ["seed", "bar 1", "bar 2"],
                           ["#5b6474", "#3E5BD6", "#2E9E52"]):
        ax.bar(range(8), v, color=c); ax.set_title(t, fontsize=10); ax.set_xticks([])
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)
    st.caption("Illustration with 8 numbers instead of 32: the two bar plans differ, but both "
               "come from the same seed.")
    colab_code(C["temporal"], "Temporal network")
    footer_nav(page_temporal)


def page_bargen():
    hero("Bar generator", "Paint one bar for one voice.", [P2, "Step 12 of 15", "MODEL"])
    shape_flow([("4 inputs joined", "[128]"), ("dense + reshape", "[2, 1, 512]"),
                ("grow time", "[16, 1, 256]"), ("grow pitch", "[16, 84, 1]"),
                ("one bar, one voice", "[1, 16, 84, 1]")])
    chips(["chords (this bar)", "style", "melody (this voice, this bar)", "groove (this voice)"],
          color="#7B3FB0", fill="#f3ecfd")
    st.caption("One bar generator per voice. Each runs once per bar.")
    colab_code(C["bargen"], "Bar generator")
    footer_nav(page_bargen)


def page_generator():
    hero("Full generator", "Every bar, every voice, stacked into one piano roll.",
         [P2, "Step 13 of 15", "MODEL"])
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box, style="rounded,filled", fontname=Helvetica];
      edge [color="#5b6474"];
      s [label="style", fillcolor="#e3eefb", color="#2F7FD6"];
      c [label="chords", fillcolor="#e8f5e9", color="#2E9E52"];
      m [label="melody x4", fillcolor="#fde7e7", color="#D63333"];
      g [label="groove x4", fillcolor="#fdebd3", color="#E8952F"];
      tc [label="temporal net", fillcolor="#f3ecfd", color="#7B3FB0"];
      tm [label="temporal nets x4", fillcolor="#f3ecfd", color="#7B3FB0"];
      bg [label="bar generators\\n4 voices x 2 bars", fillcolor="#eef2ff", color="#3E5BD6"];
      out [label="piano roll\\n[2, 16, 84, 4]", fillcolor="#fff6dd", color="#E8952F"];
      c->tc; m->tm; tc->bg; tm->bg; s->bg; g->bg; bg->out; }""")
    samples_block()
    colab_code(C["generator"], "Full generator")
    footer_nav(page_generator)


def page_critic():
    hero("Critic", "One realism score for a whole piano roll.", [P2, "Step 14 of 15", "MODEL"])
    shape_flow([("piano roll", "[2, 16, 84, 4]"), ("Conv3D layers", "time and pitch"),
                ("flatten + dense", ""), ("realism score", "[1]")], "#E8952F", "#fff4e6")
    f = asset("musegan_stats.json")
    if f:
        d = _json.loads(f.read_text())
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.hist(d["critic_real"], bins=20, alpha=.7, color="#2E9E52", label="real Bach")
        ax.hist(d["critic_fake"], bins=20, alpha=.7, color="#E8952F", label="generated")
        ax.legend(frameon=False); ax.set_xlabel("critic score"); ax.set_title("Critic scores")
        fig.tight_layout(); st.pyplot(fig)
    else:
        need_assets("Critic scores for real and generated music")
    colab_code(C["critic"], "Critic")
    footer_nav(page_critic)


def page_wgan():
    hero("WGAN-GP training", "Critic: 5 updates. Generator: 1 update. Repeat.",
         [P2, "Step 15 of 15", "TRAINING"])
    f = asset("musegan_stats.json")
    if f:
        d = _json.loads(f.read_text())
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.plot(d["critic_loss"], color="#E8952F", lw=2, label="critic loss")
        ax.plot(d["generator_loss"], color="#3E5BD6", lw=2, label="generator loss")
        ax.legend(frameon=False); ax.set_xlabel("epoch"); ax.set_title("Training")
        fig.tight_layout(); st.pyplot(fig)
    samples_block()
    colab_code(C["wgan"], "WGAN-GP training loop")
    colab_code(C["export_musegan"], "Export the results for this app")
    footer_nav(page_wgan)


def page_notebook():
    hero("Full notebook", "Every code cell, in order, in one file.",
         ["Take it to Colab", "FULL CODE"])
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
    cat = bach_catalog()
    if cat:
        keys = list(cat)
        st.selectbox("Bach piece", keys, index=keys.index(DEFAULT_PIECE) if DEFAULT_PIECE in keys
                     else 0, key="piece", format_func=lambda k: k.upper().replace("BWV", "BWV "))
    st.file_uploader("…or upload your own", type=["mid", "midi", "mxl", "musicxml", "xml"],
                     key="upload")

p_home = st.Page(page_home, title="Home", icon=":material/home:", default=True)
p_setup = st.Page(page_setup, title="Setup", icon=":material/download:")
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
