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

st.set_page_config(page_title="Chapter 11 Music Gen Lab", page_icon="🎵", layout="wide")

AMBER, INDIGO, GREEN, GREY, INK = "#E8952F", "#3E5BD6", "#2E9E52", "#7A8194", "#1B2333"

# ============================================================ theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{ --brand:#3E5BD6; --amber:#E8952F; --green:#2E9E52; --ink:#1B2333; --muted:#5b6474; }
html, body, [class*="css"], .stApp { font-family:'Inter', system-ui, -apple-system, sans-serif; }
.stApp { background:#f5f6fb; }
.block-container { padding-top:1.1rem; max-width:1090px; }

/* hero header */
.hero { border-radius:16px; padding:22px 28px; margin:2px 0 20px; color:#fff;
        box-shadow:0 10px 26px rgba(20,30,60,.14); }
.hero-indigo{ background:linear-gradient(120deg,#3E5BD6,#6d84ec); }
.hero-green { background:linear-gradient(120deg,#1f9d57,#38b76f); }
.hero-amber { background:linear-gradient(120deg,#df7d1c,#f0a63e); }
.hero-slate { background:linear-gradient(120deg,#2a3350,#48557d); }
.hero-tag { display:inline-block; background:rgba(255,255,255,.22); border-radius:999px;
            padding:3px 13px; font-size:.72rem; font-weight:700; letter-spacing:.07em; }
.hero-title { font-size:1.95rem; font-weight:800; margin:11px 0 4px; line-height:1.12; }
.hero-sub { font-size:1.03rem; opacity:.96; }

/* section subheader */
.sec { font-size:1.06rem; font-weight:700; color:var(--ink); margin:22px 0 8px;
       display:flex; align-items:center; gap:9px; }
.sec-bar { width:6px; height:19px; border-radius:3px; background:var(--brand); }

/* callouts */
.cal { border-radius:12px; padding:14px 16px 14px 18px; margin:12px 0; border-left:5px solid;
       background:#fff; box-shadow:0 1px 5px rgba(20,30,60,.05); }
.cal .cal-label { display:block; font-size:.7rem; font-weight:800; letter-spacing:.07em;
                  text-transform:uppercase; margin-bottom:6px; }
.cal .cal-body { font-size:.99rem; color:#2a3040; line-height:1.6; }
.cal.example { border-color:var(--amber); background:#fff8ef; }
.cal.example .cal-label { color:#c67614; }
.cal.concept { border-color:var(--brand); background:#eef2ff; }
.cal.concept .cal-label { color:var(--brand); }
.cal.key { border-color:var(--green); background:#eafaf0; }
.cal.key .cal-label { color:#1f8046; }
.cal code { background:rgba(0,0,0,.06); padding:1px 6px; border-radius:5px; font-size:.9em; }

/* feature cards (home) */
.feat { background:#fff; border:1px solid #e7ebf3; border-radius:14px; padding:18px 18px;
        box-shadow:0 2px 8px rgba(20,30,60,.05); height:100%; }
.feat-t { font-weight:700; font-size:1.02rem; color:var(--ink); margin-bottom:6px; }
.feat-b { font-size:.92rem; color:var(--muted); line-height:1.5; }

/* code label */
.codehead { font-size:.78rem; font-weight:800; color:var(--muted); text-transform:uppercase;
            letter-spacing:.06em; margin:16px 0 4px; }

/* widgets */
.stButton>button, .stDownloadButton>button { background:var(--brand); color:#fff; border:0;
        border-radius:9px; padding:8px 18px; font-weight:600; }
.stButton>button:hover, .stDownloadButton>button:hover { background:#3350c4; color:#fff; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e7ebf3; border-radius:12px;
        padding:14px 16px; box-shadow:0 1px 4px rgba(20,30,60,.04); }
[data-testid="stExpander"] { border:1px solid #e7ebf3; border-radius:12px; background:#fff; }
[data-testid="stExpander"] summary { font-weight:600; color:var(--ink); }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e7ebf3; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:#eef2ff; font-weight:700; }
hr { margin:1.1rem 0; }
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
            notes.append("rest"); durs.append(str(el.duration.quarterLength))
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
    ax.set_title("Piano roll: pitch x time, one colour per voice")
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
def hero(title, subtitle, tag, accent="indigo"):
    st.markdown(
        f'<div class="hero hero-{accent}"><div class="hero-tag">{tag}</div>'
        f'<div class="hero-title">{title}</div><div class="hero-sub">{subtitle}</div></div>',
        unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="sec"><span class="sec-bar"></span>{title}</div>',
                unsafe_allow_html=True)


def callout(kind, label, body_html):
    st.markdown(f'<div class="cal {kind}"><span class="cal-label">{label}</span>'
                f'<div class="cal-body">{body_html}</div></div>', unsafe_allow_html=True)


def feat(title, body):
    st.markdown(f'<div class="feat"><div class="feat-t">{title}</div>'
                f'<div class="feat-b">{body}</div></div>', unsafe_allow_html=True)


def text_block(md):
    """Explanatory text in its own clean card."""
    with st.container(border=True):
        st.markdown(md)


def colab_code(block):
    """Code in a dark Monokai 'editor' card with a working Copy button."""
    import html as _html
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    src = block.strip("\n")
    hl = highlight(src, PythonLexer(), HtmlFormatter(style="monokai", noclasses=True))
    esc = _html.escape(src)
    n = src.count("\n") + 1
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
!pip -q install music21 keras-nlp

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
            notes.append("rest");              durations.append(str(el.duration.quarterLength))
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

BATCH_SIZE = 256
def create_dataset(elements):
    ds = (tf.data.Dataset.from_tensor_slices(elements)
          .batch(BATCH_SIZE, drop_remainder=True)
          .shuffle(1000))
    vectorize = layers.TextVectorization(standardize=None, output_mode="int")
    vectorize.adapt(ds)
    return ds, vectorize, vectorize.get_vocabulary()

notes_ds, notes_vec, notes_vocab = create_dataset(notes_corpus)
durs_ds,  durs_vec,  durs_vocab  = create_dataset(durations_corpus)
seq_ds = tf.data.Dataset.zip((notes_ds, durs_ds))

print("note vocab:", len(notes_vocab), " duration vocab:", len(durs_vocab))
'''

C["window"] = '''
SEQ_LEN = 50

def make_training_set(seq_ds):
    def to_windows(notes, durs):
        # shift by one: input = first SEQ_LEN, target = same window shifted +1
        return ((notes[:-1], durs[:-1]), (notes[1:], durs[1:]))
    ds = (seq_ds
          .flat_map(lambda n, d: tf.data.Dataset.zip((
              tf.data.Dataset.from_tensor_slices(n).batch(SEQ_LEN + 1, drop_remainder=True),
              tf.data.Dataset.from_tensor_slices(d).batch(SEQ_LEN + 1, drop_remainder=True))))
          .map(to_windows)
          .batch(64))
    return ds

train_ds = make_training_set(seq_ds)
'''

C["posenc"] = '''
import keras_nlp
from tensorflow.keras import layers

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras_nlp.layers.SinePositionEncoding()   # deterministic, any length
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
# The model returns attention scores from the TransformerBlock.
# Feed a generated sequence back in and read where each prediction looked.
x1 = notes_vec([" ".join(gen_notes)])
x2 = durs_vec([" ".join(gen_durs)])
_, _ = model.predict([x1, x2], verbose=0)
# Build a sub-model that also outputs attention_scores to visualise the look-back,
# then plot it as a heatmap (rows = predicted token, cols = attended-to token).
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
N_BARS, N_STEPS, N_PITCHES, N_TRACKS = 2, 16, 84, 4

# The book uses 229 four-voice Bach chorales. music21 ships Bach chorales too, so you can
# build the training tensor offline, with no download.
from music21 import corpus
chorales = list(corpus.chorales.Iterator())     # ~370 four-voice chorales
print(len(chorales), "chorales available offline")

# For each chorale: keep 4 voices (S A T B), quantize to 16 steps per bar, one-hot the
# pitch to length 84, take 2-bar chunks -> arrays of shape [N_BARS, N_STEPS, N_PITCHES, N_TRACKS].
# Stack all chunks into the training tensor:
print("target training shape:", (None, N_BARS, N_STEPS, N_PITCHES, N_TRACKS))
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

# train critic several steps per generator step, add 10 * gradient_penalty to the critic loss.
'''

FULL_NOTEBOOK = "\n\n".join(f"# ===== {k} =====\n{v.strip()}" for k, v in C.items())


# ============================================================ pages
def page_home():
    hero("Music Generation Code Lab",
         "Live demos you can run in class, plus copy-paste Colab code for every step of Chapter 11.",
         "START HERE", "indigo")
    c1, c2, c3 = st.columns(3)
    with c1:
        feat("🎼 Live demos", "Parse real Bach, build tokens, visualize the piano roll, and hear "
             "audio, all in the browser.")
    with c2:
        feat("🧠 Full Colab code", "Every step has runnable Python with a copy button, ready to "
             "paste into Colab Pro.")
    with c3:
        feat("🎛️ Two paradigms", "Part 1 is the Transformer (music as a sequence). Part 2 is "
             "MuseGAN (music as an image).")
    section("How this app works")
    text_block("Every page has two layers: a **live demo** up top that runs real code here, and "
                "the **full Colab code** below for students to copy and run on a GPU. The heavy "
                "model training stays in Colab; everything lighter runs live in this app.")
    callout("concept", "Navigation",
            "Use the sidebar. <b>Part 1</b> is the Transformer, <b>Part 2</b> is MuseGAN. "
            "You can also use the Back / Next buttons at the bottom of each page.")
    section("Take the whole notebook")
    st.download_button("Download the whole notebook (.py)", FULL_NOTEBOOK,
                       file_name="chapter11_music.py", mime="text/x-python")
    footer_nav(page_home)


def page_setup():
    hero("Setup and data", "Install the libraries and load the Bach data. Run these first in Colab.",
         "COLAB SETUP", "slate")
    section("1. Install and import")
    colab_code(C["setup"])
    section("2. Get the data")
    st.markdown("music21 ships hundreds of Bach scores, so this needs no download.")
    colab_code(C["download"])
    section("3. Optional: graphical notation with MuseScore")
    st.markdown("music21 draws sheet music by calling MuseScore. Run this in Colab to make "
                "`score.show()` render real notation instead of only `show('text')`.")
    colab_code(C["musescore"])
    callout("concept", "If apt shows 404 errors",
            "That is Colab's stale package cache. The <code>apt-get update</code> line above fixes "
            "it. And <code>score.show('text')</code> always works with no install at all.")
    footer_nav(page_setup)


def page_parse():
    hero("1. Parse MIDI with music21",
         "Turn a MIDI file into structured events, the bridge from music to data.",
         "LIVE DEMO", "green")
    text_block(
        "A MIDI file is just instructions: which note, when it starts, how long it lasts. "
        "`music21` reads those instructions and hands us back **structured events** we can loop "
        "over in Python. The score stops being a picture and becomes a list of timed, typed "
        "objects.")
    try:
        score = load_score()
        st.success(f"Loaded a real Bach chorale with {len(score.parts)} voices "
                   f"({', '.join(p.partName or '?' for p in score.parts)}).")
        section("The structured events")
        rows = []
        for el in list(score.recurse())[:24]:
            rows.append({"offset (beat)": round(float(el.offset), 2),
                         "type": el.__class__.__name__, "value": str(el)[:44]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)
        with st.expander("How to read this table"):
            st.markdown(
                "- **offset (beat):** when the object happens, in quarter-note beats from the "
                "start. `0.0` is the very beginning.\n"
                "- **type:** what kind of object it is. The first rows are setup (`Instrument`, "
                "`Clef`, `KeySignature`, `TimeSignature`); after that the real `Note`, `Chord`, "
                "and `Rest` objects begin.\n"
                "- **value:** the content, for example a pitch like `C5`, a key like `f# minor`, "
                "or a time signature like `4/4`.")
        section("Hear the top line")
        st.audio(synth_wav(top_line(score, 48)))
        st.caption("The melody only: at each moment we keep the highest note, and synthesize the "
                   "first 48 notes with a simple tone.")
        with st.expander("What exactly is playing?"):
            st.markdown(
                "The chorale has four voices at once. Here we keep just the **top note at each "
                "step**, giving one melodic line, one note at a time (**monophonic**). That is on "
                "purpose: the Transformer in Part 1 learns a single line first, so this is the "
                "exact signal it will be trained on. The plain tone can sound beepy; we are "
                "showing the *notes*, not a real instrument.")
        section("Graphical notation (optional)")
        st.caption("music21 can draw real sheet music with `score.show()`, but it needs MuseScore "
                   "installed here. In Colab, use the MuseScore cell on the Setup page.")
        mpath = st.text_input("MuseScore executable path", value=guess_mscore(),
                              placeholder=r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")
        if st.button("Render the first bars"):
            if not mpath:
                st.warning("No MuseScore found. Install it from musescore.org, then enter the path.")
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
    colab_code(C["parse"])
    footer_nav(page_parse)


def page_streams():
    hero("2. Two streams: notes and durations",
         "Split the music into two aligned lists: what plays, and how long.", "LIVE DEMO", "green")
    text_block(
        "Every musical event carries **two facts**: which note, and how long it lasts. So we turn "
        "the parsed music into **two lists that line up** position by position. The key and time "
        "signature go at the front with duration 0.0, because they take no musical time.")
    callout("example", "Example",
            "G3 held for 1 beat, then silence for 1 beat, then A3 for 2 beats becomes:<br><br>"
            "<b>notes</b> = [G3, rest, A3] &nbsp; and &nbsp; <b>durations</b> = [1.0, 1.0, 2.0]"
            "<br><br>Read the two lists together, column by column, and you get the music back.")
    try:
        score = load_score()
        notes, durs = to_streams(score, limit=40)
        section("Live: the two streams from a real chorale")
        st.markdown("**notes**"); st.code("  ".join(notes[:36]) + " ...", language=None)
        st.markdown("**durations**"); st.code("  ".join(durs[:36]) + " ...", language=None)
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    section("Colab code")
    colab_code(C["streams"])
    footer_nav(page_streams)


def page_tokenize():
    hero("3. Tokenization: two vocabularies",
         "Give every symbol an integer id, one dictionary for notes, one for durations.",
         "LIVE DEMO", "green")
    text_block(
        "A neural network only understands **numbers**, not text like `G3`. So we build a "
        "dictionary that gives every distinct symbol its own integer id. We do this **twice**: "
        "one dictionary for the notes, one for the durations, and keep the two lists zipped "
        "together.")
    callout("example", "Example",
            "Just like a text tokenizer turns 'the cat sat' into [5, 8, 2], our music tokenizer "
            "turns [G3, rest, A3] into something like [2, 6, 3]. The numbers are just labels; the "
            "model learns what they mean from the data.")
    try:
        score = load_score()
        notes, durs = to_streams(score)
        nvocab = ["", "[UNK]"] + sorted(set(notes))
        dvocab = ["", "[UNK]"] + sorted(set(durs))
        nid = {t: i for i, t in enumerate(nvocab)}
        section("Live: note ids from a real chorale")
        cc1, cc2 = st.columns(2)
        cc1.metric("note vocabulary", len(nvocab))
        cc2.metric("duration vocabulary", len(dvocab))
        demo = notes[:12]
        st.dataframe(pd.DataFrame({"note token": demo, "id": [nid[t] for t in demo]}),
                     use_container_width=True, height=300)
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    section("Colab code")
    colab_code(C["tokenize"])
    footer_nav(page_tokenize)


def page_window():
    hero("4. Sliding-window training set",
         "Make many little 'predict the next event' examples by sliding a window.",
         "LIVE DEMO", "green")
    text_block(
        "We train the model to **predict the next event**. To make training examples, we slide a "
        "fixed-size window across the sequence: the window is the **input**, and the very next "
        "event is the **answer** we want it to predict.")
    callout("example", "Example",
            "Think of flashcards. From G3 D3 B3 A3 B3 with a window of 3:<br><br>"
            "input <b>[G3 D3 B3]</b> &rarr; answer <b>A3</b>, &nbsp; then slide: &nbsp; "
            "input <b>[D3 B3 A3]</b> &rarr; answer <b>B3</b>.<br><br>"
            "The note list and duration list slide together, so they stay aligned.")
    try:
        score = load_score()
        notes, durs = to_streams(score)
        rows = [{"window (input)": " ".join(notes[i:i + 6]), "predict next": notes[i + 6]}
                for i in range(3)]
        section("Live: windows from a real chorale")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    section("Colab code")
    colab_code(C["window"])
    footer_nav(page_window)


def page_posenc():
    hero("5. Position encoding",
         "Stamp each slot with a 'where am I' signal so order matters.", "LIVE DEMO", "green")
    text_block(
        "A Transformer looks at all the tokens in the window **at once**, so by itself it has no "
        "sense of order. Position encoding stamps each slot with a unique 'where am I in the "
        "sequence' signal, so the model can tell the first token from the last.")
    callout("example", "Why it matters",
            "The notes G, A, B going up and B, A, G going down use the exact same three symbols. "
            "Only the ORDER is different. Position encoding is what lets the model hear that "
            "difference.")
    section("Live: the sinusoidal position signal")
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
    section("Colab code")
    colab_code(C["posenc"])
    footer_nav(page_posenc)


def page_model():
    hero("6. The model: what goes in, what comes out",
         "Two token streams in, two 'what comes next' distributions out. Training lives here.",
         "MODEL + TRAINING", "indigo")
    text_block("This is the heart of Part 1. The single most important thing to be clear about is "
                "the **input** and the **output** of the Transformer.")
    ci, co = st.columns(2)
    with ci:
        section("Input")
        st.markdown("Two aligned windows of token **ids**: the notes, and the durations.")
        st.code("notes: [START, G3, D3, B3, A3]   ->  ids [1, 2, 4, 9, 3]\n"
                "durs:  [0.0,   1.0, 1.0, 1.0, 1.0] ->  ids [10, 2, 2, 2, 2]", language=None)
    with co:
        section("Output")
        st.markdown("Per position, **two probability distributions**: next note, next duration.")
        fig, ax = plt.subplots(figsize=(4.2, 1.9))
        ax.bar(["G3", "A3", "B3", "C4", "D4"], [0.08, 0.55, 0.12, 0.18, 0.07], color=INDIGO)
        ax.set_title("P(next note)", fontsize=10); ax.set_ylim(0, 0.7)
        ax.tick_params(labelsize=8); fig.tight_layout()
        st.pyplot(fig)
        st.caption("Here the model is fairly sure the next note is A3.")
    callout("key", "In to Out",
            "<b>Two token streams go in, two 'what comes next' distributions come out.</b> Between "
            "them sits one shared causal Transformer block, which can only look at the past, never "
            "the future.")
    section("The architecture")
    st.graphviz_chart("""
    digraph { rankdir=TB; node [shape=box, style=rounded, fontname=Helvetica];
      n[label="Note input"]; d[label="Duration input"];
      ne[label="Token + position\\nembedding"]; de[label="Token + position\\nembedding"];
      cat[label="Concatenate [b,l,256]"]; tb[label="Causal Transformer block"];
      no[label="Dense -> next NOTE [b,l,59]"]; do[label="Dense -> next DURATION [b,l,24]"];
      n->ne; d->de; ne->cat; de->cat; cat->tb; tb->no; tb->do; }
    """)
    callout("concept", "Where training happens",
            "The last line of the code below, <code>model.fit(train_ds, epochs=20)</code>, is the "
            "training step. It adjusts the weights so the predicted next note and duration match "
            "the real next event in Bach. Run it on a GPU in Colab.")
    section("Colab code")
    colab_code(C["model"])
    footer_nav(page_model)


def page_generate():
    hero("7. Generate: using the trained model",
         "Inference: feed the music so far, sample the next event, append, and repeat.",
         "LIVE DEMO - inference", "green")
    text_block(
        "Training is finished; now we **use** the model to compose. This is **inference**. We feed "
        "the events so far, the model returns the two 'what comes next' distributions, we "
        "**sample** one note and one duration, append them, and repeat.")
    callout("key", "In to Out",
            "<b>Input</b> = the music so far. &nbsp; <b>Output</b> = next-note + next-duration "
            "distributions. We pick from those, add the choice to the sequence, and loop until the "
            "piece is long enough.")
    section("Live: temperature, the creativity dial")
    st.markdown("Low temperature is safe and repetitive; high is adventurous. Drag it and watch "
                "the distribution change.")
    labels = ["G", "A", "Bb", "C", "D", "Eb", "F", "F#"]
    logits = np.array([3.1, 1.2, 0.4, 2.6, 1.9, 0.7, 1.4, -1.5])
    temp = st.slider("temperature", 0.1, 2.0, 1.0, 0.05)
    probs = np.exp(logits / temp); probs /= probs.sum()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(labels, probs, color=INDIGO)
    ax.set_ylabel("probability"); ax.set_ylim(0, 1); ax.set_title("next-note distribution")
    st.pyplot(fig)
    if st.button("Sample a note"):
        pick = int(np.random.choice(len(labels), p=probs))
        st.success(f"sampled: {labels[pick]}")
    st.caption("Real generation from the trained model runs in Colab (code below), or from a small "
               "pre-trained model we can ship with this app next.")
    section("Colab code")
    colab_code(C["generate"])
    footer_nav(page_generate)


def page_analysis():
    hero("8. Analysis: attention and keys",
         "Trained only to predict the next event, the model learns real music theory by itself.",
         "PAYOFF", "amber")
    text_block("Here is the surprising part. Trained only to **predict the next event**, the model "
                "quietly learns structure that nobody programmed. Two things to look at in Colab:")
    callout("key", "It discovered the key",
            "In the note-distribution heatmap, whole pitch rows stay blank: the model avoids "
            "out-of-key notes it was never told about.")
    callout("key", "It learned where to look",
            "In the attention map, early tokens lean on the key and time signature, and later "
            "tokens watch recent notes and repeated patterns.")
    section("Colab code")
    colab_code(C["analysis"])
    footer_nav(page_analysis)


def page_polyphony():
    hero("9. Polyphony: grid vs event",
         "Real music stacks notes. Two ways to flatten several voices into one sequence.",
         "LIVE DEMO", "green")
    text_block(
        "Everything so far was **one line** of music (monophonic). But real music often has "
        "**several notes at once** (polyphonic), like a chord or four singers. A Transformer reads "
        "a single list of tokens, so we must **flatten** several simultaneous lines into one "
        "sequence. There are two common ways.")
    callout("example", "Tiny example",
            "Two voices, C and E, both held for one beat, then both stop.<br><br>"
            "<b>Grid:</b> &nbsp; [C, E] , [C, E] &nbsp; (fixed time steps; each step lists every "
            "voice)<br><b>Event:</b> &nbsp; NOTE_ON C, NOTE_ON E, TIME_SHIFT 1 beat, NOTE_OFF C, "
            "NOTE_OFF E &nbsp; (a recipe of instructions)")
    section("Live: the same music, two tokenizations")
    tab1, tab2 = st.tabs(["Grid tokenization", "Event tokenization"])
    with tab1:
        st.markdown("**Grid.** Chop time into fixed steps. At each step, write down every voice in "
                    "a fixed order (soprano, alto, tenor, bass), then move on. Read it like a "
                    "spreadsheet, one column at a time.")
        st.dataframe(pd.DataFrame(
            {"S": [74, 74, 75], "A": [70, 70, 70], "T": [65, 65, 60], "B": [58, 55, 55]}),
            use_container_width=True)
        st.caption("Simple and regular, but rigid on fine timing.")
    with tab2:
        st.markdown("**Event.** Write the music as instructions: turn a note on, turn it off, wait "
                    "a bit. Like the messages a keyboard sends, or holes punched in a player-piano "
                    "roll.")
        st.code("NOTE_ON<74>  NOTE_ON<70>  NOTE_ON<65>  NOTE_ON<58>\n"
                "TIME_SHIFT<1.0>\nNOTE_OFF<58>  NOTE_ON<55>\nTIME_SHIFT<0.5>", language=None)
        st.caption("Flexible (exact timing, triplets, dynamics), but the sequences get longer.")
    section("Colab code")
    colab_code(C["polyphony"])
    footer_nav(page_polyphony)


def page_musegan_data():
    hero("10. Piano-roll tensor (MuseGAN)",
         "A new view: draw the score as an image, then use convolution tools.", "LIVE DEMO", "green")
    text_block(
        "MuseGAN takes a totally different view. Instead of a list of tokens, it draws the music on "
        "a **grid**: pitch up the side, time across the bottom, a filled cell wherever a note "
        "sounds. That grid is basically an **image**, so we can use the same convolution tools we "
        "use for pictures.")
    callout("example", "Picture it",
            "A spreadsheet: rows are pitches, columns are time, a cell is 1 if that pitch is "
            "playing then. Four voices = four stacked sheets (channels), like the R, G, B channels "
            "of a colour image.")
    section("Live: a real chorale as a piano roll")
    try:
        st.pyplot(piano_roll_fig(load_score()))
    except Exception as e:
        st.warning(f"Live view unavailable here ({e}).")
    callout("concept", "Data shape",
            "bars = 2 &nbsp;·&nbsp; steps = 16 &nbsp;·&nbsp; pitches = 84 &nbsp;·&nbsp; tracks = 4 "
            "(S A T B) &nbsp; &rarr; &nbsp; <code>[batch, 2, 16, 84, 4]</code>")
    section("Colab code")
    colab_code(C["musegan_data"])
    footer_nav(page_musegan_data)


def page_temporal():
    hero("11. The temporal network",
         "Turn one random vector into one plan per bar, so bars connect.", "MODEL", "indigo")
    text_block(
        "A GAN generator normally turns one random vector into one output. But a piece has several "
        "**bars** that should relate to each other. The temporal network takes one random vector "
        "and expands it into **one vector per bar**, so the bars connect instead of being random "
        "and independent.")
    callout("example", "Analogy",
            "Like planning a short story: one seed idea is expanded into a plan for chapter 1 and "
            "chapter 2 that both follow from the same idea.")
    section("Colab code")
    colab_code(C["temporal"])
    footer_nav(page_temporal)


def page_bargen():
    hero("12. The bar generator", "Paint a single bar for a single voice, then repeat everywhere.",
         "MODEL", "indigo")
    text_block(
        "The bar generator does one small job: **paint a single bar for a single voice**. It takes "
        "the latent pieces for that bar and voice and grows them, first along time, then along "
        "pitch, into a small piano-roll patch.")
    callout("example", "Analogy",
            "Like a printer filling in one tile of the final picture. Run it for every bar and "
            "every voice to build the whole score.")
    section("Colab code")
    colab_code(C["bargen"])
    footer_nav(page_bargen)


def page_generator():
    hero("13. The full generator", "Repeat the small painter across every bar and every voice.",
         "MODEL", "indigo")
    text_block(
        "Now we repeat that small painter **everywhere**. For each bar and each voice, gather the "
        "right latent pieces (chords, style, melody, groove), run the bar generator, and stack the "
        "results into the full multi-voice score.")
    section("Colab code")
    colab_code(C["generator"])
    footer_nav(page_generator)


def page_critic():
    hero("14. The critic", "The judge: one realness score for a whole score.", "MODEL", "indigo")
    text_block(
        "The critic is the **judge**. It looks at a whole score (real Bach, or a fake from the "
        "generator) and outputs a single number: how real does this look? The generator tries to "
        "earn high scores; the critic tries to tell real from fake.")
    callout("example", "Analogy",
            "Like an art expert scoring how convincingly a forgery imitates the master.")
    st.markdown("It uses **Conv3D** over the `[bars, steps, pitches, tracks]` block, with no batch "
                "normalization (a WGAN-GP requirement).")
    section("Colab code")
    colab_code(C["critic"])
    footer_nav(page_critic)


def page_wgan():
    hero("15. WGAN-GP training", "The adversarial game, exactly like your GAN lecture.",
         "TRAINING", "indigo")
    text_block(
        "This is the training loop for the whole GAN, and it is the **same adversarial game** as "
        "your GAN lecture. The critic learns to score real above fake, the generator learns to "
        "fool the critic, and a **gradient penalty** keeps the critic well-behaved.")
    section("Colab code")
    colab_code(C["wgan"])
    footer_nav(page_wgan)


def page_notebook():
    hero("Take it to Colab", "Grab the whole notebook and paste it into Colab Pro cells.",
         "FULL CODE", "slate")
    st.download_button("Download the whole notebook (.py)", FULL_NOTEBOOK,
                       file_name="chapter11_music.py", mime="text/x-python")
    st.code(FULL_NOTEBOOK, language="python")
    footer_nav(page_notebook)


# ============================================================ navigation
try:
    st.logo("logo.png", size="large")
except Exception:
    pass

p_home = st.Page(page_home, title="Home", icon=":material/home:", default=True)
p_setup = st.Page(page_setup, title="Setup and data", icon=":material/download:")
p_parse = st.Page(page_parse, title="1. Parse MIDI", icon=":material/music_note:")
p_streams = st.Page(page_streams, title="2. Two streams", icon=":material/list:")
p_tokenize = st.Page(page_tokenize, title="3. Tokenization", icon=":material/tag:")
p_window = st.Page(page_window, title="4. Sliding windows", icon=":material/view_column:")
p_posenc = st.Page(page_posenc, title="5. Position encoding", icon=":material/waves:")
p_model = st.Page(page_model, title="6. Model + training", icon=":material/account_tree:")
p_generate = st.Page(page_generate, title="7. Generate (inference)", icon=":material/casino:")
p_analysis = st.Page(page_analysis, title="8. Analysis", icon=":material/insights:")
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
    "Part 1  ·  Transformer": [p_parse, p_streams, p_tokenize, p_window, p_posenc, p_model,
                               p_generate, p_analysis, p_polyphony],
    "Part 2  ·  MuseGAN": [p_mdata, p_temporal, p_bargen, p_gen, p_critic, p_wgan],
    "Take it to Colab": [p_notebook],
})
nav.run()
