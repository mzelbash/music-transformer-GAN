# music-transformer-GAN

A Streamlit code lab for Chapter 11 (music generation) of *Generative Deep Learning* by David Foster.
It covers both models from the chapter:

1. **Part 1, Transformer:** music as a sequence of tokens (notes and durations).
2. **Part 2, MuseGAN:** music as a multitrack piano-roll image.

Every page has a small live demo that runs in the browser (parsing Bach, building tokens, the piano roll,
position encoding, temperature) and the full Python code for that step, ready to paste into Colab.
The heavy training runs in Colab on a GPU.

## Data

The book trains the Transformer on the J.S. Bach cello suites and MuseGAN on Bach chorales.
This app uses the Bach scores and chorales that ship with `music21`, so everything runs with no download.
The pipeline is the same.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run music_lab.py
```

## Run the code in Colab

1. Open the **Full notebook** page and download `chapter11_music.py`, or copy each page's code cell.
2. Use a GPU runtime.
3. Run the cells in order: Setup, Parse, Streams, Tokenization, Sliding windows, Position encoding,
   Model, Generate, Analysis. For MuseGAN: Piano-roll tensor, Temporal network, Bar generator,
   Full generator, Critic, WGAN-GP training.

## Shapes at a glance

1. Transformer input: two id windows of length 50 (notes and durations), each embedded to 128 and joined to 256.
2. MuseGAN data: `[batch, 2 bars, 16 steps, 84 pitches, 4 tracks]`.
