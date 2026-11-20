# music-transformer-GAN

Run it live: https://music-transformer-gan-7s28l5evy6cpwxbp3isjrn.streamlit.app/

A Streamlit code lab for Chapter 11 (music generation) of *Generative Deep Learning* by David Foster.
It covers both models from the chapter:

1. **Part 1, Transformer:** music as a sequence of tokens (notes and durations).
2. **Part 2, MuseGAN:** music as a multitrack piano-roll image.

Every page is a live demo with real Bach (pick any chorale in the sidebar, or upload your own MIDI),
plus the full Python code for that step in a collapsible panel, ready to paste into Colab.
Music plays in the browser with piano sound and a scrolling piano roll.
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

## Show the trained models in the app

The Generate, Look inside, Critic and WGAN-GP pages can play music the trained models wrote and show
their attention maps and losses. To fill them in:

1. In Colab, after training, run **Export the results for this app** (on the Generate page for the
   Transformer, and on the WGAN-GP page for MuseGAN). It downloads `assets.zip`.
2. Unzip it and upload the `assets` folder to the root of this repo.

The app picks up these files automatically:

| File | Used on |
|---|---|
| `assets/transformer_t0.5.mid`, `t1.0`, `t1.5` | Generate |
| `assets/attention.npz` | Look inside |
| `assets/transformer_loss.json` | Model + training |
| `assets/musegan_sample_1.mid` … `_4.mid` | Full generator, WGAN-GP |
| `assets/musegan_stats.json` | Critic, WGAN-GP |
