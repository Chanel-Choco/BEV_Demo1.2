"""
Streamlit proof-of-concept: upload an artwork -> P(AI-generated) from Model A, Model B and their ensemble,
plus optional Grad-CAM and SHAP explanations. Run with:   streamlit run app.py
Matches the v14 notebooks (LFAB v4).
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from inference import (BAND_LABELS, DEVICE, gradcam_overlay, lfab_band_shap, lfab_gain_map, lfab_gains,
                       load_models, pixel_shap, predict_prob, prepare_image)

st.set_page_config(page_title="Generated AI Art Detector (thesis demo)", page_icon="🎨", layout="wide")


@st.cache_resource(show_spinner="Loading model weights...")
def get_models(weights_dir):
    return load_models(weights_dir)


@st.cache_data(show_spinner=False)
def cached_pixel_shap(_model, model_label, crop_bytes, max_evals):
    """Cached so moving other widgets doesn't recompute the (slow) pixel SHAP. `_model` is not hashed;
    model_label + the image bytes + max_evals form the cache key."""
    crop = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(224, 224, 3)
    return pixel_shap(_model, crop, max_evals=max_evals)


st.title("Generated AI Art Detector")
st.caption("Thesis Proof of Concept: Enhancing Lightweight CNN Models with Explainable AI for Detecting "
           "AI-Generated Art on Online Platforms")

with st.sidebar:
    st.header("Settings")
    weights_dir = st.text_input("Weights folder", value=os.path.join(os.path.dirname(__file__), "weights"),
                                help="Folder containing mobilenet_lfab_model.pth and efficientnet_lfab_model.pth")
    threshold = st.slider("Decision threshold (P(AI) ≥ this → 'AI-generated')", 0.30, 0.90, 0.50, 0.01,
                          help="0.5 is what the notebooks report as the default.")
    match_training = st.checkbox("Match training preprocessing (JPEG-recompress every upload)", value=True,
                                 help="Notebook 1 saved every training image as a 256x256 JPEG. Leave on unless "
                                      "you are deliberately testing the effect of skipping it.")
    st.subheader("Explanations")
    show_cam = st.checkbox("Grad-CAM heatmaps", value=True)
    show_band_shap = st.checkbox("SHAP: LFAB frequency bands", value=True,
                                 help="Exact Shapley values over LFAB's 6 frequency bands. Fast (64 forward passes per model).")
    show_pix_shap = st.checkbox("SHAP: pixel regions (slow)", value=False,
                                help="Partition explainer with a blur masker, as in Notebook 2. Takes seconds to "
                                     "a minute or more depending on the machine.")
    shap_evals = st.slider("Pixel-SHAP evaluations (more = finer but slower)", 100, 800, 250, 50,
                           disabled=not show_pix_shap)
    st.caption(f"Running on: **{DEVICE}**")

models, missing = get_models(weights_dir)
if missing:
    st.error("Some checkpoints were not found:\n\n" + "\n".join(f"- {m}" for m in missing))
if not models:
    st.info("Copy `mobilenet_lfab_model.pth` (Notebook 2 output) and `efficientnet_lfab_model.pth` "
            "(Notebook 3 output) into the weights folder, then refresh.")
    st.stop()

up = st.file_uploader("Upload an artwork image", type=["jpg", "jpeg", "png", "webp"])

with st.expander("About this demo / limits"):
    st.markdown(
        "- Trained on 224×224 crops of artwork from ArtBench, WikiArt, MidJourney and Stable Diffusion. "
        "In-distribution test accuracy: 91.1% (Model A) and 92.3% (Model B), 93.3% for the ensemble.\n"
        "- On two generators never seen in training, accuracy drops to about 80% (Model A) and 83% (Model B).\n"
        "- On photo-style images (CIFAKE) it is weak, roughly 64% accuracy, so don't feed it photographs.\n"
        "- The score is a model probability, not proof. Noise, heavy compression or resizing lower reliability.\n"
        "- LFAB adds little accuracy in the in-distribution ablation (about +0.6 points for both models). On "
        "unseen generators it's a mixed bag: roughly flat for Model B (-0.3 pts) but a real drop for Model A "
        "(-4.3 pts vs. no LFAB), so treat Model A's unseen-generator score as the less reliable of the two."
    )

if up is None:
    st.stop()

# ---- step-by-step run, so the audience can see what is happening -----------------------------
results, cams, band_shap = {}, {}, {}
with st.status("Analysing image...", expanded=True) as status:
    st.write("Preprocessing (RGB → 256×256 → center-crop 224 → normalise), same as training")
    img256, tensor = prepare_image(up, force_jpeg=match_training)
    crop = np.array(img256.crop((16, 16, 240, 240)))  # the exact 224x224 pixels the models see
    time.sleep(0.2)

    st.write("Running the models")
    for label, model in models.items():
        t0 = time.perf_counter()
        results[label] = (predict_prob(model, tensor), (time.perf_counter() - t0) * 1000)

    st.write("Combining (ensemble = average of the models' probabilities)")
    ensemble = sum(p for p, _ in results.values()) / len(results)

    if show_cam:
        st.write("Grad-CAM: where did each model look?")
        for label, model in models.items():
            cams[label] = gradcam_overlay(model, tensor)

    if show_band_shap:
        st.write("SHAP over LFAB's frequency bands (exact, 64 coalitions per model)")
        for label, model in models.items():
            band_shap[label] = lfab_band_shap(model, tensor)
    status.update(label="Done", state="complete", expanded=False)

# ---- results ---------------------------------------------------------------------------------
verdict_ai = ensemble >= threshold
left, right = st.columns([1, 1.4])
with left:
    st.image(crop, caption="What the models see (224×224 center crop)")
with right:
    st.subheader("🤖 Likely AI-generated" if verdict_ai else "🖌️ Likely human-made")
    st.progress(min(max(ensemble, 0.0), 1.0), text=f"Ensemble P(AI) = {ensemble:.1%}   (threshold {threshold:.0%})")
    for label, (p, ms) in results.items():
        st.write(f"**{label}**: P(AI) = {p:.1%}  ·  {ms:.0f} ms")
    if abs(ensemble - threshold) < 0.10:
        st.warning("Close to the threshold: treat this as uncertain.")

if cams:
    st.subheader("Grad-CAM (red = regions pushing the score toward 'AI-generated')")
    cols = st.columns(len(cams))
    for col, (label, overlay) in zip(cols, cams.items()):
        col.image(overlay, caption=label)

if band_shap:
    st.subheader("SHAP: which LFAB frequency bands moved the score?")
    cols = st.columns(len(band_shap))
    for col, (label, (phi, p_all, p_none)) in zip(cols, band_shap.items()):
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.barh(range(len(phi)), phi * 100, color=["#d62728" if v > 0 else "#1f77b4" for v in phi])
        ax.set_yticks(range(len(phi)))
        ax.set_yticklabels([b.split(" (")[0] for b in BAND_LABELS], fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Shapley value (percentage points of P(AI))", fontsize=8)
        ax.set_title(label.split(" + ")[0], fontsize=9)
        fig.tight_layout()
        col.pyplot(fig)
        plt.close(fig)
        gains = lfab_gains(models[label])
        col.caption(f"Total LFAB effect: {(p_all - p_none) * 100:+.2f} pts (all bands kept vs. all dropped). "
                    f"Mean learned gain per band, low→high frequency (1.00 = neutral): "
                    f"{', '.join(f'{g:.2f}' for g in gains)}.")
    st.caption("Red pushes toward 'AI-generated', blue toward 'human'. Band 0 is the lowest radial frequency "
               "band and Band 5 the highest. If all bars are close to zero, LFAB had little influence on this "
               "image, which fits the ablation result that LFAB adds little accuracy.")

    with st.expander("Learned LFAB gain maps (frequency filter, averaged over channels)"):
        cols = st.columns(len(band_shap))
        for col, label in zip(cols, band_shap):
            gm = lfab_gain_map(models[label])
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            im = ax.imshow(gm, cmap="coolwarm", vmin=0.0, vmax=2.0)
            ax.set_title(label.split(" + ")[0], fontsize=9)
            ax.set_xlabel("horizontal frequency", fontsize=8)
            ax.set_ylabel("vertical frequency", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            col.pyplot(fig)
            plt.close(fig)
        st.caption("Gain 1.0 (white) = frequency left unchanged; below 1 damps it, above 1 amplifies it "
                   "(range 0 to 2). A map that stays near white means the filter barely moved from neutral.")

if show_pix_shap:
    st.subheader("SHAP: which image regions moved the score?")
    shap_label = st.radio("Model to explain", list(models), horizontal=True)
    with st.spinner("Computing pixel SHAP... this can take a while on CPU"):
        t0 = time.perf_counter()
        sv = cached_pixel_shap(models[shap_label], shap_label, crop.tobytes(), int(shap_evals))
        elapsed = time.perf_counter() - t0
    lim = max(float(np.abs(sv).max()), 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(crop)
    axes[0].axis("off")
    axes[0].set_title("Input", fontsize=9)
    axes[1].imshow(crop.mean(axis=-1), cmap="gray", alpha=0.6)
    im = axes[1].imshow(sv, cmap="bwr", vmin=-lim, vmax=lim, alpha=0.7)
    axes[1].axis("off")
    axes[1].set_title("SHAP (red → AI, blue → human)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"{shap_label} · {int(shap_evals)} evaluations · {elapsed:.1f} s. SHAP values sum to the change "
               f"in P(AI) between the blurred baseline and this image; fewer evaluations give coarser, noisier maps.")
