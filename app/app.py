import tempfile
import json
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.cm as cm

from text import run_pipeline
import vision


@st.cache_data
def list_dataset_images(dataset_dir: str):
	p = Path(dataset_dir)
	files = [str(x) for x in p.rglob("*.jpg")] + [str(x) for x in p.rglob("*.png")]
	files.sort()
	return files


def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):
	h = heatmap - np.min(heatmap)
	if np.max(h) > 0:
		h = h / np.max(h)
	else:
		h = np.zeros_like(h)

	cmap = cm.get_cmap("jet")
	colored = cmap(h)  
	colored = (colored * 255).astype("uint8")

	heat_pil = Image.fromarray(colored).convert("RGBA")

	heat_pil = heat_pil.resize(pil_img.size, resample=Image.BILINEAR)

	base = pil_img.convert("RGBA")
	blended = Image.blend(base, heat_pil, alpha=alpha)
	return blended


def get_display_image(pil_img: Image.Image, max_dim: int = 640) -> Image.Image:
	img = pil_img.copy()
	img.thumbnail((max_dim, max_dim), Image.LANCZOS)
	return img


def save_uploaded_file(uploaded_file) -> str:
	suffix = Path(uploaded_file.name).suffix or ".png"
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="uploaded_", dir="/tmp") as tf:
		tf.write(uploaded_file.getbuffer())
		return tf.name


st.set_page_config(page_title="Crop Assistant", layout="wide")

st.title("MultiModal Crop Assistant")

col1, col2 = st.columns([1, 2])

with col1:
	st.header("Input")

	upload = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])

	selected = None
	if upload is not None:
		selected = save_uploaded_file(upload)

	question = st.text_area("Question for the assistant", "What is likely causing this disease and suggested actions?")

	# Only show the grad rollout option if an image has been uploaded
	if selected is not None:
		show_grad = st.checkbox("Show Grad Rollout overlay")
	else:
		show_grad = False

	run = st.button("Run pipeline")

with col2:
	st.header("Output")
	answer_box = st.empty()
	image_box = st.empty()
	subgraph_box = st.expander("Subgraph / Context (raw)")

if run:
	pil_img = None
	if selected is not None:
		try:
			pil_img = Image.open(selected).convert("RGB")
		except Exception as e:
			st.error(f"Could not open image: {e}")
			pil_img = None

	try:
		with st.spinner("Running pipeline...."):
			answer, subgraph = run_pipeline(question, selected)

		answer_box.markdown("**Assistant answer**")
		answer_box.write(answer)

		try:
			pretty = json.dumps(subgraph, indent=2)
		except Exception:
			pretty = str(subgraph)
		subgraph_box.code(pretty, language="json")

	except Exception as e:
		st.error(f"Pipeline error: {e}")

	try:
		if pil_img is not None:
			display_pil = get_display_image(pil_img, max_dim=640)

			last_selected = st.session_state.get("_last_selected", None)
			last_question = st.session_state.get("_last_question", None)

			should_run_vision = True
			if last_selected == selected and last_question == question:
				should_run_vision = False

			if show_grad and should_run_vision:
				with st.spinner("Generating grad rollout (may take time)..."):
					arr = np.array(pil_img)
					probs = vision.predict_fn([arr])
					pred_id = int(np.argmax(probs[0]))

					heatmap = vision.generate_grad_rollout(vision.inference_model, vision.inference_processor, pil_img, pred_id)

				overlay = overlay_heatmap_on_image(display_pil, heatmap, alpha=0.5)

				left, right = st.columns(2)
				left.image(display_pil, caption="Original image", use_container_width=False, width=600)
				right.image(overlay, caption="Grad rollout overlay", use_container_width=False, width=600)

				st.session_state["_last_selected"] = selected
				st.session_state["_last_question"] = question
			else:
				image_box.image(display_pil, caption="Input image", use_container_width=False, width=600)

				st.session_state["_last_selected"] = selected
				st.session_state["_last_question"] = question

	except Exception as e:
		st.warning(f"Could not generate/display grad rollout: {e}")



