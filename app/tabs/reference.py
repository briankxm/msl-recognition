"""
Reference Library tab — shows what each sign is supposed to look like.

No model involved. Displays a grid of reference images from the raw dataset,
with an expandable gallery for each class showing multiple samples.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from PIL import Image

from app.ui_helpers import get_reference_images
from src import config


MODE_LABELS = {
    "alphabet": "Alphabet (A\u2013Z)",
    "number": "Number (0\u201310)",
}


def render(selected_mode, encoder):
    """Render the Reference Library tab.

    Args:
        selected_mode: "alphabet" or "number"
        encoder: fitted LabelEncoder (to get class labels)
    """
    st.subheader(f"Reference Library \u2014 {MODE_LABELS[selected_mode]}")

    if encoder is None:
        st.info("No trained models found. Class labels unavailable.")
        return

    classes = [str(c) for c in encoder.classes_]
    st.caption(f"Here are some sample references for {len(classes)} classes. You can follow along each hand gesture, below this section are more samples for you to learn.")

    # --- grid overview: one image per class ---
    cols_per_row = 6
    for row_start in range(0, len(classes), cols_per_row):
        row_classes = classes[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, cls in zip(cols, row_classes):
            imgs = get_reference_images(selected_mode, cls, max_samples=1)
            with col:
                if imgs:
                    st.image(
                        imgs[0],
                        caption=cls,
                        width="stretch",
                    )
                else:
                    st.warning(f"{cls}")

    st.divider()

    # --- expandable detail: multiple samples per class ---
    st.subheader("Browse all samples")
    selected_class = st.selectbox("Select a class to view", classes, key="ref_class")
    if selected_class:
        imgs = get_reference_images(selected_mode, selected_class, max_samples=8)
        if not imgs:
            st.info(f"No images found for class '{selected_class}'.")
            return

        st.caption(f"{len(imgs)} sample(s) for **{selected_class}**")
        detail_cols = st.columns(min(len(imgs), 4))
        for i, img_path in enumerate(imgs):
            with detail_cols[i % len(detail_cols)]:
                st.image(img_path, width="stretch")
