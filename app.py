import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# --- PAGE SETUP ---
st.set_page_config(page_title="ML Sandbox", page_icon="🧠")
st.title("🧠 ML Learning Sandbox")
st.markdown("### Module 1: Decision Boundaries & Overfitting")

# --- THE INTERACTIVE SLIDER ---
# This creates a slider on the left side of your screen
k_value = st.sidebar.slider("Adjust 'K' (Number of Neighbors)", 1, 50, 5)

st.write(f"Adjust the slider to see how the model learns. Currently, **K = {k_value}**")

# --- GENERATE DATA ---
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# --- THE MODEL (This is what you're adjusting!) ---
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X, y)

# --- PLOTTING THE CHART ---
fig, ax = plt.subplots(figsize=(10, 6))
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create the colors
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
ax.set_title(f"KNN Decision Boundary (K = {k_value})")

# Show it in the app
st.pyplot(fig)