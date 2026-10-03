import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import synthese
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.markdown("""
<style>
    /* Réduit l'écart entre chaque élément vertical de la sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important; /* Par défaut souvent à 1rem */
    }

    /* Optionnel : réduit le padding intérieur en haut de la sidebar */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Synthèse lame Glockenspiel - GUI")

with st.sidebar:
    st.header("Paramètres")
    st.subheader("Modèle")
    amortissement = st.checkbox("Amortissement", value=True, key="amortissement")

    st.subheader("Géométrie")
    L_cm = st.number_input("Longueur (cm)", value=12.0, format="%.3f", step=0.1, key="L_cm")
    l_cm = st.number_input("Largeur (cm)", value=3.0, format="%.3f", step=0.1, key="l_cm")
    h_cm = st.number_input("Épaisseur (cm)", value=0.08, format="%.3f", step=0.01, key="h_cm")
    L = L_cm / 100
    l = l_cm / 100
    h = h_cm / 100
    S = l * h
    st.write(f"Surface S = {S:.4f} m²")

    st.subheader("Matériau")
    rho = st.number_input("Densité ρ (kg/m³)", value=7850, step=100, key="rho")
    E = st.number_input("Module de Young E (Pa)", value=2e12, format="%e", step=1e10, key="E")
    alpha = st.number_input("Coefficient d'amortissement α", value=3.0, step=0.01, disabled=not amortissement, key="alpha")
    beta = st.number_input("Coefficient d'amortissement β", value=1e-7, format="%e", step=1e-8, disabled=not amortissement, key="beta")

    st.subheader("Excitation")
    excitation_type = st.radio("Type d'excitation", options=["porte", "dirac"], horizontal=True, key="excitation_type")
    excitation_duree = st.number_input("Durée de l'impact (s)", value=1e-3, format="%e", step=0.0001, disabled=(excitation_type == "dirac"), key="excitation_duree")

    st.subheader("Synthèse")
    duree = st.number_input("Durée de la synthèse (s)", value=2.0, step=0.1, key="duree")
    fs = st.number_input("Fréquence d'échantillonnage (Hz)", value=44100, step=1000, key="fs")
    n_modes = st.number_input("Nombre de modes", value=32, min_value=1, max_value=128, step=1, key="n_modes")

amortissement = st.session_state.amortissement

L = st.session_state.L_cm / 100
l = st.session_state.l_cm / 100
h = st.session_state.h_cm / 100
S = l * h
I_y = (l * h**3) / 12
rho = st.session_state.rho
E = st.session_state.E

duree = st.session_state.duree
fs = st.session_state.fs
N_modes = st.session_state.n_modes

I_0 = 1.0  # Amplitude de l'impulsion (N·s)
Te = st.session_state.excitation_duree

alpha = st.session_state.alpha if amortissement else 0.0
beta = st.session_state.beta if amortissement else 0.0

if "x_impact" not in st.session_state:
    st.session_state.x_impact = None
if "x_ecoute" not in st.session_state:
    st.session_state.x_ecoute = None
if "last_click" not in st.session_state:
    st.session_state.last_click = None

st.header("Positions d'impact et d'écoute")

click_type = st.radio(
    "Sélectionner la position de l'excitation ou de l'écoute",
    options=["Excitation", "Écoute"],
    horizontal=True
)

IMG_WIDTH = 700
IMG_HEIGHT = max(50, int(IMG_WIDTH * (l / L)))

img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), "#787171")
draw = ImageDraw.Draw(img)

if st.session_state.x_impact is not None:
    x_px = int(st.session_state.x_impact * IMG_WIDTH)
    draw.line([(x_px, 0), (x_px, IMG_HEIGHT)], fill="blue", width=4)

if st.session_state.x_ecoute is not None:
    x_px = int(st.session_state.x_ecoute * IMG_WIDTH)
    draw.line([(x_px, 0), (x_px, IMG_HEIGHT)], fill="red", width=4)

st.write("Cliquez sur l'image pour sélectionner la position de l'impact (bleu) ou de l'écoute (rouge).")
clic = streamlit_image_coordinates(img, key="rect")

if clic is not None and clic != st.session_state.last_click:
    st.session_state.last_click = clic
    x_relatif = clic['x'] / IMG_WIDTH
    if click_type == "Excitation":
        st.session_state.x_impact = x_relatif
    else:
        st.session_state.x_ecoute = x_relatif

    st.rerun()

st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.session_state.x_impact is not None:
        st.info(f"**Excitation :** {st.session_state.x_impact * L:.3f} m (ratio: {st.session_state.x_impact:.2f})")
    else:
        st.info("**Excitation :** *En attente*")
        
with col2:
    if st.session_state.x_ecoute is not None:
        st.error(f"**Écoute :** {st.session_state.x_ecoute * L:.3f} m (ratio: {st.session_state.x_ecoute:.2f})")
    else:
        st.error("**Écoute :** *En attente*")

st.header("Résultats")

if st.button("SIMULATION", type="primary"):
    
    # Les deux positions doivent être définies
    if st.session_state.x_impact is None or st.session_state.x_ecoute is None:
        st.warning("Sélectionner les positions d'excitation et d'écoute avant de lancer la simulation.")
    else:
        with st.spinner("En cours..."):
            
            x_impact_phys = st.session_state.x_impact * L
            x_ecoute_phys = st.session_state.x_ecoute * L
            
            exc_type = st.session_state.excitation_type
            
            # Lancement de la synthèse
            # (Note : si synthese_lame prend alpha et beta en argument, vous pouvez les ajouter ici)
            temps, audio = synthese.synthese_lame(
                L, S, rho, E, I_y, duree, fs, 
                x_impact_phys, I_0, x_ecoute_phys, N_modes,
                alpha=alpha, beta=beta,
                excitation_type=exc_type, Te=Te
            )
            
            # Normalisation du signal audio pour éviter les distorsions dans le lecteur Streamlit
            max_amp = np.max(np.abs(audio))
            audio_norm = audio / max_amp if max_amp > 0 else audio
            audio_norm = np.float32(audio_norm)

        st.success("Terminé.")

        # Lecteur audio
        st.subheader("Audio")
        st.audio(audio_norm, sample_rate=fs)

        # Forme d'onde & spectrogramme
        st.subheader("DSP")
        
        # On divise l'espace en deux colonnes pour les graphiques
        col_onde, col_spec = st.columns(2)
        
        with col_onde:
            st.markdown("**Forme d'onde**")
            fig_wave, ax_wave = plt.subplots(figsize=(6, 4))
            ax_wave.plot(temps, audio, color='#1f77b4', linewidth=0.8)
            ax_wave.set_xlabel("Temps (s)")
            ax_wave.set_ylabel("Amplitude")
            ax_wave.grid(True, alpha=0.3)
            st.pyplot(fig_wave)
            
        with col_spec:
            st.markdown("**Spectrogramme**")
            fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
            Pxx, freqs, bins, im = ax_spec.specgram(audio, Fs=fs, NFFT=2048, cmap='viridis')
            ax_spec.set_xlabel("Temps (s)")
            ax_spec.set_ylabel("Fréquence (Hz)")
            fig_spec.colorbar(im, ax=ax_spec, label="Intensité (dB)")
            st.pyplot(fig_spec)

        # Déformées
        st.subheader("12 premières déformées modales")
        
        with st.spinner("Génération des graphiques modaux..."):
            # Création du dossier temporaire pour l'image
            output_directory = Path("output")
            output_directory.mkdir(exist_ok=True)
            modes_path = output_directory / "modes.jpg"

            synthese.visualiser_modes(L, modes_path)
            
            if modes_path.exists():
                st.image(str(modes_path), width='stretch')
            else:
                st.error("L'image des modes n'a pas pu être générée.")