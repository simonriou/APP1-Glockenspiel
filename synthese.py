import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

def phi(x, k, sigma):
    """Évalue la déformée spatiale analytique en un point ou sur une grille."""
    return np.cos(k*x) + np.cosh(k*x) + sigma * (np.sin(k*x) + np.sinh(k*x))



def synthese_lame(L, S, rho, E, I_y, duree, fs, x_impact, I_0, x_ecoute, N_modes, alpha=0.0, beta=0.0, excitation_type="dirac", Te=0.001):
    """
    - Poutre 1D (on ne considère que la flexion selon z dans une succession de plans (y, z))
    - Avec amortissement de Rayleigh (alpha + beta * omega^2)
    - Approximation des k_n en considérant 1/cosh(x) ~ 0
    - CLs libre-libre
    - Excitation localisée en temps et en espace OU fonction porte (durée Te)
    """

    # Vecteur temps pour l'audio final
    t = np.linspace(0, duree, int(duree * fs), endpoint=False)
    
    # Grille spatiale (uniquement pour calculer les intégrales de masse)
    x_grille = np.linspace(0, L, 1000)
    
    # Signal audio final
    signal_total = np.zeros_like(t)

    # Calcul des déformées modales et fréquences propres
    for n in range(1, N_modes + 1):
        
        # Nombre d'onde (k_n)
        k = (2 * n + 1) * np.pi / (2 * L)
        
        # Pulsation temporelle (omega_n)
        omega_n = (k**2) * np.sqrt((E * I_y) / (rho * S))
        # Amortissement de Rayleigh
        # Le terme alpha représente l'amortissement visqueux, et beta l'amortissement structurel
        delta_n = 0.5 * (alpha + beta * omega_n**2)  # Amortissement de Rayleigh
        omega_d = np.sqrt(omega_n**2 - delta_n**2)  # Pulsation amortie (=omega_n si pas d'amortissement)
        
        # Ratio géométrique (sigma_n) tend strictement vers -1
        if k * L > 50:
            sigma = -1.0
        else:
            sigma = (np.cos(k*L) - np.cosh(k*L)) / (np.sinh(k*L) - np.sin(k*L))
            
        # 4. Calcul de la masse modale (M_n) par intégration numérique
        # On évalue la forme sur toute la grille, on l'élève au carré, et on intègre l'aire.
        forme_sur_grille = phi(x_grille, k, sigma)
        M_n = np.trapezoid(rho * S * forme_sur_grille**2, x_grille)
        K_n = M_n * omega_n**2  # Raideur modale
        
        # Amplitudes temporelles
        A_n = 0
        # Force d'excitation reçue par le mode n
        amplitude_impact = phi(x_impact, k, sigma)
        
        # Déformée modale évaluée au point d'écoute
        amplitude_ecoute = phi(x_ecoute, k, sigma)

        if excitation_type == "dirac":
            # Amplitude temporelle associée au mode n
            A_n = (I_0 * amplitude_impact) / (M_n * omega_d)
            
            # Contribution temporelle du mode n pour une excitation de type Dirac
            q_n_t = A_n * np.sin(omega_d * t) * np.exp(-delta_n * t) # Décroissance exponentielle si amortissement (= 1 si pas d'amortissement)

        elif excitation_type == "porte":
            # Amplitude temporelle associée au mode n
            A_n = (I_0 * amplitude_impact) / (K_n)

            # Contribution temporelle du mode n pour une excitation de type porte
            tmps_decale = np.maximum(0, t - Te)
            heaviside_Te = (t >= Te).astype(float) # u(t - Te)
            reponse_retardee = 1 - np.exp(-delta_n * tmps_decale) * (np.cos(omega_d * tmps_decale) + (delta_n / omega_d) * np.sin(omega_d * tmps_decale))
            reponse_retardee = reponse_retardee * heaviside_Te

            reponse_t = 1 - np.exp(-delta_n * t) * (np.cos(omega_d * t) + (delta_n / omega_d) * np.sin(omega_d * t))
            q_n_t = reponse_t - reponse_retardee

            q_n_t *= A_n

        # Accumulateur des contributions modales pour le signal final
        signal_total += amplitude_ecoute * q_n_t

    # Normalisation
    pic_max = np.max(np.abs(signal_total))
    if pic_max > 0:
        signal_audio = signal_total / pic_max
    else:
        signal_audio = signal_total
        
    return t, signal_audio


def visualiser_modes(L, chemin_sortie):
    """Trace et sauvegarde les formes spatiales des premiers modes."""
    x = np.linspace(0, L, 1000)
    figure, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)

    for index, axe in enumerate(axes.flat, start=1):
        k = (2 * index + 1) * np.pi / (2 * L)
        if k * L > 50:
            sigma = -1.0
        else:
            sigma = (np.cos(k * L) - np.cosh(k * L)) / (np.sinh(k * L) - np.sin(k * L))

        forme = phi(x, k, sigma)
        maximum = np.max(np.abs(forme))
        if maximum > 0:
            forme = forme / maximum

        axe.plot(x, forme, color="steelblue")
        axe.axhline(0, color="black", linewidth=0.6)
        axe.set_title(f"Mode {index}")
        axe.set_ylabel("Amplitude normalisée")
        axe.grid(True, alpha=0.3)

    for axe in axes[-1]:
        axe.set_xlabel("Position x (m)")

    figure.suptitle("Formes spatiales des 12 premiers modes")
    figure.tight_layout()
    figure.savefig(chemin_sortie, format="jpg", dpi=150)
    plt.close(figure)


if __name__ == "__main__":

    config_path = Path(__file__).with_name("config.json")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    geometry = config["geometry"]
    material = config["material"]
    synthesis = config["synthesis"]
    playing = config["playing"]
    model = config["model"]

    amortissement = model["damping"] # 1 si amortissement, 0 sinon

    # Coefficients de Rayleigh pour l'amortissement
    alpha = 0.0
    beta = 0.0

    if amortissement:
        alpha = material["alpha"]
        beta = material["beta"]

    porte_t = model["porte_t"] # 1 si on considère une excitation comme fonction porte, 0 comme Dirac
    excitation_type = "porte" if porte_t else "dirac"

    L = geometry["length"]
    l = geometry["width"]
    h = geometry["thickness"]
    S = l * h
    I_y = (l * h**3) / 12
    rho = material["density"]
    E = material["young_modulus"]

    duree = synthesis["duration"]
    fs = synthesis["sampling_frequency"]
    N_modes = synthesis["modes"]

    x_impact = playing["impact_position"]
    I_0 = playing["impulse"]
    x_ecoute = playing["listening_position"]
    Te = playing["impact_duration"] # Durée de l'impact (s) si excitation de type porte

    # Lancement du calcul
    temps, audio = synthese_lame(L, S, rho, E, I_y, duree, fs, x_impact, I_0, x_ecoute, N_modes, alpha=alpha, beta=beta, excitation_type=excitation_type, Te=Te)
    
    # Export audio
    output_directory = Path("output")
    output_directory.mkdir(exist_ok=True)
    write(output_directory / f"synthese_lame_amort_{excitation_type}_{N_modes}.wav", fs, audio.astype(np.float32))

    # Export des formes spatiales des 12 premiers modes
    visualiser_modes(L, output_directory / "modes.jpg")