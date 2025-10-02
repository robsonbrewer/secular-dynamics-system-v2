# -*- coding: utf-8 -*-
"""
main.py
-------
Executa o cálculo secular (Laplace–Lagrange) e gera:
 - Fig. 7.1 (evolução de e e I)
 - Fig. 7.4 (coeficiente A(a) e frequências g,f)
com base em Murray & Dermott (2000), cap. 7.

Fluxo:
1) Carregar dados de /data/inputPlanets.json e /data/constants.json.
2) Construir matrizes seculares A, B.
3) Diagonalizar → obter autovalores/autovetores.
4) Reconstruir séries temporais (h,k,p,q) → (e,I).
5) Calcular A(a) para corpo fictício.
6) Plotar Fig. 7.1 e Fig. 7.4.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.secular import (
    load_bodies_from_json,
    load_constants,
    build_AB_from_bodies,
    secular_eigendecomp,
    build_time_series,
    hk_to_e_varpi,
    pq_to_i_Omega,
    initial_conditions_from_json,
    b32_1, mean_motion_radyr,
    DEG2RAD, RAD2DEG
)

# ===============================================================
# Funções auxiliares
# ===============================================================
def A_for_test_particle(a_test, bodies, consts):
    """
    Calcula o coeficiente A(a) para um corpo fictício em semi-eixo a_test,
    conforme M&D eq. (7.55).
    """
    n = mean_motion_radyr(a_test, consts)
    soma = 0.0
    for b in bodies:
        aj = b["a"]
        mj = b["mass"]
        alpha = min(a_test, aj) / max(a_test, aj)
        soma += mj * alpha * b32_1(alpha)
    A_val = 0.25 * n * soma
    return A_val * RAD2DEG  # rad/yr → deg/yr


def plot_fig74(a_values, A_values, g_vals_degyr, s_vals_degyr, out_dir="output"):
    """
    Reproduz a Fig. 7.4 de Murray & Dermott:
      - Curva A(a)
      - Linhas sólidas: g1, g2 (excentricidade)
      - Linhas tracejadas: f1, f2 (inclinação)
    """
    plt.figure(figsize=(7,5))

    # Curva A(a)
    plt.plot(a_values, A_values, color="black", linewidth=1.0, label=r"$A(a)$")

    # Linhas horizontais sólidas (g1,g2)
    for g in g_vals_degyr:
        plt.axhline(g, color="black", linestyle="-", linewidth=0.8)

    # Linhas tracejadas (f1,f2) — ignorando autovalores nulos
    for f in s_vals_degyr:
        if abs(f) > 1e-6:
            plt.axhline(abs(f), color="black", linestyle="--", linewidth=0.8)

    plt.xlim(0, 30)
    plt.ylim(0, 0.05)
    plt.xlabel("Semi-major axis (AU)")
    plt.ylabel(r"$A$ (deg/yr)")
    plt.title("Fig. 7.4 – A(a) and eigenfrequencies g,f")

    plt.grid(False)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/fig74.png", dpi=300)
    plt.close()

# ===============================================================
# Execução principal
# ===============================================================
def main():
    # ---------------------------
    # 1) Carregar dados
    # ---------------------------
    bodies = load_bodies_from_json("data/inputPlanets.json")
    consts = load_constants("data/constants.json")

    print("="*70)
    print("Condições iniciais dos corpos:")
    for b in bodies:
        print(f" {b['name']}: a={b['a']} AU, e={b['e']}, I={b['I']} deg")
    print("="*70)

    # ---------------------------
    # 2) Matrizes seculares
    # ---------------------------
    M = build_AB_from_bodies(bodies, consts)

    print("Matriz A (excentricidade) [deg/yr]:")
    print(M.A_degyr)
    print("-"*70)
    print("Matriz B (inclinação) [deg/yr]:")
    print(M.B_degyr)
    print("="*70)

    # ---------------------------
    # 3) Autovalores/autovetores
    # ---------------------------
    eig = secular_eigendecomp(M)
    g_vals_degyr = np.real(eig.g_radyr) * RAD2DEG
    s_vals_degyr = np.real(eig.s_radyr) * RAD2DEG

    print("Autovalores g (deg/yr):", g_vals_degyr)
    print("Autovetores Sg:")
    print(np.real(eig.Sg))
    print("-"*70)
    print("Autovalores s (deg/yr):", s_vals_degyr)
    print("Autovetores Ss:")
    print(np.real(eig.Ss))
    print("="*70)

    # ---------------------------
    # 4) Séries temporais (Fig. 7.1)
    # ---------------------------
    h0, k0, p0, q0 = initial_conditions_from_json(bodies)
    t_years = np.linspace(-1e5, 1e5, 2000)

    h, k, p, q = build_time_series(eig, h0, k0, p0, q0, t_years)
    e, _ = hk_to_e_varpi(h, k)
    inc, _ = pq_to_i_Omega(p, q)

    plt.figure(figsize=(12,5))

    # Excentricidade
    plt.subplot(1,2,1)
    for i, b in enumerate(bodies):
        plt.plot(t_years, e[:,i], label=b["name"])
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Time (years)")
    plt.ylabel("e")
    plt.title("Secular evolution of eccentricity")
    plt.legend()

    # Inclinação
    plt.subplot(1,2,2)
    for i, b in enumerate(bodies):
        plt.plot(t_years, np.degrees(inc[:,i]), label=b["name"])
    plt.axhline(0, color="k", linestyle="--")
    plt.xlabel("Time (years)")
    plt.ylabel("I [deg]")
    plt.title("Secular evolution of inclination")
    plt.legend()

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/fig71.png", dpi=300)
    plt.close()

    # ---------------------------
    # 5) Calcular A(a) (Fig. 7.4)
    # ---------------------------
    a_test = np.linspace(2.0, 30.0, 400)
    A_values = [A_for_test_particle(a, bodies, consts) for a in a_test]

    plot_fig74(a_test, A_values, g_vals_degyr, s_vals_degyr)


if __name__ == "__main__":
    main()
