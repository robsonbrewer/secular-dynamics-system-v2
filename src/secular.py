# -*- coding: utf-8 -*-
"""
secular.py
-----------
Núcleo da teoria secular linear (Laplace–Lagrange), conforme
Murray & Dermott (2000), cap. 7 (eqs. 7.29–7.34 e 7.42).

Fluxo:
1) Ler os corpos de /data/inputPlanets.json e as constantes de /data/constants.json.
2) Calcular as matrizes seculares A (excentricidade) e B (inclinação),
   usando coeficientes de Laplace b_{3/2}^{(1)} e b_{3/2}^{(2)} (via integral).
3) Diagonalizar A e B → autovalores (g,s) e autovetores (Sg,Ss).
   **Convencionar e normalizar autovetores** (norma 1 e primeiro elemento > 0).
4) Reconstruir séries temporais: (h,k,p,q) → (e,ϖ,i,Ω).
5) Utilidades: salvar intermediários e converter variáveis.

Unidades:
- Entrada: a [AU], massas em fração de M0 (ex.: m_J≈9.54e-4), ângulos em graus.
- Constantes: G em AU^3/yr^2/M☉ (no repo: ~4π^2), M0≈1.
- Movimento médio: n_i = sqrt(G*M0/a_i^3) [rad/ano].
- Saída A,B: graus/ano (facilita comparação com literatura).
- Autovalores internos para evolução temporal: rad/ano.
"""

from __future__ import annotations
import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ---------------------------
# Conversões/constantes
# ---------------------------
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
ARCSEC_PER_RAD = 206264.80624709636  # 1 rad ≈ 206265 arcsec
TWOPI = 2.0 * np.pi

# ---------------------------
# Estruturas de dados
# ---------------------------
@dataclass
class SecularMatrices:
    """Matrizes seculares em graus/ano (conveniente para leitura/comparação)."""
    A_degyr: np.ndarray  # excentricidade (n x n)
    B_degyr: np.ndarray  # inclinação   (n x n)

@dataclass
class EigenData:
    """
    Autovalores/autovetores para A e B.
    Autovalores são fornecidos em RAD/ANO (adequados à evolução temporal).
    """
    g_radyr: np.ndarray
    Sg: np.ndarray
    s_radyr: np.ndarray
    Ss: np.ndarray

# ---------------------------
# Utilidades de pasta
# ---------------------------
def _ensure_output_dir(path: str = "output") -> None:
    """Garante a existência do diretório de saída."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ---------------------------
# Leitura de JSONs
# ---------------------------
def load_bodies_from_json(json_path: str) -> List[Dict]:
    """
    Lê /data/inputPlanets.json no formato:
    {
      "planets": [
        {"name":"Jupiter","mass":9.54e-4,"a":5.204,"e":0.0489,"I":1.303,"omega":14.753,"Omega":100.464},
        {"name":"Saturn", "mass":2.86e-4,"a":9.537,"e":0.0565,"I":2.485,"omega":92.431,"Omega":113.665}
      ]
    }
    Retorna a lista de dicionários de corpos.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bodies = data.get("planets", [])
    if len(bodies) < 2:
        raise ValueError("É necessário pelo menos 2 corpos em inputPlanets.json.")
    return bodies

def load_constants(json_path: str) -> Dict:
    """
    Lê /data/constants.json, ex.:
      { "G": 39.47841760435743, "M0": 1.0, "AU": 1.0, "degToRad": 0.01745... }
    Insere valores padrão caso alguma chave esteja ausente.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        consts = json.load(f)
    consts.setdefault("G", 39.47841760435743)  # ~4π^2
    consts.setdefault("M0", 1.0)
    consts.setdefault("AU", 1.0)
    consts.setdefault("degToRad", np.pi/180.0)
    return consts

# ---------------------------
# Coeficientes de Laplace via integral
# ---------------------------
def laplace_b_sj(alpha: float, s: float, j: int, npsi: int = 8192) -> float:
    """
    b_s^{(j)}(α) = (1/π) ∫_0^{2π} cos(jψ) / (1 - 2α cosψ + α^2)^s dψ

    Integração numérica por média de amostras igualmente espaçadas (regra do trapézio).
    Requer 0 < α < 1 (na teoria L–L sempre escolhemos α = min(a_i,a_j)/max(a_i,a_j)).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha deve estar em (0,1) para cálculo de b_s^{(j)}.")
    psi = np.linspace(0.0, TWOPI, npsi, endpoint=False)
    denom = (1.0 - 2.0 * alpha * np.cos(psi) + alpha * alpha) ** s
    integrand = np.cos(j * psi) / denom
    integral = integrand.mean() * TWOPI  # média * comprimento do intervalo
    return (1.0 / np.pi) * integral

def b32_1(alpha: float) -> float:
    """Coeficiente b_{3/2}^{(1)}(α)."""
    return laplace_b_sj(alpha, s=1.5, j=1)

def b32_2(alpha: float) -> float:
    """Coeficiente b_{3/2}^{(2)}(α)."""
    return laplace_b_sj(alpha, s=1.5, j=2)

# ---------------------------
# Matrizes seculares A e B
# ---------------------------
def mean_motion_radyr(a_au: float, consts: Dict) -> float:
    """
    Movimento médio: n = sqrt(G * M0 / a^3) [rad/ano],
    com G e M0 lidos de /data/constants.json.
    """
    G = consts.get("G", 39.47841760435743)
    M0 = consts.get("M0", 1.0)
    return np.sqrt(G * M0 / (a_au ** 3))

def build_AB_from_bodies(bodies: List[Dict], consts: Dict) -> SecularMatrices:
    """
    Constrói as matrizes A e B (em graus/ano) para N corpos.

    Fórmulas (M&D 7.29–7.34). Para i ≠ j, com α_ij = min(a_i,a_j)/max(a_i,a_j):
      A_ij = - n_i * (1/4) * m_j * α_ij * b_{3/2}^{(2)}(α_ij)
      B_ij = + n_i * (1/4) * m_j * α_ij * b_{3/2}^{(1)}(α_ij)
    Diagonal:
      A_ii = + n_i * (1/4) * Σ_{j≠i} m_j * α_ij * b_{3/2}^{(1)}(α_ij)
      B_ii = - Σ_{j≠i} B_ij

    A e B são calculadas inicialmente em rad/ano e convertidas para deg/ano ao final.
    """
    m = np.array([b["mass"] for b in bodies], dtype=float)   # fração de M0
    a = np.array([b["a"] for b in bodies], dtype=float)      # AU
    n = np.array([mean_motion_radyr(ai, consts) for ai in a])  # rad/yr

    N = len(bodies)
    A = np.zeros((N, N), dtype=float)  # rad/yr
    B = np.zeros((N, N), dtype=float)

    # Termos fora da diagonal
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            alpha = min(a[i], a[j]) / max(a[i], a[j])  # α < 1
            A[i, j] = - n[i] * 0.25 * m[j] * alpha * b32_2(alpha)
            B[i, j] = + n[i] * 0.25 * m[j] * alpha * b32_1(alpha)

    # Termos diagonais
    for i in range(N):
        sA = 0.0
        for j in range(N):
            if i == j:
                continue
            alpha = min(a[i], a[j]) / max(a[i], a[j])
            sA += m[j] * alpha * b32_1(alpha)
        A[i, i] = n[i] * 0.25 * sA
        B[i, i] = - np.sum(B[i, :])  # soma da linha deve ser nula

    # rad/yr → deg/yr para saída intermediária
    A_degyr = A * RAD2DEG
    B_degyr = B * RAD2DEG
    return SecularMatrices(A_degyr=A_degyr, B_degyr=B_degyr)

# ---------------------------
# Diagonalização e salvamento
# ---------------------------
def secular_eigendecomp(M: SecularMatrices) -> EigenData:
    """
    Calcula autovalores e autovetores de A e B.
    Entrada: matrizes em deg/yr; Saída: autovalores em rad/yr.

    Convenção/normalização dos autovetores (consistência com Murray & Dermott):
    - Cada autovetor é normalizado para norma unitária.
    - O primeiro elemento de cada autovetor é tornado positivo (remove ambiguidade ±).
    """
    # Autovalores/autovetores (em deg/yr)
    g_degyr, Sg = np.linalg.eig(M.A_degyr)
    s_degyr, Ss = np.linalg.eig(M.B_degyr)

    # Normalização e convenção de sinais — matriz A
    for k in range(Sg.shape[1]):
        if Sg[0, k] < 0:
            Sg[:, k] *= -1.0
        norm = np.linalg.norm(Sg[:, k])
        if norm != 0.0:
            Sg[:, k] /= norm

    # Normalização e convenção de sinais — matriz B
    for k in range(Ss.shape[1]):
        if Ss[0, k] < 0:
            Ss[:, k] *= -1.0
        norm = np.linalg.norm(Ss[:, k])
        if norm != 0.0:
            Ss[:, k] /= norm

    # Conversão dos autovalores para rad/ano
    g_radyr = np.real(g_degyr) * DEG2RAD
    s_radyr = np.real(s_degyr) * DEG2RAD

    return EigenData(g_radyr=g_radyr, Sg=Sg, s_radyr=s_radyr, Ss=Ss)

def save_intermediates(M: SecularMatrices, eig: EigenData, out_dir: str = "output") -> None:
    """
    Salva intermediários para inspeção e validação:
      - output/A.csv, output/B.csv (deg/yr)
      - output/eigen_g_arcsec_per_yr.csv, output/eigen_s_arcsec_per_yr.csv
      - output/eigenvec_Sg.csv, output/eigenvec_Ss.csv
    """
    _ensure_output_dir(out_dir)
    np.savetxt(os.path.join(out_dir, "A.csv"), M.A_degyr, delimiter=",")
    np.savetxt(os.path.join(out_dir, "B.csv"), M.B_degyr, delimiter=",")

    g_arcsecyr = eig.g_radyr * ARCSEC_PER_RAD
    s_arcsecyr = eig.s_radyr * ARCSEC_PER_RAD
    np.savetxt(os.path.join(out_dir, "eigen_g_arcsec_per_yr.csv"), g_arcsecyr, delimiter=",")
    np.savetxt(os.path.join(out_dir, "eigen_s_arcsec_per_yr.csv"), s_arcsecyr, delimiter=",")
    np.savetxt(os.path.join(out_dir, "eigenvec_Sg.csv"), np.real(eig.Sg), delimiter=",")
    np.savetxt(os.path.join(out_dir, "eigenvec_Ss.csv"), np.real(eig.Ss), delimiter=",")

# ---------------------------
# Reconstrução temporal
# ---------------------------
def build_time_series(eig: EigenData,
                      h0: np.ndarray, k0: np.ndarray,
                      p0: np.ndarray, q0: np.ndarray,
                      t_years: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstrói (h,k,p,q) ao longo do tempo usando as soluções normais (M&D, eq. 7.42):

      z(t) = h + i k = Sg @ [C * exp(i * g * t)],   com C = Sg^{-1} z0,  z0 = h0 + i k0
      w(t) = p + i q = Ss @ [D * exp(i * s * t)],   com D = Ss^{-1} w0,  w0 = p0 + i q0

    Retorna os arrays (h, k, p, q) com dimensão (T, Ncorpos).
    """
    # Variáveis complexas iniciais
    z0 = h0 + 1j * k0
    w0 = p0 + 1j * q0

    # Constantes modais C e D por sistema linear
    C = np.linalg.solve(eig.Sg, z0)
    D = np.linalg.solve(eig.Ss, w0)

    # Evolução temporal das fases modais
    exp_g = np.exp(1j * np.outer(eig.g_radyr, t_years))  # (Nmodes, T)
    exp_s = np.exp(1j * np.outer(eig.s_radyr, t_years))  # (Nmodes, T)

    # Reconstrução (T, N)
    Zt = (eig.Sg @ (C[:, None] * exp_g)).T
    Wt = (eig.Ss @ (D[:, None] * exp_s)).T

    # Separação real/imag
    h = np.real(Zt)
    k = np.imag(Zt)
    p = np.real(Wt)
    q = np.imag(Wt)
    return h, k, p, q

# ---------------------------
# Conversões (h,k,p,q) ↔ (e,ϖ,i,Ω)
# ---------------------------
def hk_to_e_varpi(h: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converte (h,k) em (e, ϖ), com ϖ ∈ [0, 2π)."""
    e = np.sqrt(h*h + k*k)
    varpi = np.arctan2(k, h) % (2*np.pi)
    return e, varpi

def pq_to_i_Omega(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converte (p,q) em (I, Ω), com I em radianos e Ω ∈ [0, 2π)."""
    inc = 2.0 * np.sqrt(p*p + q*q)  # I (rad)
    Omega = np.arctan2(q, p) % (2*np.pi)
    return inc, Omega

# ---------------------------
# Condições iniciais a partir do JSON
# ---------------------------
def initial_conditions_from_json(bodies: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monta (h0, k0, p0, q0) a partir de (e, I, ω, Ω) definidos no JSON:
      ϖ = ω + Ω  (longitude do periélio)
      h0 = e cos ϖ,  k0 = e sin ϖ
      p0 = (I/2) cos Ω,  q0 = (I/2) sin Ω

    Retorna vetores (h0, k0, p0, q0) com dimensão (Ncorpos,).
    """
    e0 = np.array([b.get("e", 0.0) for b in bodies], dtype=float)
    I0 = np.radians(np.array([b.get("I", 0.0) for b in bodies], dtype=float))
    omega = np.radians(np.array([b.get("omega", 0.0) for b in bodies], dtype=float))
    Omega = np.radians(np.array([b.get("Omega", 0.0) for b in bodies], dtype=float))
    varpi0 = (omega + Omega) % (2*np.pi)

    h0 = e0 * np.cos(varpi0)
    k0 = e0 * np.sin(varpi0)
    p0 = 0.5 * I0 * np.cos(Omega)
    q0 = 0.5 * I0 * np.sin(Omega)
    return h0, k0, p0, q0
