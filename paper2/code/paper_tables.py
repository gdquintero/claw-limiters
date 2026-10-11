import numpy as np
def f(v):
    if not np.isfinite(v): return "--"
    m, e = f"{v:.2e}".split("e"); return f"${m}\\,(\\scriptstyle{{{int(e)}}})$"
def F(v):  # mantisa x 10^e compacto
    if not np.isfinite(v): return "--"
    m, e = f"{v:.2e}".split("e"); return f"${m}\\mathrm{{e}}{{{int(e)}}}$"
R2 = np.load("hyb2_res.npy", allow_pickle=True).item()
D1 = np.load("hyb1_conv.npy", allow_pickle=True).item()
out = []
# ---- Tabla 1: adveccion lineal
out.append(r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Linear advection with the one-step wave-propagation scheme: $L_1$ errors. Zalesak test ($N=200$, $t=20$) for three Courant numbers; wave packet ($N=200$, $t=10$, $\nu=0.5$); $\sin 6\pi x$ after one period ($\nu=0.9$, $N=3200$) and observed order between $N=800$ and $N=3200$.}\label{tab:adv}
\begin{tabular}{lcccccc}\toprule
 & \multicolumn{3}{c}{Zalesak} & wave packet & \multicolumn{2}{c}{$\sin 6\pi x$, $\nu=0.9$} \\ \cmidrule(lr){2-4}\cmidrule(lr){6-7}
limiter & $\nu=0.1$ & $\nu=0.5$ & $\nu=0.9$ & $\nu=0.5$ & $N=3200$ & order \\ \midrule""")
rows = [("MINBEE", "1", "MINBEE"), ("MC", "1", "MC"), ("$H_0$ (original hybrid)", "2", "H0 original"), ("Arora--Roe", "2", "Arora-Roe"),
        ("CFL-Superbee", "2", "CFL-Superbee"), (r"Kemm $\beta=2/3$", "2", "Kemm beta=2/3"),
        (r"TOPBEE-$\delta$, $\delta=1$", "2", "H3 d=1"), (r"TOPBEE-$\delta$, $\delta=0.3$", "2", "H3 d=0.3"),
        (r"TOPBEE-$\delta$, $\delta=0.05$", "2", "H3 d=0.05"), (r"TOPBEE-$\delta$, $\delta=0.03$", "2", "H3 d=0.03")]
for lab, src, n in rows:
    if src == "1":
        z = [D1["res"][n][j][0] for j in range(3)]; w = D1["res"][n][1][1]; s = D1["conv"][n][0.9]; s32, s8 = s[5], s[3]
    else:
        z = [R2[n][("zal", nu)] for nu in (0.1, 0.5, 0.9)]; w = R2[n][("wp", 0.5)]; s32, s8 = R2[n][("sin", 0.9, 3200)], R2[n][("sin", 0.9, 800)]
    out.append(f"{lab} & " + " & ".join(F(v) for v in z) + f" & {F(w)} & {F(s32)} & {np.log2(s8 / s32) / 2:.2f} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# ---- Tabla 2: Euler 1D
RE = np.load("euler_pareto.npy", allow_pickle=True).item()["R"]
out.append(r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Euler equations in 1D (CLAWPACK, Roe solver with Harten--Hyman entropy fix, CFL $0.9$): $L_1$ error of the density. Sod and Lax with $N=200$ (exact solution); Shu--Osher and Woodward--Colella with $N=400$ (reference: $N=12800$); smooth entropy wave with $N=200$ and observed order between $N=400$ and $N=800$. Limiters per wave family: acoustic / contact.}\label{tab:euler1d}
\begin{tabular}{lcccccc}\toprule
scheme (acoustic / contact) & Sod & Lax & Shu--Osher & W--C blast & entropy wave & order \\ \midrule""")
rows = [("MC / MC", "MC"), ("Superbee / Superbee", "superbee"), ("MINBEE / MINBEE", "MINBEE (idea 1)"), ("$H_0$ / $H_0$", "Hibrido original (idea 2)"),
        (r"AR$_{0.75}$ / AR$_{0.75}$ (Kemm)", "Kemm: AR t=.75 todas"), (r"AR$_{0.75}$ / CFL-Superbee (Kemm)", "Kemm: CS contacto + AR t=.75"),
        (r"$T_{1,0.75}$ / $T_{1,1}$", "H dc=1.0"), (r"$T_{1,0.75}$ / $T_{0.3,1}$", "H dc=0.3"), (r"$T_{1,0.75}$ / $T_{0.05,1}$", "H dc=0.05"), (r"$T_{1,0.75}$ / $T_{0.03,1}$", "H dc=0.03")]
for lab, n in rows:
    v = [RE[(n, p, 0.9)] for p in ("sod", "lax", "shuosher", "blast")] + [RE[(n, "ent", 0.9, 200)]]
    out.append(f"{lab} & " + " & ".join(F(x) for x in v) + f" & {np.log2(RE[(n, 'ent', 0.9, 400)] / RE[(n, 'ent', 0.9, 800)]):.2f} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
# ---- Tabla 3: Euler 2D
R2D = np.load("euler2d_res.npy", allow_pickle=True).item()["R"]
out.append(r"""\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Euler equations in 2D (CLAWPACK, unsplit with transverse corrections, CFL $0.9$): errors relative to MC (values $<1$ are better than MC). Riemann problem: $L_1$ density error at $200^2$ against an $800^2$ reference computed with MC; implosion: $L_1$ density error against a 1D radial reference; Gresho vortex: $L_1$ velocity error against the exact solution on $4N\times N$ grids. Limiters per wave family: acoustic / (entropy, shear).}\label{tab:euler2d}
\begin{tabular}{lcccccc}\toprule
scheme & Riemann $200^2$ & impl.\ $100^2$ & impl.\ $200^2$ & Gresho $N=20$ & $N=40$ & $N=80$ \\ \midrule""")
rows = [("MC", "MC"), ("Superbee", "superbee"), ("MINBEE", "MINBEE (idea 1)"), ("$H_0$", "Hibrido original (idea 2)"),
        (r"AR$_{0.75}$ (Kemm)", "Kemm: AR t=.75 todas"), (r"AR$_{0.75}$ / CFL-Superbee (Kemm)", "Kemm: CS lineales + AR t=.75"),
        (r"$T_{1,0.75}$ / $T_{1,1}$", "H dc=1"), (r"$T_{1,0.75}$ / $T_{0.3,1}$", "H dc=0.3"), (r"$T_{1,0.75}$ / $T_{0.05,1}$", "H dc=0.05")]
cols = [("quad-refMC", 200), ("impl", 100), ("impl", 200), ("gresho", 20), ("gresho", 40), ("gresho", 80)]
for lab, n in rows:
    out.append(f"{lab} & " + " & ".join(f"{R2D[(n, c, N)] / R2D[('MC', c, N)]:.2f}" for c, N in cols) + r" \\")
out.append(r"\bottomrule\end{tabular}\end{table}")
open("../tables.tex", "w").write("\n".join(out) + "\n")
print("\n".join(out))
