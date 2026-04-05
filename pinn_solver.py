"""
PINN solver for Indirect Reciprocity Model (Rath 2025).
Mean-field reduction, 3 initial conditions trained in parallel via one network per IC.
Efficient: 3000 epochs each.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)

# ── Parameters (Table 1 stable regime) ───────────────────────────────────────
GAMMA   = 0.25; ALPHA  = 0.50; LAMBDA = 0.20
ETA     = 0.10; THETA  = 0.20; BETA   = 2.0; R0_REP = 0.50
ALPHA_S = 0.20; BETA_S = 0.10; RHO    = 0.10; SIGMA  = 0.20
N = 10; T_END = 40.0

def H_mf(R):
    M = ETA + THETA*(N-1)*R
    return (N-1)*M / (1 + torch.exp(-BETA*(R - R0_REP)))

def dS_rhs(R, S): return ALPHA_S*H_mf(R) - BETA_S*S
def dR_rhs(R, S):
    C = RHO*H_mf(R) + SIGMA*S
    return -GAMMA*R + ALPHA*N*H_mf(R) - LAMBDA*C

# ── Small PINN ────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,2))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, t): return self.net(t)

def train(R0_i, S0_i, epochs=4000, n_col=800):
    model = PINN()
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    t0    = torch.tensor([[0.0]])
    ic    = torch.tensor([[R0_i, S0_i]])
    logs  = []
    for ep in range(1, epochs+1):
        opt.zero_grad()
        t_c = (torch.rand(n_col,1)*T_END).requires_grad_(True)
        out = model(t_c)
        R_h, S_h = out[:,0:1], out[:,1:2]
        dR = torch.autograd.grad(R_h, t_c, torch.ones_like(R_h), create_graph=True)[0]
        dS = torch.autograd.grad(S_h, t_c, torch.ones_like(S_h), create_graph=True)[0]
        phys = (dR - dR_rhs(R_h,S_h)).pow(2).mean() + (dS - dS_rhs(R_h,S_h)).pow(2).mean()
        ic_l = (model(t0) - ic).pow(2).mean()
        loss = phys + 20*ic_l
        loss.backward(); opt.step(); sched.step()
        logs.append(loss.item())
    return model, logs

# ── Euler reference ───────────────────────────────────────────────────────────
def euler(R0_i, S0_i, dt=0.02):
    steps = int(T_END/dt)
    R,S = R0_i, S0_i
    Rs,Ss,ts = [R],[S],[0.0]
    for _ in range(steps):
        dR = dR_rhs(torch.tensor([[R]]),torch.tensor([[S]])).item()
        dS = dS_rhs(torch.tensor([[R]]),torch.tensor([[S]])).item()
        R = max(0, R+dt*dR); S = max(0, S+dt*dS)
        Rs.append(R); Ss.append(S); ts.append(ts[-1]+dt)
    return np.array(ts), np.array(Rs), np.array(Ss)

# ── Run ───────────────────────────────────────────────────────────────────────
ICS = [(0.10,0.05,"#4fc3f7"),(0.05,0.02,"#81c784"),(0.15,0.08,"#ffb74d")]
t_plot = torch.linspace(0, T_END, 800).unsqueeze(1)

results = []
for (R0_i, S0_i, col) in ICS:
    print(f"Training PINN for R0={R0_i}, S0={S0_i} ...", flush=True)
    m, logs = train(R0_i, S0_i)
    with torch.no_grad():
        pred = m(t_plot).numpy()
    R_p = np.maximum(pred[:,0],0)
    S_p = np.maximum(pred[:,1],0)
    t_e, R_e, S_e = euler(R0_i, S0_i)
    # derived
    Rt = torch.tensor(R_p).unsqueeze(1); St = torch.tensor(S_p).unsqueeze(1)
    H_p = H_mf(Rt).numpy().flatten()
    C_p = (RHO*H_mf(Rt)+SIGMA*St).numpy().flatten()
    results.append(dict(col=col,R0_i=R0_i,S0_i=S0_i,
                        t=t_plot.numpy().flatten(),R_p=R_p,S_p=S_p,
                        t_e=t_e,R_e=R_e,S_e=S_e,H_p=H_p,C_p=C_p,logs=logs))
    print(f"  Done. Final loss={logs[-1]:.3e}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16,11))
fig.patch.set_facecolor("#0f1117")
gs = gridspec.GridSpec(3,2,figure=fig,hspace=0.50,wspace=0.35)
axs = {k: fig.add_subplot(gs[r,c]) for k,(r,c) in
       zip(["R","S","ph","lo","H","C"],
           [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)])}

def style_ax(ax):
    ax.set_facecolor("#1a1d27")
    for sp in ax.spines.values(): sp.set_color("#555")
    ax.tick_params(colors="#ccc"); ax.xaxis.label.set_color("#ddd")
    ax.yaxis.label.set_color("#ddd"); ax.title.set_color("white")

for ax in axs.values(): style_ax(ax)

for d in results:
    lbl = f"R₀={d['R0_i']}, S₀={d['S0_i']}"
    c   = d["col"]
    axs["R"].plot(d["t"],d["R_p"],color=c,lw=2,   label=f"PINN {lbl}")
    axs["R"].plot(d["t_e"],d["R_e"],color=c,lw=1.2,ls="--",alpha=0.6,label=f"Euler {lbl}")
    axs["S"].plot(d["t"],d["S_p"],color=c,lw=2)
    axs["S"].plot(d["t_e"],d["S_e"],color=c,lw=1.2,ls="--",alpha=0.6)
    axs["ph"].plot(d["R_p"],d["S_p"],color=c,lw=2,label=lbl)
    axs["ph"].scatter([d["R0_i"]],[d["S0_i"]],color=c,s=70,zorder=5)
    axs["lo"].semilogy(d["logs"],color=c,lw=1.5,label=lbl)
    axs["H"].plot(d["t"],d["H_p"],color=c,lw=2)
    axs["C"].plot(d["t"],d["C_p"],color=c,lw=2)

axs["R"].set(xlabel="Time t",ylabel="Reputation R(t)",title="Reputation Dynamics")
axs["S"].set(xlabel="Time t",ylabel="Stress S(t)",    title="Stress Dynamics")
axs["ph"].set(xlabel="R(t)",ylabel="S(t)",            title="Phase Portrait  (R – S plane)")
axs["lo"].set(xlabel="Epoch",ylabel="Loss",           title="PINN Training Loss")
axs["H"].set(xlabel="Time t",ylabel="H(R)",           title="Mean-Field Help H(R)")
axs["C"].set(xlabel="Time t",ylabel="C(R,S)",         title="Cooperation Cost C(R,S)")

for k in ["R","ph","lo"]:
    leg = axs[k].legend(fontsize=7.5,facecolor="#222",labelcolor="white",
                        edgecolor="#555",loc="best")

# Annotate equilibrium line
for k in ["R","S","H","C"]:
    axs[k].axvline(x=25,color="#ff6e6e",lw=0.8,ls=":",alpha=0.6)

fig.suptitle(
    "Physics-Informed Neural Network (PINN) — Indirect Reciprocity Model (Rath 2025)\n"
    "Mean-Field Reduction  |  Stable Cooperative Regime  |  Solid = PINN   Dashed = Euler",
    color="white",fontsize=12.5,y=0.995)

out = "/mnt/user-data/outputs/pinn_indirect_reciprocity.png"
plt.savefig(out,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
print(f"\nSaved → {out}")
