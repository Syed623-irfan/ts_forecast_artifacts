# ts_forecast_experiment.py
# Requirements: numpy, pandas, scikit-learn, matplotlib, statsmodels, torch

import sys, time, math, warnings, os
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1) Data generation (long-memory fractional integrated noise + multi-seasonality)
def frac_integration_white_noise(n, d, seed=0):
    np.random.seed(seed)
    wn = np.random.normal(size=n)
    w = [1.0]
    for k in range(1, n):
        w.append(w[-1] * ((k - 1 - d) / k))
    w = np.array(w)
    w_abs = np.abs(w)
    cutoff = np.where(w_abs < 1e-6)[0]
    trunc = cutoff[0] if len(cutoff) > 0 else n
    w = w[:trunc]
    series = np.convolve(wn, w)[:n]
    return series

N = 2000
d = 0.35
frac_part = frac_integration_white_noise(N, d, seed=42)
t = np.arange(N)
trend = 0.0006 * t
season1 = 2.0 * np.sin(2 * np.pi * t / 24)          # daily-like
season2 = 1.0 * np.sin(2 * np.pi * t / (24*7))      # weekly-like
noise_mult = 1.0 + 0.4 * np.sin(2 * np.pi * t / (24*365*2))
series = trend + season1 + season2 + noise_mult * frac_part
# regime shifts to make non-stationary
for shift_start in [600, 1200]:
    series[shift_start:shift_start+150] += np.linspace(0, 2.0 * (np.random.rand()-0.5), 150)

dates = pd.date_range(start="2000-01-01", periods=N, freq="H")
ts = pd.Series(series, index=dates, name="y")
print("Generated series length:", len(ts))

# 2) Train/val/test splits and scaling
train_len = 1200
val_len = 400
test_len = N - train_len - val_len
train = ts.iloc[:train_len].copy()
val = ts.iloc[train_len:train_len+val_len].copy()
test = ts.iloc[train_len+val_len:].copy()
print("Splits train/val/test:", len(train), len(val), len(test))

scaler = StandardScaler()
scaler.fit(train.values.reshape(-1,1))
def scale_series(s): return scaler.transform(s.values.reshape(-1,1)).flatten()
train_s = scale_series(train)
val_s = scale_series(val)
test_s = scale_series(test)

INPUT_LEN = 168  # 1 week
HORIZONS = [5,10,20]

def create_windows(series_array, input_len, horizon):
    X, Y = [], []
    n = len(series_array)
    for i in range(n - input_len - horizon + 1):
        X.append(series_array[i:i+input_len])
        Y.append(series_array[i+input_len:i+input_len+horizon])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

datasets = {}
for h in HORIZONS:
    X_train, Y_train = create_windows(train_s, INPUT_LEN, h)
    X_val, Y_val = create_windows(np.concatenate([train_s[-INPUT_LEN:], val_s]), INPUT_LEN, h)
    combined_for_test = np.concatenate([val_s[-INPUT_LEN:], test_s])
    X_test, Y_test = create_windows(combined_for_test, INPUT_LEN, h)
    datasets[h] = (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    print(f"H{h}: train {len(X_train)} val {len(X_val)} test {len(X_test)}")

# 3) Baseline: SARIMAX (fast) with fallback to seasonal-naive
baseline_results = {}
order = (1,0,1)
seasonal_order = (1,0,1,24)

sarima_fit = None
try:
    print("\nFitting SARIMAX (constrained maxiter)...")
    sarima = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order,
                                       enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=False, maxiter=25)
    print("SARIMAX fitted.")
except Exception as e:
    print("SARIMAX fit failed or too slow; will use fallback baseline. Error:", e)
    sarima_fit = None

for h in HORIZONS:
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets[h]
    preds = None
    if sarima_fit is not None:
        preds_list = []
        combined_start_index = train_len + val_len - INPUT_LEN
        for i in range(len(X_test)):
            obs_index = combined_start_index + i + INPUT_LEN - 1
            start = obs_index + 1
            end = obs_index + h
            try:
                fc = sarima_fit.get_prediction(start=start, end=end)
                preds_list.append(fc.predicted_mean.values)
            except Exception:
                preds_list.append(np.full(h, np.nan))
        preds = np.array(preds_list)
    if sarima_fit is None or np.isnan(preds).any():
        # Seasonal-naive fallback: repeat the last 24-hour block (original scale)
        print(f"Using seasonal-naive baseline for horizon {h}")
        preds = []
        combined = np.concatenate([val.values[-INPUT_LEN:], test.values])
        for i in range(len(X_test)):
            obs_idx = i + INPUT_LEN - 24
            if obs_idx < 0:
                rep = combined[-24:][:h]
            else:
                rep = combined[obs_idx:obs_idx+24][:h]
                if len(rep) < h:
                    rep = np.pad(rep, (0, h-len(rep)), 'edge')
            preds.append(np.array(rep))
        preds = np.array(preds)
    baseline_results[h] = (preds, Y_test)

# 4) Transformer seq2seq with attention (PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class TimeSeriesDataset(Dataset):
    def __init__(self,X,Y): self.X=X; self.Y=Y
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.Y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self,x): return x + self.pe[:,:x.size(1)].to(x.device)

class SimpleTransformer(nn.Module):
    def __init__(self,input_len,out_len,d_model=48,nhead=4,enc_layers=1,dec_layers=1,ff=128,dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1,d_model)
        self.pos = PositionalEncoding(d_model, max_len=max(input_len,out_len)+10)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.out = nn.Linear(d_model,1)
        self.dec_start = nn.Parameter(torch.randn(1,1,d_model))
        self.d_model = d_model
    def forward(self,src,tgt_len):
        b = src.size(0)
        src = self.input_proj(src.unsqueeze(-1)) * math.sqrt(self.d_model)
        src = self.pos(src)
        memory = self.encoder(src.permute(1,0,2))
        dec_input = self.dec_start.repeat(b,1,1)
        outputs=[]
        for _ in range(tgt_len):
            di = self.pos(dec_input)
            dec_out = self.decoder(di.permute(1,0,2), memory)
            last = dec_out[-1]
            out_step = self.out(last)
            outputs.append(out_step.squeeze(-1))
            next_token = last.unsqueeze(1)
            dec_input = torch.cat([dec_input, next_token], dim=1)
        return torch.stack(outputs, dim=1)

def train_transformer(h, datasets, epochs=20, batch_size=128, lr=1e-3):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets[h]
    train_loader = DataLoader(TimeSeriesDataset(X_train,Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val,Y_val), batch_size=batch_size, shuffle=False)
    model = SimpleTransformer(INPUT_LEN, h, d_model=48, nhead=4, enc_layers=1, dec_layers=1, ff=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    best_state = None; best_val=float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        tr_losses=[]
        for xb,yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad()
            preds = model(xb, tgt_len=h)
            loss = crit(preds, yb)
            loss.backward(); opt.step()
            tr_losses.append(loss.item())
        # val
        model.eval()
        val_losses=[]
        with torch.no_grad():
            for xb,yb in val_loader:
                xb=xb.to(device); yb=yb.to(device)
                preds = model(xb, tgt_len=h)
                val_losses.append(crit(preds,yb).item())
        mt = np.mean(tr_losses) if tr_losses else None
        mv = np.mean(val_losses) if val_losses else None
        if epoch%5==0 or epoch==1:
            print(f"H{h} E{epoch}/{epochs} train={mt:.6f} val={mv:.6f}")
        if mv is not None and mv < best_val:
            best_val = mv; best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(datasets[h][4]).to(device)
        preds = model(X_test_t, tgt_len=h).cpu().numpy()
    return preds, datasets[h][5], model

dl_results = {}
for h in HORIZONS:
    print("\nTraining Transformer for H", h)
    preds, Y_test, model = train_transformer(h, datasets, epochs=20, batch_size=128, lr=1e-3)
    dl_results[h] = (preds, Y_test, model)
    print(f"Done H{h}: preds shape {preds.shape}")

# 5) Evaluation and metrics (inverse-scaling for DL predictions)
def inv_scale(arr):
    arr = np.array(arr)
    flat = arr.reshape(-1,1)
    inv = scaler.inverse_transform(flat).reshape(arr.shape)
    return inv

report=[]
for h in HORIZONS:
    base_preds, Y_test = baseline_results[h]
    if base_preds.shape[1] != h:
        base_preds = np.array([np.pad(row, (0,max(0,h-row.shape[0])), 'edge') if len(row)<h else row[:h] for row in base_preds])
    Y_test_orig = inv_scale(Y_test)
    rmse_base = math.sqrt(mean_squared_error(Y_test_orig.flatten(), base_preds.flatten()))
    mae_base = mean_absolute_error(Y_test_orig.flatten(), base_preds.flatten())
    dl_preds_s, Y_test_s, _ = dl_results[h]
    dl_preds = inv_scale(dl_preds_s)
    rmse_dl = math.sqrt(mean_squared_error(Y_test_orig.flatten(), dl_preds.flatten()))
    mae_dl = mean_absolute_error(Y_test_orig.flatten(), dl_preds.flatten())
    report.append({"horizon":h, "rmse_baseline":rmse_base, "mae_baseline":mae_base, "rmse_transformer":rmse_dl, "mae_transformer":mae_dl})

report_df = pd.DataFrame(report).set_index("horizon")
print("\nEvaluation report:")
print(report_df)

# Per-step metrics example for h=20
h=20
dl_preds_s, Y_test_s, _ = dl_results[h]
dl_preds = inv_scale(dl_preds_s)
Y_test_orig = inv_scale(Y_test_s)
per_step_rmse = np.sqrt(np.mean((dl_preds - Y_test_orig)**2, axis=0))
print("\nPer-step RMSE for h=20:", per_step_rmse)

# Save artifacts
# out_dir = "./ts_forecast_artifacts"
out_dir = "C:/Users/Syed Irfan/Documents/ts_forecast_artifacts"
os.makedirs(out_dir, exist_ok=True)
report_df.to_csv(os.path.join(out_dir, "report_df.csv"))
with open(os.path.join(out_dir, "summary.txt"), "w") as f:
    f.write("Short summary and results. See report_df.csv for numeric metrics.\n")
torch.save({h: dl_results[h][2].state_dict() for h in HORIZONS}, os.path.join(out_dir, "transformers_states.pth"))
print("Artifacts saved to", out_dir)

# Quick plot
plt.figure(figsize=(8,3))
idx = min(5, dl_preds.shape[0]-1)
plt.plot(range(h), Y_test_orig[idx], label="True")
plt.plot(range(h), dl_preds[idx], '--', label="Transformer")
plt.plot(range(h), baseline_results[h][0][idx], ':', label="Baseline")
plt.legend(); plt.title("Example forecast (h=20)"); plt.tight_layout(); plt.show()
