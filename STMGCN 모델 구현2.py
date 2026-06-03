#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. 라이브러리

import numpy as np
import pandas as pd
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import haversine_distances

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# In[3]:


# 2. CSV 로딩

file_path = r"C:\Users\win11\Desktop\prE\04cargo.csv"

df_raw = pd.read_csv(file_path)

print(df_raw.shape)
print(df_raw.columns)
df_raw.head()


# In[4]:


# 3. 컬럼 정리

df = df_raw[["MMSI", "Date", "Lat", "Long", "SOG", "COG", "Heading"]].copy()

df = df.rename(columns={
    "Date": "timestamp",
    "Lat": "lat",
    "Long": "lon",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading"
})

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

for col in ["lat", "lon", "sog", "cog", "heading"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["MMSI", "timestamp", "lat", "lon", "sog", "cog", "heading"])

df = df[
    (df["lat"].between(-90, 90)) &
    (df["lon"].between(-180, 180)) &
    (df["sog"] >= 0) &
    (df["cog"].between(0, 360)) &
    (df["heading"].between(0, 360))
].copy()

df = df.sort_values(["MMSI", "timestamp"]).reset_index(drop=True)

print(df.shape)
df.head()


# In[5]:


# 4. 이동 선박 / 정지 선박 기준

MOVING_THRESHOLD = 0.5

moving_df = df[df["sog"] > MOVING_THRESHOLD].copy()
stationary_df = df[df["sog"] <= MOVING_THRESHOLD].copy()

print("전체 데이터:", len(df))
print("이동 선박 데이터:", len(moving_df))
print("정지/저속 선박 데이터:", len(stationary_df))


# In[6]:


# 5. Grid 설정
# Interaction Graph 학습용: 0.5 NM
# 최종 혼잡도 시각화용: 1.0 NM

TRAIN_GRID_NM = 0.5
VIS_GRID_NM = 1.0

TRAIN_LAT_GRID_SIZE = (TRAIN_GRID_NM * 1.852) / 111
TRAIN_LON_GRID_SIZE = (TRAIN_GRID_NM * 1.852) / (111 * np.cos(np.radians(35.0)))

VIS_LAT_GRID_SIZE = (VIS_GRID_NM * 1.852) / 111
VIS_LON_GRID_SIZE = (VIS_GRID_NM * 1.852) / (111 * np.cos(np.radians(35.0)))

print("TRAIN_LAT_GRID_SIZE:", TRAIN_LAT_GRID_SIZE)
print("TRAIN_LON_GRID_SIZE:", TRAIN_LON_GRID_SIZE)
print("VIS_LAT_GRID_SIZE:", VIS_LAT_GRID_SIZE)
print("VIS_LON_GRID_SIZE:", VIS_LON_GRID_SIZE)


# In[7]:


# 6. AIS 데이터를 5분 단위로 resampling

RESAMPLE_INTERVAL = "5min"

moving_df = moving_df.sort_values(["MMSI", "timestamp"]).reset_index(drop=True)

def resample_ais_5min(df):
    result = []

    for mmsi, group in df.groupby("MMSI"):
        group = group.sort_values("timestamp").copy()
        group = group.set_index("timestamp")

        numeric_cols = ["lat", "lon", "sog", "cog", "heading"]

        resampled = group[numeric_cols].resample(RESAMPLE_INTERVAL).mean()
        resampled = resampled.interpolate(method="linear")
        resampled = resampled.dropna()

        resampled["MMSI"] = mmsi
        resampled = resampled.reset_index()

        result.append(resampled)

    if len(result) == 0:
        return pd.DataFrame()

    return pd.concat(result, ignore_index=True)

moving_resampled = resample_ais_5min(moving_df)
moving_resampled = moving_resampled.sort_values(["MMSI", "timestamp"]).reset_index(drop=True)

print(moving_resampled.shape)
moving_resampled.head()


# In[8]:


# 7. 정규화

feature_cols = ["lat", "lon", "sog", "cog", "heading"]
target_cols = ["lat", "lon"]

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

moving_norm = moving_resampled.copy()

moving_norm[feature_cols] = x_scaler.fit_transform(moving_resampled[feature_cols])
moving_norm[target_cols] = y_scaler.fit_transform(moving_resampled[target_cols])

moving_norm.head()


# In[9]:


# 8. Interaction Graph 생성

def latlon_to_grid(lat, lon, lat_min, lon_min, lat_grid_size, lon_grid_size):
    row = int((lat - lat_min) / lat_grid_size)
    col = int((lon - lon_min) / lon_grid_size)
    return row, col


def build_interaction_counts(df):
    transition_counts = defaultdict(int)

    lat_min = df["lat"].min()
    lon_min = df["lon"].min()

    for mmsi, group in df.groupby("MMSI"):
        group = group.sort_values("timestamp")
        coords = group[["lat", "lon"]].values

        for i in range(len(coords) - 1):
            g1 = latlon_to_grid(
                coords[i][0],
                coords[i][1],
                lat_min,
                lon_min,
                TRAIN_LAT_GRID_SIZE,
                TRAIN_LON_GRID_SIZE
            )

            g2 = latlon_to_grid(
                coords[i + 1][0],
                coords[i + 1][1],
                lat_min,
                lon_min,
                TRAIN_LAT_GRID_SIZE,
                TRAIN_LON_GRID_SIZE
            )

            transition_counts[(g1, g2)] += 1

    return transition_counts, lat_min, lon_min


transition_counts, train_lat_min, train_lon_min = build_interaction_counts(moving_resampled)

print("interaction edge 수:", len(transition_counts))
print("train_lat_min:", train_lat_min)
print("train_lon_min:", train_lon_min)


# In[10]:


# 9. Next-step sequence 생성
# 과거 10개 시점 → 다음 5분 위치 예측

def create_next_step_sequences(raw_df, norm_df, seq_len=10):
    sequences = []

    for mmsi in raw_df["MMSI"].unique():
        raw_group = raw_df[raw_df["MMSI"] == mmsi].sort_values("timestamp").reset_index(drop=True)
        norm_group = norm_df[norm_df["MMSI"] == mmsi].sort_values("timestamp").reset_index(drop=True)

        if len(raw_group) <= seq_len:
            continue

        raw_data = raw_group[feature_cols].values.astype(np.float32)
        norm_data = norm_group[feature_cols].values.astype(np.float32)

        for i in range(len(raw_group) - seq_len):
            raw_seq = raw_data[i:i + seq_len]
            norm_seq = norm_data[i:i + seq_len]
            target = norm_data[i + seq_len][:2]

            sequences.append((raw_seq, norm_seq, target))

    return sequences


SEQ_LEN = 10

sequences = create_next_step_sequences(
    moving_resampled,
    moving_norm,
    seq_len=SEQ_LEN
)

print("sequence 수:", len(sequences))
print("raw_seq:", sequences[0][0].shape)
print("norm_seq:", sequences[0][1].shape)
print("target:", sequences[0][2].shape)


# In[11]:


# 10. Distance Graph + Interaction Graph 생성

def create_distance_graph(raw_seq, sigma=0.1):
    coords = raw_seq[:, :2]
    dist = haversine_distances(np.radians(coords))
    A = np.exp(-dist / sigma)
    return A.astype(np.float32)


def create_interaction_graph(raw_seq):
    N = len(raw_seq)
    A = np.zeros((N, N), dtype=np.float32)

    nodes = [
        latlon_to_grid(
            lat,
            lon,
            train_lat_min,
            train_lon_min,
            TRAIN_LAT_GRID_SIZE,
            TRAIN_LON_GRID_SIZE
        )
        for lat, lon, sog, cog, heading in raw_seq
    ]

    for i in range(N):
        for j in range(N):
            A[i, j] = transition_counts.get((nodes[i], nodes[j]), 0)

    if A.max() > 0:
        A = A / (A.max() + 1e-6)

    return A.astype(np.float32)


def create_multi_graph(raw_seq):
    A_distance = create_distance_graph(raw_seq)
    A_interaction = create_interaction_graph(raw_seq)

    graphs = np.stack([A_distance, A_interaction], axis=0)

    return graphs.astype(np.float32)


sample_graph = create_multi_graph(sequences[0][0])

print("sample_graph shape:", sample_graph.shape)


# In[12]:


# 11. Train/Test 분리

train_ratio = 0.8
train_size = int(len(sequences) * train_ratio)

train_sequences = sequences[:train_size]
test_sequences = sequences[train_size:]

print("train:", len(train_sequences))
print("test:", len(test_sequences))


# In[13]:


# 12. Dataset / DataLoader

class AISDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        raw_seq, norm_seq, target = self.sequences[idx]

        graphs = create_multi_graph(raw_seq)

        return (
            torch.tensor(norm_seq, dtype=torch.float32),
            torch.tensor(graphs, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


BATCH_SIZE = 32

train_dataset = AISDataset(train_sequences)
test_dataset = AISDataset(test_sequences)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

for X, graphs, y in train_loader:
    print("X:", X.shape)
    print("graphs:", graphs.shape)
    print("y:", y.shape)
    break


# In[14]:


# 13. STMGNN 모델 정의

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        B, T, _ = X.shape

        I = torch.eye(T, device=X.device).unsqueeze(0).expand(B, T, T)
        A_hat = A + I

        degree = torch.sum(A_hat, dim=-1)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)

        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        XW = self.linear(X)
        out = A_norm @ XW

        return out


class GraphAttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score_layer = nn.Linear(hidden_dim, 1)

    def forward(self, H_list):
        scores = []

        for H in H_list:
            score = self.score_layer(H).mean(dim=1)
            scores.append(score)

        scores = torch.cat(scores, dim=1)
        weights = F.softmax(scores, dim=1)

        fused = 0

        for i, H in enumerate(H_list):
            w = weights[:, i].view(-1, 1, 1)
            fused = fused + w * H

        return fused


class TemporalBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x

        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)

        return self.norm(x + residual)


class STMGNN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_graphs=2, output_dim=2):
        super().__init__()

        self.num_graphs = num_graphs

        self.gcn_layers = nn.ModuleList([
            GCNLayer(input_dim, hidden_dim)
            for _ in range(num_graphs)
        ])

        self.fusion = GraphAttentionFusion(hidden_dim)

        self.temporal1 = TemporalBlock(hidden_dim)
        self.temporal2 = TemporalBlock(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, graphs):
        H_list = []

        for i in range(self.num_graphs):
            A = graphs[:, i, :, :]
            H = self.gcn_layers[i](X, A)
            H_list.append(H)

        H = self.fusion(H_list)
        H = self.temporal1(H)
        H = self.temporal2(H)

        last_hidden = H[:, -1, :]

        out = self.output_layer(last_hidden)

        return out


# In[15]:


# 14. 학습 준비

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = STMGNN(
    input_dim=5,
    hidden_dim=64,
    num_graphs=2,
    output_dim=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# In[16]:


# 15. 학습

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, graphs, y in train_loader:
        X = X.to(device)
        graphs = graphs.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(X, graphs)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")


# In[17]:


# 16. 다음 5분 위치 예측 확인

model.eval()

with torch.no_grad():
    for X, graphs, y in test_loader:
        X = X.to(device)
        graphs = graphs.to(device)
        y = y.to(device)

        pred = model(X, graphs)

        pred_original = y_scaler.inverse_transform(pred.cpu().numpy())
        true_original = y_scaler.inverse_transform(y.cpu().numpy())

        print("예측값 Lat/Lon:")
        print(pred_original[:5])

        print("실제값 Lat/Lon:")
        print(true_original[:5])

        break


# In[18]:


# 17. Recursive Forecasting 함수
# forecast_minutes 값을 바꿔서 10분, 15분, 30분 예측 가능

def predict_next_position(model, raw_seq, norm_seq):
    graph = create_multi_graph(raw_seq)

    X = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
    graphs = torch.tensor(graph, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(X, graphs).cpu().numpy()

    pred_latlon = y_scaler.inverse_transform(pred_norm)[0]

    return pred_norm[0], pred_latlon


def recursive_forecast_single_vessel(model, raw_seq, norm_seq, forecast_minutes):
    steps = int(forecast_minutes / 5)

    raw_seq = raw_seq.copy()
    norm_seq = norm_seq.copy()

    for _ in range(steps):
        pred_norm_latlon, pred_latlon = predict_next_position(model, raw_seq, norm_seq)

        last_raw = raw_seq[-1].copy()
        last_norm = norm_seq[-1].copy()

        new_raw = last_raw.copy()
        new_raw[0] = pred_latlon[0]
        new_raw[1] = pred_latlon[1]

        new_norm = last_norm.copy()
        new_norm[0] = pred_norm_latlon[0]
        new_norm[1] = pred_norm_latlon[1]

        raw_seq = np.vstack([raw_seq[1:], new_raw])
        norm_seq = np.vstack([norm_seq[1:], new_norm])

    return raw_seq[-1, :2]


# In[55]:


# 18. 여러 base_time 예측 + 성능평가 + 시각화용 데이터 저장

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def assign_visual_grid(lat, lon, lat_min, lon_min):
    row = int((lat - lat_min) / VIS_LAT_GRID_SIZE)
    col = int((lon - lon_min) / VIS_LON_GRID_SIZE)
    return row, col


def make_grid_count_from_positions(positions, lat_min, lon_min):
    grid_counter = Counter()

    for lat, lon in positions:
        if np.isnan(lat) or np.isnan(lon):
            continue

        grid = assign_visual_grid(lat, lon, lat_min, lon_min)
        grid_counter[grid] += 1

    return grid_counter


def evaluate_grid_congestion(pred_grid_count, actual_grid_count, top_k=5):
    all_grids = sorted(set(pred_grid_count.keys()) | set(actual_grid_count.keys()))

    if len(all_grids) == 0:
        return None

    pred_counts = np.array([pred_grid_count.get(g, 0) for g in all_grids], dtype=float)
    actual_counts = np.array([actual_grid_count.get(g, 0) for g in all_grids], dtype=float)

    mae = np.mean(np.abs(pred_counts - actual_counts))
    rmse = np.sqrt(np.mean((pred_counts - actual_counts) ** 2))

    nonzero_mask = actual_counts > 0

    if nonzero_mask.sum() > 0:
        mape = np.mean(
            np.abs(
                (actual_counts[nonzero_mask] - pred_counts[nonzero_mask])
                / actual_counts[nonzero_mask]
            )
        ) * 100
    else:
        mape = np.nan

    if len(actual_counts) > 1 and np.std(actual_counts) > 0 and np.std(pred_counts) > 0:
        corr = np.corrcoef(actual_counts, pred_counts)[0, 1]
    else:
        corr = np.nan

    actual_top_k = set([all_grids[i] for i in np.argsort(actual_counts)[::-1][:top_k]])
    pred_top_k = set([all_grids[i] for i in np.argsort(pred_counts)[::-1][:top_k]])

    top_k_overlap = len(actual_top_k & pred_top_k)
    top_k_accuracy = top_k_overlap / top_k

    return {
        "grid_count": len(all_grids),
        "grid_mae": mae,
        "grid_rmse": rmse,
        "grid_mape_percent": mape,
        "pearson_correlation": corr,
        f"top_{top_k}_overlap": top_k_overlap,
        f"top_{top_k}_accuracy": top_k_accuracy
    }


def predict_next_position(model, raw_seq, norm_seq):
    graph = create_multi_graph(raw_seq)

    X = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
    graphs = torch.tensor(graph, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(X, graphs).cpu().numpy()

    pred_latlon = y_scaler.inverse_transform(pred_norm)[0]

    return pred_norm[0], pred_latlon


def recursive_forecast_single_vessel(model, raw_seq, norm_seq, forecast_minutes):
    steps = int(forecast_minutes / 5)

    raw_seq = raw_seq.copy()
    norm_seq = norm_seq.copy()

    for _ in range(steps):
        pred_norm_latlon, pred_latlon = predict_next_position(model, raw_seq, norm_seq)

        last_raw = raw_seq[-1].copy()
        last_norm = norm_seq[-1].copy()

        new_raw = last_raw.copy()
        new_raw[0] = pred_latlon[0]
        new_raw[1] = pred_latlon[1]

        new_norm = last_norm.copy()
        new_norm[0] = pred_norm_latlon[0]
        new_norm[1] = pred_norm_latlon[1]

        raw_seq = np.vstack([raw_seq[1:], new_raw])
        norm_seq = np.vstack([norm_seq[1:], new_norm])

    return raw_seq[-1, :2]


def generate_base_times(df, forecast_minutes, num_base_times=20):
    all_times = sorted(df["timestamp"].unique())

    start_idx = int(len(all_times) * 0.8)
    candidate_times = all_times[start_idx:]

    max_allowed_time = df["timestamp"].max() - pd.Timedelta(minutes=forecast_minutes)

    candidate_times = [
        t for t in candidate_times
        if t <= max_allowed_time
    ]

    if len(candidate_times) == 0:
        raise ValueError("forecast_minutes 이후 실제값이 존재하는 base_time이 없습니다.")

    if len(candidate_times) <= num_base_times:
        return candidate_times

    selected_idx = np.linspace(
        0,
        len(candidate_times) - 1,
        num_base_times
    ).astype(int)

    return [candidate_times[i] for i in selected_idx]


def make_prediction_inputs_at_base_time(df_raw, df_norm, base_time, seq_len=10):
    pred_inputs = []
    vessel_ids = []

    current_df = df_raw[df_raw["timestamp"] <= base_time].copy()

    if len(current_df) == 0:
        return pred_inputs, vessel_ids

    latest_idx = current_df.groupby("MMSI")["timestamp"].idxmax()
    current_snapshot = current_df.loc[latest_idx].copy()

    moving_snapshot = current_snapshot[current_snapshot["sog"] > MOVING_THRESHOLD].copy()

    for _, row in moving_snapshot.iterrows():
        mmsi = row["MMSI"]
        t = row["timestamp"]

        hist_raw = df_raw[
            (df_raw["MMSI"] == mmsi) &
            (df_raw["timestamp"] <= t)
        ].sort_values("timestamp").tail(seq_len)

        hist_norm = df_norm[
            (df_norm["MMSI"] == mmsi) &
            (df_norm["timestamp"] <= t)
        ].sort_values("timestamp").tail(seq_len)

        if len(hist_raw) < seq_len or len(hist_norm) < seq_len:
            continue

        raw_seq = hist_raw[feature_cols].values.astype(np.float32)
        norm_seq = hist_norm[feature_cols].values.astype(np.float32)

        pred_inputs.append((raw_seq, norm_seq))
        vessel_ids.append(mmsi)

    return pred_inputs, vessel_ids


def get_actual_future_positions(df, vessel_ids, base_time, forecast_minutes):
    actual_positions = []

    target_time = base_time + pd.Timedelta(minutes=forecast_minutes)

    for mmsi in vessel_ids:
        vessel_data = df[
            (df["MMSI"] == mmsi) &
            (df["timestamp"] >= target_time)
        ].sort_values("timestamp")

        if len(vessel_data) == 0:
            actual_positions.append([np.nan, np.nan])
            continue

        actual_row = vessel_data.iloc[0]

        actual_positions.append([
            actual_row["lat"],
            actual_row["lon"]
        ])

    return np.array(actual_positions)


def get_stationary_snapshot_at_base_time(df, base_time):
    current_df = df[df["timestamp"] <= base_time].copy()

    if len(current_df) == 0:
        return pd.DataFrame()

    latest_idx = current_df.groupby("MMSI")["timestamp"].idxmax()
    current_snapshot = current_df.loc[latest_idx].copy()

    stationary_snapshot = current_snapshot[
        current_snapshot["sog"] <= MOVING_THRESHOLD
    ].copy()

    return stationary_snapshot


# In[56]:


# 19. 여러 base_time에 대해 예측/평가 실행

forecast_minutes = 15   # 10, 15, 30 등으로 변경 가능
NUM_BASE_TIMES = 20

base_times = generate_base_times(
    moving_resampled,
    forecast_minutes=forecast_minutes,
    num_base_times=NUM_BASE_TIMES
)

print("base_time 개수:", len(base_times))
print(base_times[:5])

vis_grid_lat_min = df["lat"].min()
vis_grid_lon_min = df["lon"].min()

trajectory_results = []
grid_results = []
all_error_distances = []
prediction_visualization_data = []

model.eval()

for idx, base_time in enumerate(base_times):
    print(f"[{idx+1}/{len(base_times)}] base_time:", base_time)

    pred_inputs, vessel_ids = make_prediction_inputs_at_base_time(
        moving_resampled,
        moving_norm,
        base_time,
        seq_len=SEQ_LEN
    )

    if len(pred_inputs) == 0:
        print("예측 가능한 이동 선박 없음. skip")
        continue

    predicted_positions = []

    for raw_seq, norm_seq in pred_inputs:
        future_latlon = recursive_forecast_single_vessel(
            model,
            raw_seq,
            norm_seq,
            forecast_minutes=forecast_minutes
        )
        predicted_positions.append(future_latlon)

    predicted_positions = np.array(predicted_positions)

    actual_positions = get_actual_future_positions(
        moving_resampled,
        vessel_ids,
        base_time,
        forecast_minutes
    )

    valid_mask = ~np.isnan(actual_positions).any(axis=1)

    if valid_mask.sum() == 0:
        print("실제 비교 위치 없음. skip")
        continue

    pred_valid = predicted_positions[valid_mask]
    actual_valid = actual_positions[valid_mask]

    errors_km = haversine_km(
        actual_valid[:, 0],
        actual_valid[:, 1],
        pred_valid[:, 0],
        pred_valid[:, 1]
    )

    all_error_distances.extend(errors_km.tolist())

    trajectory_results.append({
        "base_time": base_time,
        "forecast_minutes": forecast_minutes,
        "valid_ship_count": len(actual_valid),
        "mean_error_km": np.mean(errors_km),
        "median_error_km": np.median(errors_km),
        "p90_error_km": np.percentile(errors_km, 90),
        "max_error_km": np.max(errors_km),
        "mean_error_nm": np.mean(errors_km) / 1.852,
        "median_error_nm": np.median(errors_km) / 1.852,
        "p90_error_nm": np.percentile(errors_km, 90) / 1.852,
        "max_error_nm": np.max(errors_km) / 1.852,
    })

    stationary_snapshot = get_stationary_snapshot_at_base_time(df, base_time)

    pred_moving_grid_count = make_grid_count_from_positions(
        pred_valid,
        vis_grid_lat_min,
        vis_grid_lon_min
    )

    actual_moving_grid_count = make_grid_count_from_positions(
        actual_valid,
        vis_grid_lat_min,
        vis_grid_lon_min
    )

    stationary_grid_count = Counter()

    for _, row in stationary_snapshot.iterrows():
        grid = assign_visual_grid(
            row["lat"],
            row["lon"],
            vis_grid_lat_min,
            vis_grid_lon_min
        )
        stationary_grid_count[grid] += 1

    pred_total_grid_count = Counter(pred_moving_grid_count)
    pred_total_grid_count.update(stationary_grid_count)

    actual_total_grid_count = Counter(actual_moving_grid_count)
    actual_total_grid_count.update(stationary_grid_count)

    grid_result = evaluate_grid_congestion(
        pred_total_grid_count,
        actual_total_grid_count,
        top_k=5
    )

    if grid_result is not None:
        grid_result["base_time"] = base_time
        grid_result["forecast_minutes"] = forecast_minutes
        grid_results.append(grid_result)

        prediction_visualization_data.append({
            "base_time": base_time,
            "forecast_minutes": forecast_minutes,
            "pred_valid": pred_valid.copy(),
            "actual_valid": actual_valid.copy(),
            "pred_grid_count": pred_total_grid_count.copy(),
            "actual_grid_count": actual_total_grid_count.copy()
        })

print("완료")
print("시각화 가능한 base_time 수:", len(prediction_visualization_data))


# In[57]:


# 20. 성능평가 요약표

trajectory_results_df = pd.DataFrame(trajectory_results)
grid_results_df = pd.DataFrame(grid_results)

trajectory_summary_df = pd.DataFrame([{
    "forecast_minutes": forecast_minutes,
    "base_time_count": len(trajectory_results_df),
    "total_valid_ship_count": trajectory_results_df["valid_ship_count"].sum(),
    "mean_error_km_avg": trajectory_results_df["mean_error_km"].mean(),
    "median_error_km_avg": trajectory_results_df["median_error_km"].mean(),
    "p90_error_km_avg": trajectory_results_df["p90_error_km"].mean(),
    "mean_error_nm_avg": trajectory_results_df["mean_error_nm"].mean(),
    "median_error_nm_avg": trajectory_results_df["median_error_nm"].mean(),
    "p90_error_nm_avg": trajectory_results_df["p90_error_nm"].mean(),
}])

grid_summary_df = pd.DataFrame([{
    "forecast_minutes": forecast_minutes,
    "base_time_count": len(grid_results_df),
    "grid_mae_avg": grid_results_df["grid_mae"].mean(),
    "grid_rmse_avg": grid_results_df["grid_rmse"].mean(),
    "grid_mape_percent_avg": grid_results_df["grid_mape_percent"].mean(),
    "pearson_correlation_avg": grid_results_df["pearson_correlation"].mean(),
    "top_5_accuracy_avg": grid_results_df["top_5_accuracy"].mean(),
}])

print("Trajectory-level Summary")
display(trajectory_summary_df)

print("Grid-level Summary")
display(grid_summary_df)


# In[58]:


# 21. 저장된 base_time 목록 확인

print("저장된 base_time 개수:", len(prediction_visualization_data))

for i, item in enumerate(prediction_visualization_data):
    print(i, item["base_time"])


# In[59]:


# 22. 시각화할 base_time 선택

VIS_IDX = 0   # 보고 싶은 번호로 변경

vis_data = prediction_visualization_data[VIS_IDX]

selected_base_time = vis_data["base_time"]
selected_forecast_minutes = vis_data["forecast_minutes"]

pred_valid = vis_data["pred_valid"]
actual_valid = vis_data["actual_valid"]

pred_grid_count = vis_data["pred_grid_count"]
actual_grid_count = vis_data["actual_grid_count"]

print("선택 base_time:", selected_base_time)
print("forecast_minutes:", selected_forecast_minutes)
print("비교 선박 수:", len(pred_valid))


# In[60]:


# 23. 예측 위치 vs 실제 위치 산점도

plt.figure(figsize=(9, 7))

plt.scatter(
    actual_valid[:, 1],
    actual_valid[:, 0],
    s=40,
    marker="x",
    label="Actual Future Position"
)

plt.scatter(
    pred_valid[:, 1],
    pred_valid[:, 0],
    s=40,
    marker="*",
    label="Predicted Future Position"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(
    f"Predicted vs Actual Vessel Positions\n"
    f"base_time={selected_base_time}, forecast={selected_forecast_minutes}min"
)
plt.legend()
plt.grid(True)

ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.ticklabel_format(style="plain", axis="both")

plt.show()


# In[61]:


# 24. 실제 위치와 예측 위치 연결선

plt.figure(figsize=(9, 7))

for i in range(len(actual_valid)):
    plt.plot(
        [actual_valid[i, 1], pred_valid[i, 1]],
        [actual_valid[i, 0], pred_valid[i, 0]],
        linewidth=0.8,
        alpha=0.4
    )

plt.scatter(
    actual_valid[:, 1],
    actual_valid[:, 0],
    s=35,
    marker="x",
    label="Actual Future Position"
)

plt.scatter(
    pred_valid[:, 1],
    pred_valid[:, 0],
    s=35,
    marker="*",
    label="Predicted Future Position"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(
    f"Prediction Error Lines\n"
    f"base_time={selected_base_time}, forecast={selected_forecast_minutes}min"
)
plt.legend()
plt.grid(True)

ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax.ticklabel_format(style="plain", axis="both")

plt.show()


# In[62]:


# 25. Heatmap 변환 함수

def grid_count_to_heatmap(grid_count):
    if len(grid_count) == 0:
        return np.zeros((1, 1))

    max_row = max([g[0] for g in grid_count.keys()])
    max_col = max([g[1] for g in grid_count.keys()])

    heatmap = np.zeros((max_row + 1, max_col + 1))

    for grid, count in grid_count.items():
        heatmap[grid[0], grid[1]] = count

    return heatmap


actual_heatmap = grid_count_to_heatmap(actual_grid_count)
pred_heatmap = grid_count_to_heatmap(pred_grid_count)


# In[63]:


# 26. 실제 혼잡도 Heatmap

plt.figure(figsize=(10, 8))

plt.imshow(
    actual_heatmap,
    origin="lower",
    aspect="auto"
)

plt.colorbar(label="Actual Vessel Count")
plt.xlabel("Grid Col")
plt.ylabel("Grid Row")
plt.title(
    f"Actual Congestion\n"
    f"base_time={selected_base_time}, forecast={selected_forecast_minutes}min"
)

plt.show()


# In[64]:


# 27. 예측 혼잡도 Heatmap

plt.figure(figsize=(10, 8))

plt.imshow(
    pred_heatmap,
    origin="lower",
    aspect="auto"
)

plt.colorbar(label="Predicted Vessel Count")
plt.xlabel("Grid Col")
plt.ylabel("Grid Row")
plt.title(
    f"Predicted Congestion\n"
    f"base_time={selected_base_time}, forecast={selected_forecast_minutes}min"
)

plt.show()


# In[65]:


# 28. 실제 vs 예측 혼잡도 비교

max_row = max(actual_heatmap.shape[0], pred_heatmap.shape[0])
max_col = max(actual_heatmap.shape[1], pred_heatmap.shape[1])

actual_pad = np.zeros((max_row, max_col))
pred_pad = np.zeros((max_row, max_col))

actual_pad[:actual_heatmap.shape[0], :actual_heatmap.shape[1]] = actual_heatmap
pred_pad[:pred_heatmap.shape[0], :pred_heatmap.shape[1]] = pred_heatmap

vmax = max(actual_pad.max(), pred_pad.max())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im0 = axes[0].imshow(
    actual_pad,
    origin="lower",
    aspect="auto",
    vmin=0,
    vmax=vmax
)
axes[0].set_title("Actual Congestion")
axes[0].set_xlabel("Grid Col")
axes[0].set_ylabel("Grid Row")

im1 = axes[1].imshow(
    pred_pad,
    origin="lower",
    aspect="auto",
    vmin=0,
    vmax=vmax
)
axes[1].set_title("Predicted Congestion")
axes[1].set_xlabel("Grid Col")
axes[1].set_ylabel("Grid Row")

fig.colorbar(im1, ax=axes.ravel().tolist(), label="Vessel Count")

plt.suptitle(
    f"Actual vs Predicted Congestion\n"
    f"base_time={selected_base_time}, forecast={selected_forecast_minutes}min"
)

plt.show()


# In[66]:


# 29. 혼잡도 오차 Heatmap

error_heatmap = np.abs(pred_pad - actual_pad)

plt.figure(figsize=(10, 8))

plt.imshow(
    error_heatmap,
    origin="lower",
    aspect="auto"
)

plt.colorbar(label="Absolute Error of Vessel Count")
plt.xlabel("Grid Col")
plt.ylabel("Grid Row")
plt.title(
    f"Congestion Prediction Error\n"
    f"base_time={selected_base_time}, forecast={selected_forecast_minutes}min"
)

plt.show()


# In[67]:


# 30. 여러 base_time 성능 변화 그래프

plt.figure(figsize=(10, 5))

plt.plot(
    trajectory_results_df["base_time"],
    trajectory_results_df["mean_error_km"],
    marker="o"
)

plt.xlabel("Base Time")
plt.ylabel("Mean Error (km)")
plt.title(f"Mean Trajectory Error by Base Time ({forecast_minutes} min)")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    grid_results_df["base_time"],
    grid_results_df["grid_mae"],
    marker="o"
)

plt.xlabel("Base Time")
plt.ylabel("Grid MAE")
plt.title(f"Grid Congestion MAE by Base Time ({forecast_minutes} min)")
plt.grid(True)
plt.xticks(rotation=45)
plt.show()


# In[ ]:




