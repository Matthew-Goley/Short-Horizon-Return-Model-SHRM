import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AAPL_PATH = RAW_DIR / "AAPL.csv"

print("loading successful")

def ComputeTicker(ticker, window):
    path = RAW_DIR / f"{ticker}.csv"
    df = pd.read_csv(path)

    # convert to datetime

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.reset_index(drop=True)

    #use only this columns

    df = df[["date", "open", "high", "low", "adj_close", "volume"]]
    df = df.dropna()

    # create features

    # FUTURE RETURNS (for model training)

    df["log_ret"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df = df.dropna(subset=["log_ret"])

    # PRICE SLOPE

    df["log_price"] = np.log(df["adj_close"])
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def rolling_slope(y):
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        return num / denom
    
    df["price_slope"] = df["log_price"].rolling(window).apply(rolling_slope, raw=False)
    df = df.dropna(subset=["price_slope"])

    # VOLATILITY Z SCORE

    # rolling volatility
    df["rolling_vol"] = df["log_ret"].rolling(window).std()
    df["future_ret"] = df["log_ret"].rolling(window).sum().shift(-window)
    df = df.dropna(subset=["future_ret"])
    df["target"] = df["future_ret"] / (df["rolling_vol"] + 1e-8)

    # z score
    vol_mean = df["rolling_vol"].rolling(window).mean()
    vol_std = df["rolling_vol"].rolling(window).std()

    df["volat_z"] = (df["rolling_vol"] - vol_mean) / vol_std
    df = df.dropna(subset=["volat_z"])

    # DRAWDOWN STATE

    # rolling peak
    rolling_max = df["adj_close"].rolling(window).max()

    # drawdown
    df["drawdown"] = (df["adj_close"] - rolling_max) / rolling_max

    # binary drawdown state (1 = in drawdow, 0 = near peak)
    df["draw_state"] = (df["drawdown"] < 0).astype(int)

    df = df.dropna(subset=["draw_state"])

    # LOG VOLUME

    df["log_volume"] = np.log(df["volume"] + 1)
    df = df.dropna(subset=["log_volume"])

    # debug
    df = df.dropna().reset_index(drop=True)
    assert df.isna().sum().sum() == 0

    # model_df = df[["price_slope", "volat_z", "draw_state", "log_volume", "target"]].copy()
    # model_df = model_df.reset_index(drop=True)

    return df

def create_sequences(df, seq_len, len_shift=1):
    X = []
    y_class = []
    y_ret = []
    
    features = df[["price_slope", "log_volume", "volat_z", "draw_state"]].values
    targets_class = df["target"].values
    targets_ret = df["future_ret"].values

    for i in range(seq_len - 1, len(df), len_shift):
        X.append(features[i - seq_len + 1 : i + 1])
        y_class.append(targets_class[i])
        y_ret.append(targets_ret[i])

    return np.array(X), np.array(y_class), np.array(y_ret)

def split_sets(X, y_class, y_ret):
        split_idx = int(0.7 * len(X))

        X_train = X[:split_idx]
        y_class_train = y_class[:split_idx]
        y_ret_train = y_ret[:split_idx]

        X_val = X[split_idx:]
        y_class_val = y_class[split_idx:]
        y_ret_val = y_ret[split_idx:]

        return X_train, y_class_train, y_ret_train, X_val, y_class_val, y_ret_val

def CCOMPUTEALL(window=12, seq_len = 24, len_shift=1, sector="tech"):
    tech = pd.read_csv(DATA_DIR / "tech.csv")["ticker"].tolist()
    financials = pd.read_csv(DATA_DIR / "financials.csv")["ticker"].tolist()
    consumer = pd.read_csv(DATA_DIR / "consumer.csv")["ticker"].tolist()
    staples = pd.read_csv(DATA_DIR / "staples.csv")["ticker"].tolist()
    healthcare = pd.read_csv(DATA_DIR / "healthcare.csv")["ticker"].tolist()
    energy = pd.read_csv(DATA_DIR / "energy.csv")["ticker"].tolist()
    industry = pd.read_csv(DATA_DIR / "industry.csv")["ticker"].tolist()

    sector_map = {
        "tech": tech,
        "financials": financials,
        "consumer": consumer,
        "staples": staples,
        "healthcare": healthcare,
        "energy": energy,
        "industry": industry,
        "all": tech + financials + consumer + staples + healthcare + energy + industry
    }

    tickers = sector_map[sector]
    total_ticker_count = 0
    master_df_list = []

    all_X_train = []
    all_y_class_train = []
    all_y_ret_train = []

    all_X_val = []
    all_y_class_val = []
    all_y_ret_val = []

    for ticker in tickers:
        path = RAW_DIR / f"{ticker}.csv"
        if not path.exists():
            print(f"skipping {ticker} - no CSV Found")
            continue

        total_ticker_count += 1
        df = ComputeTicker(ticker, window)
        
        # create sequences
        X, y_class, y_ret = create_sequences(df, seq_len, len_shift)

        X_train, y_class_train, y_ret_train, X_val, y_class_val, y_ret_val = split_sets(
            X, y_class, y_ret
        )

        all_X_train.append(X_train)
        all_y_class_train.append(y_class_train)
        all_y_ret_train.append(y_ret_train)

        all_X_val.append(X_val)
        all_y_class_val.append(y_class_val)
        all_y_ret_val.append(y_ret_val)

    print(f"Total Tickers: {total_ticker_count}")
    
    X_train = np.concatenate(all_X_train, axis=0)
    y_class_train = np.concatenate(all_y_class_train, axis=0)
    y_ret_train = np.concatenate(all_y_ret_train, axis=0)

    X_val = np.concatenate(all_X_val, axis=0)
    y_class_val = np.concatenate(all_y_class_val, axis=0)
    y_ret_val = np.concatenate(all_y_ret_val, axis=0)

    return X_train, y_class_train, y_ret_train, X_val, y_class_val, y_ret_val

if __name__ == "__main__":
    X_train, y_class_train, y_ret_train, X_val, y_class_val, y_ret_val = CCOMPUTEALL()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")