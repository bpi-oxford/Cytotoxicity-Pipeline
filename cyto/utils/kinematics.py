import pandas as pd
import dask.dataframe as dd
import numpy as np
import dask.array as da

def cal_kinematics(tracks, x_col="x", y_col="y", frame_col="frame", track_id_col="track_id"):
    if isinstance(tracks, pd.DataFrame):
        tracks_ddf = dd.from_pandas(tracks)
    else:
        tracks_ddf = tracks

    grouped = tracks_ddf.groupby(track_id_col)
    sorted_grouped = grouped.apply(lambda x: x.sort_values(frame_col), meta=tracks_ddf).reset_index(drop=True)
    tracks_ddf = sorted_grouped.assign(
        dx_from_previous_point=sorted_grouped.groupby(track_id_col)[x_col].shift(-1)-sorted_grouped[x_col],
        dy_from_previous_point=sorted_grouped.groupby(track_id_col)[y_col].shift(-1)-sorted_grouped[y_col],
        dt=sorted_grouped.groupby(track_id_col)[frame_col].shift(-1)-sorted_grouped[frame_col]
        )
    
    # tracks_ddf["dx_from_previous_point"] = tracks_ddf["dx_from_previous_point"].astype(float)
    # tracks_ddf["dy_from_previous_point"] = tracks_ddf["dy_from_previous_point"].astype(float)
    # tracks_ddf["dt"] = tracks_ddf["dt"].astype(float)
    
    tracks_ddf = tracks_ddf.rename(columns={
        "dx_from_previous_point": "dx from previous point", 
        "dy_from_previous_point": "dy from previous point",
        "dt": "dt"
        }).fillna(0)

    print("a")

    # cumulative time
    tracks_ddf["dt acc"] = tracks_ddf.groupby([track_id_col])[frame_col].transform("min").reset_index(drop=True)
    tracks_ddf["dt acc"] = tracks_ddf[frame_col] - tracks_ddf["dt acc"]
    
    tracks_ddf["displacement from previous point"] = tracks_ddf[["dx from previous point","dy from previous point"]].apply(lambda row: np.linalg.norm(row, ord=2), axis=1, meta=('displacement from previous point', 'f8')).fillna(0)

    tracks_ddf["dx from origin"] = tracks_ddf.groupby([track_id_col])['dx from previous point'].cumsum().fillna(0)
    tracks_ddf["dy from origin"] = tracks_ddf.groupby([track_id_col])['dy from previous point'].cumsum().fillna(0)
    tracks_ddf["displacement from origin"] = tracks_ddf[["dx from origin","dy from origin"]].apply(lambda row: np.linalg.norm(row, ord=2), axis=1, meta=('displacement from origin', 'f8'))
    tracks_ddf["dx acc"] = tracks_ddf['dx from previous point'].apply(lambda x: abs(x)).fillna(0) # abs displacement
    tracks_ddf["dy acc"] = tracks_ddf['dy from previous point'].apply(lambda x: abs(x)).fillna(0) # abs displacement
    tracks_ddf["dx acc"] = tracks_ddf.groupby([track_id_col])['dx acc'].cumsum().fillna(0)
    tracks_ddf["dy acc"] = tracks_ddf.groupby([track_id_col])['dy acc'].cumsum().fillna(0)
    tracks_ddf["distance traveled"] = tracks_ddf.groupby([track_id_col])['displacement from previous point'].cumsum().fillna(0)
    tracks_ddf["path efficiency"] = (tracks_ddf["distance traveled"]/tracks_ddf["displacement from origin"]).fillna(0).replace([np.inf, -np.inf], 0)

    print("b")

    # velocity
    tracks_ddf["vel_x"] = (tracks_ddf["dx from previous point"]/tracks_ddf["dt"]).fillna(0).replace([np.inf, -np.inf], 0)
    tracks_ddf["vel_y"] = (tracks_ddf["dy from previous point"]/tracks_ddf["dt"]).fillna(0).replace([np.inf, -np.inf], 0)
    tracks_ddf["speed"] = (tracks_ddf["displacement from previous point"]/tracks_ddf["dt"]).fillna(0).replace([np.inf, -np.inf], 0)
    tracks_ddf["average speed"] = (tracks_ddf["distance traveled"]/tracks_ddf["dt acc"]).fillna(0).replace([np.inf, -np.inf], 0)

    

    return tracks_ddf.compute()

def cal_msd(tracks):
    if isinstance(tracks, pd.DataFrame):
        tracks_ddf = dd.from_pandas(tracks, chunksize=2000)
    else:
        tracks_ddf = tracks

    tracks_ddf["dr2"] = tracks_ddf["displacement from origin"]**2

    msd_mean = tracks_ddf.groupby("frame")["dr2"].mean().to_frame(name='msd')
    msd_sd = tracks_ddf.groupby("frame")["dr2"].std().to_frame(name='sd')
    msd = dd.concat([msd_mean, msd_sd], axis=1)
    msd = msd.reset_index().sort_values(by=["frame"])

    return msd.compute()

def compute_msd_time_lag(time_lag, data):
    msd_list = []
    for t in data['frame'].unique():
        t1 = t
        t2 = t + time_lag

        if t2 in data['frame'].values:
            displacement_sq = data[(data['frame'] == t2)]['displacement squared'].values
            msd = np.mean(np.abs(displacement_sq - data[(data['frame'] == t1)]['displacement squared'].values))
            msd_list.append(msd)
    return np.mean(msd_list) if msd_list else np.nan, np.std(msd_list) if msd_list else np.nan