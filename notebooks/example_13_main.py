import pandas as pd
import numpy as np
from cyto.postprocessing.sparse_to_sparse import *
from dask.distributed import Client
import dask.dataframe as dd
import argparse
import logging
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import glob

def get_args():
    parser = argparse.ArgumentParser("example_13",
                                     description="Single Cell Killing Analysis Script")
    parser.add_argument(
        "-t","--tcell",
        help="Path to T cell tracked csv file",
        required=True,
    )
    parser.add_argument(
        "-c","--cancer",
        help="Path to Cancer cell tracked csv file",
        required=True,
    )
    parser.add_argument(
        "-o","--output",
        help="Output directory",
        required=True,
    )
    parser.add_argument(
        "-p","--precomputed",
        help="Load precomputed results from output directory",
        action="store_true"
    )
    parser.add_argument(
        "-v","--verbosity", 
        action='count', 
        default=0,
        help="Increase output verbosity (default: ERROR, -v: WARNING, -vv: INFO, -vvv: DEBUG)"
    )

    args = parser.parse_args()
    return args

def identify_killer_t_cell(df, cdi_thres=0.5):
    # Step1: Filter out cells with lower CDI values, to keep only live cells in the df
    df_filtered = df.groupby('track_id').filter(lambda x: ((x['CDI_smoothed'] > 0) & (x['CDI_smoothed'] < cdi_thres)).any())
    
    # Step 2: Filter for rows where 'value' is greater than or equal to cdi_thres
    df_filtered = df_filtered[df_filtered['CDI_smoothed'] >= cdi_thres]

    # Step 3: Group by 'track_id' and get the first occurrence of 'frame' for each group
    first_frame = df_filtered.sort_values(by=['frame']).groupby('track_id').first().reset_index()

    # Step 4: Extract the 'track_id' and 'frame' columns
    result = first_frame[['track_id', 'frame', 'contacting cell labels']]

    return result

def identify_multi_killer_t_cell(df,cdi_thres_lower=0.2,cdi_thres_upper=0.8):
    # Step1: Filter out cells with lower CDI values, to keep only live cells in the df
    df_filtered = df.groupby('track_id').filter(lambda x: ((x['CDI_smoothed'] > 0) & (x['CDI_smoothed'] < cdi_thres_lower)).any())
    
    # Step 2: Filter for rows where 'value' is within the cdi_thres bound
    df_filtered = df_filtered[(df_filtered['CDI_smoothed'] >= cdi_thres_lower) & (df_filtered['CDI_smoothed'] <= cdi_thres_upper)]

    # Step 3: Extract the 'track_id' and 'frame' columns
    result = df_filtered[['track_id', 'frame', 'contacting cell labels']]

    return result

def setup_logger(verbosity):
    # Create a logger
    logger = logging.getLogger(__name__)

    # Set logging level based on verbosity
    if verbosity >= 3:
        logger.setLevel(logging.DEBUG)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    elif verbosity == 1:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)
    
    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logger.level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    ch.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(ch)
    
    return logger

def main(args,logger):
    # start dask client
    client = Client()
    logger.info(client)

    # make output dir
    os.makedirs(os.path.join(args.output,"plot"),exist_ok=True)

    # Data loading
    if args.precomputed and glob.glob(os.path.join(args.output,"tcell.csv/*.part")):
        logger.info("Loading precomputed T Cell data: {}".format(os.path.join(args.output,"tcell.csv/*.part")))
        tcell_ddf = dd.read_csv(os.path.join(args.output,"tcell.csv/*.part"))
    else:
        TCELL_DATA_PATH = args.tcell
        logger.info("T Cell data path: {}".format(TCELL_DATA_PATH))
        tcell_ddf = dd.read_csv(TCELL_DATA_PATH)

        # Data normalization and CDI calculation
        tcell_ddf, tcell_ctfr_lp, tcell_ctfr_up = intensity_norm_percentile(tcell_ddf,channel="ctfr_mean", percentile=1)
        tcell_ddf, tcell_pi_lp, tcell_pi_up = intensity_norm_percentile(tcell_ddf,channel="pi_mean", percentile=1)
        
        logger.info("Saving T Cell results")
        tcell_ddf.to_csv(os.path.join(args.output,"tcell.csv"),index=False)
    tcell_df = tcell_ddf.compute()

    if args.precomputed and glob.glob(os.path.join(args.output,"cancer.csv/*.part")):
        logger.info("Loading precomputed Cancer Cell data: {}".format(os.path.join(args.output,"tcell.csv/*.part")))
        cancer_ddf = dd.read_csv(os.path.join(args.output,"cancer.csv/*.part"))
    else:
        CANCER_DATA_PATH = args.cancer
        logger.info("Cancer Cell data path: {}".format(CANCER_DATA_PATH))
        cancer_ddf = dd.read_csv(CANCER_DATA_PATH)

        cancer_ddf, cancer_gfp_lp, cancer_gfp_up = intensity_norm_percentile(cancer_ddf,channel="gfp_mean", percentile=1)
        cancer_ddf, cancer_pi_lp, cancer_pi_up = intensity_norm_percentile(cancer_ddf,channel="pi_mean", percentile=1)

        cancer_ddf = calculate_cdi(cancer_ddf, viability_col="gfp_mean_norm", death_col="pi_mean_norm")

        # CDI smoothing
        cancer_ddf = compute_savgol_filter(cancer_ddf, track_id_col='track_id', frame_col='frame', value_col='CDI', window_length=500, polyorder=3)

        logger.info("Saving Cancer Cell results")

        cancer_ddf.to_csv(os.path.join(args.output,"cancer.csv"),index=False)
    cancer_df = cancer_ddf.compute()

    ####################### MATCH CDI of Cancer Cells and Backtrace with T Cells in contact
    killer_t_cells = {}
    # iterate over different CDI thresholds
    for i in tqdm(range(2,9,1),desc="Querying single killer"):
        cdi_thres = 0.1*i
        if args.precomputed and os.path.exist(os.path.join(args.output,"killer_tcells_{:.1f}.csv".format(cdi_thres))):
            killer_t_cells[cdi_thres] = dd.read_csv(os.path.join(args.output,"killer_tcells_{:.1f}.csv".format(cdi_thres)))
            killer_t_cells[cdi_thres] = killer_t_cells[cdi_thres].compute()
        else:
            killer_t_cells_ = identify_killer_t_cell(cancer_df,cdi_thres)
            killer_t_cells[cdi_thres] = killer_t_cells_

            # save the killer t cell df
            killer_t_cells_.to_csv(os.path.join(args.output,"killer_tcells_{:.1f}.csv".format(cdi_thres)),index=False)

    multi_killer_t_cells = {}
    # iterate over different CDI thresholds
    for i in tqdm(range(2,5,1),desc="Querying multiple killer"):
        cdi_thres_lower = 0.1*i
        cdi_thres_upper = 1- 0.1*i
        if args.precomputed and os.path.exist(os.path.join(args.output,"multi_killer_tcells_{:.1f}.csv".format(cdi_thres_lower))):
            multi_killer_t_cells[cdi_thres_lower] = dd.read_csv("./tmp/multi_killer_tcells_{:.1f}.csv".format(cdi_thres_lower))
            multi_killer_t_cells[cdi_thres_lower] = multi_killer_t_cells[cdi_thres_lower].compute()
        else:
            multi_killer_t_cells_ = identify_multi_killer_t_cell(cancer_df,cdi_thres_lower=cdi_thres_lower,cdi_thres_upper=cdi_thres_upper)
            multi_killer_t_cells[cdi_thres_lower] = multi_killer_t_cells_

            # save the multiple killer t cell df
            multi_killer_t_cells_.to_csv(os.path.join(args.output,"multi_killer_tcells_{:.1f}.csv".format(cdi_thres_lower)),index=False)

    # plot on the killing counts
    fig, axs = plt.subplots(2,2,figsize=(8,8))
    ######### single killer #########
    axs[0,0].set_title("Raw Killing Count")
    axs[0,0].set_ylabel("Killing Count")
    axs[0,0].set_xlabel("Time/hr")

    # binned plot
    num_bins = 20
    axs[0,1].set_title("Time Binned Killing Count ({}s)".format(num_bins*10))
    axs[0,1].set_ylabel("Killing Count")
    axs[0,1].set_xlabel("Time/hr")

    for i in range(2,9,1):
        cdi_thres = i*0.1
        killing_count = killer_t_cells[cdi_thres].groupby("frame")["track_id"].count()

        axs[0,0].scatter(killing_count.index*10/3600, killing_count, s=1, label="CDI_thres={:.1f}".format(cdi_thres))

        # Bin the x-values
        bins = pd.cut(killing_count.index, bins=num_bins)

        # Group by the bins and sum the y-values for each bin
        binned_data = killing_count.groupby(bins).sum()

        # Compute the midpoints of each bin for plotting
        bin_centers = binned_data.index.map(lambda interval: interval.mid)
        axs[0,1].plot(bin_centers.to_numpy()*10/3600, binned_data, label="CDI_thres={:.1f}".format(cdi_thres))

    axs[0,0].set_xlim(left=0)
    axs[0,0].set_ylim(bottom=0)
    axs[0,1].set_xlim(left=0)
    axs[0,1].set_ylim(bottom=0)

    axs[0,0].legend()
    axs[0,1].legend()

    ######### multi killer #########
    axs[1,0].set_title("Raw Killing Count (Multi)")
    axs[1,0].set_ylabel("Killing Count")
    axs[1,0].set_xlabel("Time/hr")

    # binned plot
    axs[1,1].set_title("Time Binned Killing Count (Multi)({}s)".format(num_bins*10))
    axs[1,1].set_ylabel("Killing Count")
    axs[1,1].set_xlabel("Time/hr")

    for i in range(2,5,1):
        cdi_thres = 0.1*i

        multi_killing_count = multi_killer_t_cells[cdi_thres].groupby("frame")["track_id"].count()

        axs[1,0].scatter(multi_killing_count.index*10/3600, multi_killing_count, s=1, label="CDI_thres={:.1f}".format(cdi_thres))

        # Bin the x-values
        bins = pd.cut(multi_killing_count.index, bins=num_bins)

        # Group by the bins and sum the y-values for each bin
        binned_data = multi_killing_count.groupby(bins).sum()

        # Compute the midpoints of each bin for plotting
        bin_centers = binned_data.index.map(lambda interval: interval.mid)
        axs[1,1].plot(bin_centers.to_numpy()*10/3600, binned_data, label="CDI_band=[{:.1f},{:.1f}]".format(cdi_thres,1-cdi_thres))

    axs[1,0].set_xlim(left=0)
    axs[1,0].set_ylim(bottom=0)
    axs[1,1].set_xlim(left=0)
    axs[1,1].set_ylim(bottom=0)

    axs[1,0].legend()
    axs[1,1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.output,"plot","killing_count.png"),dpi=300)

    killer_t_cells = identify_killer_t_cell(cancer_ddf)
    killer_t_cells = killer_t_cells.compute()

    # put the killer t cells into a pandas df
    contacts = {
        "frame": [],
        "label": [],
        "kill": []
    }
    for idx, row in killer_t_cells.iterrows():
        tcells = row["contacting cell labels"]
        tcells = tcells.strip('[]')

        for j in tcells.split():
            contacts["frame"].append(row["frame"])
            contacts["label"].append(int(j))        
            contacts["kill"].append(True)

    contacts = pd.DataFrame.from_dict(contacts)

    # offset the label id by time
    contacts = dd.from_pandas(contacts,npartitions=100)

    contacts['label_time_offset'] = contacts.apply(lambda x: x["label"] + tcell_ddf[tcell_ddf['frame']<x["frame"]].count(), axis=1)
    contacts = contacts.compute()

6

if __name__ == "__main__":
    args = get_args()
    logger = setup_logger(args.verbosity)
    main(args,logger)