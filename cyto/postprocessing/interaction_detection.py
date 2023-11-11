import math
from tqdm import tqdm
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util, exposure, data, io
)

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class DetectInteractions():
    def __init__(self, verbose=True) -> None:
        """
        Detect interactions between two masked images
        
        Args:
            verbose (bool): Turn on or off the processing printout
        """
        self.verbose = verbose
    
    def __call__(self, data_target, data_other) -> Any:
        """
        Args:
            data_target: Target cells
            data_other: Interacting cells (e.g. T cells)
        """
        labels_target = data_target["labels"]
        labels_other = data_other["labels"]
        
        df_target = data_target["features"]
        df_other = data_other["features"]
        
        touchingCol = []
        
        for frameNum in tqdm(range(len(labels_target))):
            df_target_frame = df_target[df_target["frame"] == frameNum]
            df_other_frame = df_other[df_other["frame"] == frameNum]
            labels_target_frame = labels_target[frameNum]
            labels_other_frame = labels_other[frameNum]
            
            detected_target_cells = measure.regionprops(labels_target_frame)
            for index, row in df_target_frame.iterrows():
                y, x, shape = row["y"], row["x"], row["shape"]
                centerX, centerY = row["bbox_xsize"]//2, row["bbox_ysize"]//2
                left, right = row["bbox_xstart"], row["bbox_xstart"] + row["bbox_xsize"]
                bottom, top = row["bbox_ystart"], row["bbox_ystart"] + row["bbox_ysize"]
                
                labels_other_zone = labels_other_frame[bottom:top, left:right]
                labels_target_zone = labels_target_frame[bottom:top, left:right]
                labels_combined = labels_other_zone + labels_target_zone
                
                detected_cells_prior = measure.regionprops(labels_other_zone)
                detected_cells_post = measure.regionprops(labels_combined)
                
                before_cell_features = []
                for p in detected_cells_prior:
                    yP, xP = d.centroid
                    before_cell_features.append((p.area, xP, yP))
                
                after_cell_features = []
                for a in detected_cells_post:
                    yA, xA = a.centroid
                    after_cell_features.append((a.area, xA, yA))
                    
                touching_features, touching_cells = [], []
                for b in before_cell_features:
                    if b not in after_cell_features: # Interaction Detected #
                        touching_features.append(b)
                        
                for t in touching_features:
                    a, xT, yT = t # area, x, y
                    label_touching = labels_other_zone[int(yT), int(xT)]
                    for index, row in df_other_frame.iterrows():
                        if labels_other_frame[int(row["y"]), int(row["x"])] == label_touching:
                            touching_cells.append(row["label"]) # check this is actually the cell's "Id"
                touchingCol.append(touching_cells)
    df_target["touching"] = touchingCol


class VideoInteractions():
    def __init__(self, verbose=True) -> None:
        """
        Export GIF of two cells' interactions
        
        Args:
            verbose (bool): Turn on or off the processing printout
        """
        self.verbose = verbose
    
    def __call__(self, data_target, image_target, target_cell_label, data_interacting, image_interacting, interacting_cell_label,
         buffer_frames = 30, output_name = "", buffer_pixels = 20, fps = 15) -> Any:
        """
        Args:
            data_target (dataframe): Target cells
            image_target (numpy array): Image of the Target Cells (across all frames)
            target_cell_label (int): label of the target cell of interest
            data_interacting: Interacting cells (e.g. T cells)
            image_interacting (numpy array): Image of the Target Cells (across all frames)
            interacting_cell_label (int): label of the cell interacting with the target cell
            buffer_frames (int): How many frames before and after the first interaction to show in the GIF
            output_name (string): Where to write the GIF to (exclude .gif extension)
            buffer_pixels (int): How many pixels to leave around the interaction
            fps (int): Frames per second in the GIF
        """
        if output_name == "":
            output_name = f'Target={target_cell_label}_Interacting={interacting_cell_label}'
        labels_target = data_target["labels"]
        labels_interacting = data_interacting["labels"]

        df_target = data_target["features"]
        df_interacting = data_interacting["features"]

        interacting_cell_df = df_interacting[df_interacting['label'] == interacting_cell_label]
        target_cell_df = df_target[df_target['label'] == target_cell_label]
        y_interacting, x_interacting = list(interacting_cell_df['y']), list(interacting_cell_df['x'])
        xlims = [max(0, min(x_interacting) - buffer_pixels), min(image_interacting.shape[2], max(x_interacting) + buffer_pixels)]
        ylims = [max(0, min(y_interacting) - buffer_pixels), min(image_interacting.shape[1], max(y_interacting) + buffer_pixels)]

        target_dead_frames = []
        for index, row in target_cell_df.iterrows():
            if not row['alive']:
                target_dead_frames.append(row['frame'])

        snapshots = [] # frames
        for index, row in interacting_cell_df.iterrows():
            y, x, shape, frame, touching = row["y"], row["x"], (row["bbox_ysize"], row["bbox_ysize"]), row["frame"], row['touching']

            target_region = image_target[frame][ylims[0]:ylims[1], xlims[0]:xlims[1]]
            blank = np.zeros((target_region.shape))

            interacting_region = blank.copy()
            bottom, top = y - ylims[0] - shape[0]//2, y - ylims[0] + shape[0]//2
            left, right = x - xlims[0] - shape[1]//2, x - xlims[0] + shape[1]//2
            interacting_region[bottom:top, left:right] = image_interacting[frame][y - shape[0]//2: y + shape[0]//2, x - shape[1]//2: x + shape[1]//2]

            snap = [target_region/np.max(target_region), interacting_region/np.max(interacting_region), blank.copy()] # RGB
            if len(touching[0]) > 0:
                snap[0] += snap[1] / np.max(snap[1]) # Color Transformation

            if int(frame) in [int(f) for f in target_cell_df['frame']]:
                target_cell_df_frame = target_cell_df[target_cell_df['frame'] == frame]
                y, x = int(target_cell_df_frame['y']), int(target_cell_df_frame['x'])
                maskValue = labels_target[frame][y, x]
                labels_target_frame = labels_target[frame][:]
                labels_target_frame = (labels_target_frame == maskValue).astype(int)
                labels_target_frame = labels_target_frame[ylims[0]:ylims[1], xlims[0]:xlims[1]]
                if frame in target_dead_frames:
                    snap[0] += labels_target_frame/np.max(labels_target_frame)/2
                    snap[1] += labels_target_frame/np.max(labels_target_frame)/2
                    snap[2] += labels_target_frame/np.max(labels_target_frame)/2

                else:
                    snap[2] += labels_target_frame/np.max(labels_target_frame)/2

            snapshots.append(np.transpose(snap, (1, 2, 0)))

        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        a = snapshots[0]
        im = plt.imshow(a)
        def animate_func(i):
            if i % fps == 0:
                print( '.', end ='' )
            im.set_array(snapshots[i])
            return [im]

        fps=15
        anim = animation.FuncAnimation(
                                       fig, 
                                       animate_func,
                                       frames = len(snapshots)
                                       )
        anim.save(f'{output_name}.gif', fps=fps, dpi=100, savefig_kwargs={'transparent': True, 'facecolor': 'none'})