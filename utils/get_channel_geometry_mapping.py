import numpy as np

def get_channel_geometry_mapping() -> np.ndarray:
    """
    Creates a 16x8 mapping matrix.
    Each cell (R, C) contains the 1D array index (0-127) of the channel 
    recorded at that physical position, based on the hardware multiplexing order.
    """
    mapping = np.zeros((16, 8), dtype=int)
    
    for R in range(16):
        for C in range(8):
            # Determine which of the 4 arrays we are in (0: TL, 1: TR, 2: BL, 3: BR)
            array_idx = (R // 8) * 2 + (C // 4)
            
            # Local coordinates within the specific 8x4 array
            r_local = R % 8
            c_local = C % 4
            
            # Sequence index (0 to 31) within the single array.
            # Hardware starts at bottom-right (r=7, c=3) and moves left, then up.
            m = 4 * (7 - r_local) + (3 - c_local)
            
            # Reconstruct the global channel index k (0-127)
            k = 4 * m + array_idx
            mapping[R, C] = k
            
    return mapping