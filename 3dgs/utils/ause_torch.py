import torch

def ause_torch(error, uncertainty):
    """
    AUSE (area under specification error)

    The area between two different AUSC curves (area under specification curve), which are defined as the mean error after filtering out a fraction of pixels.
    The two different AUSC curves are fist the one using the true pixel error to filter out pixels and the second uses an uncertainty measure.
    """
    err_vec = torch.tensor(error).reshape(-1)
    unc_vec = torch.tensor(uncertainty).reshape(-1)

    vec_mask = torch.logical_and(err_vec != 0.0, unc_vec != 0.0)# should it be logical_or instead?
    if not torch.any(vec_mask):
        ause_err = torch.zeros(size=100, device=error.device)
        ause_err_by_var = torch.zeros(size=100, device=error.device)
        ause = 0.0
        return ause, ause_err, ause_err_by_var
    err_vec = err_vec[vec_mask]
    unc_vec = unc_vec[vec_mask]

    ratio_removed = torch.linspace(0, 0.999, 100, device=error.device)

    # AUSC for error
    err_vec_sorted, _ = torch.sort(err_vec)
    # Calculate the error when removing a fraction pixels with error
    n_valid_pixels = len(err_vec)
    ratio_idx = ((1-ratio_removed)*n_valid_pixels).to(int)[:-1]

    err_slices = torch.cumsum(err_vec_sorted, dim=0)[ratio_idx-1] / ratio_idx

    # AUSC for uncertainty
    _, var_vec_sorted_idxs = torch.sort(unc_vec)
    # Sort error by variance
    err_vec_sorted_by_var = err_vec[var_vec_sorted_idxs]

    err_by_var_slices = torch.cumsum(err_vec_sorted_by_var, dim=0)[ratio_idx-1] / ratio_idx

    # Normalize and append
    # (normalize by start value and not by max value
    # to avoid low AUSE value due to a large tail of the AUSC for uncertainty)
    start_val = err_slices[0]
    ause_err = err_slices / start_val

    ause_err_by_var = err_by_var_slices / start_val

    ause = torch.trapz(ause_err_by_var - ause_err, ratio_removed[:len(ause_err)])

    return ause.item(), ause_err, ause_err_by_var