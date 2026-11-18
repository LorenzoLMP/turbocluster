import numpy as np
import cupy as cp


def volume_integral(snap, qt, indices):
    if isinstance(qt, str):
        qt = snap[qt]

    # using paicos omp-vectorized routines
    a = snap.get_sum_of_array((qt * snap["0_Volume"])[indices])

    return a


def volume_average(snap, qt, indices):
    if isinstance(qt, str):
        qt = snap[qt]

    # using paicos omp-vectorized routines
    a = snap.get_sum_of_array((qt * snap["0_Volume"])[indices])
    vol = snap.get_sum_of_array((snap["0_Volume"])[indices])

    return a / vol


def extract_turbulent_var(
    snap,
    sf,
    variable,
    filter_length,
    weight,
    filter_type="gaussian",
    iterative=False,
    selection=None,
):

    filt_var = sf.filter_variable(
        variable,
        filter_length,
        weight=weight,
        filter_type=filter_type,
        iterative=iterative,
        selection=selection,
    )

    print(
        "min/max/avg occupancy cartesian tiling %d / %d / %.2f"
        % (
            sf.tile.particles_per_tile.min(),
            sf.tile.particles_per_tile.max(),
            cp.sum(sf.tile.particles_per_tile)
            / np.prod(sf.tile.particles_per_tile.shape),
        )
    )
    if isinstance(variable, str):
        variable = snap[variable]

    smoothVar = np.zeros_like(variable)
    turbVar = np.zeros_like(variable)

    smoothVar[sf.index] = filt_var
    turbVar[sf.index] = variable[sf.index] - filt_var
    turbVar[~sf.indicesFirstPass * sf.index] = (
        0  # this is for the thin buffer, which we set to zero
    )

    return smoothVar, turbVar


def decompose_magnetic_energy(
    snap,
    sf,
    mag_field,
    filter_length,
    weight,
    filter_type,
    iterative=False,
    recycle=False,
    snap_out=None,
    selection=None,
):
    #################################################################
    # This function decomposes the filtered total magnetic energy density
    # according to Perrone+2026:
    # \epsilon_B = \langle e_B \rangle_l = 1/(8\pi) \langle B^2 \rangle_l
    # = \epsilon_{B,bulk} + \epsilon_{B,turb}
    # where
    # \epsilon_{B,bulk} = 1/(8\pi) (\langle \bm B \rangle_l)^2
    # is the energy density of the mean magnetic field,
    # and
    # \epsilon_{B,turb} =  1/(8\pi) \sum_i \mu_2 (B_i, B_i)
    # with
    # \mu_2 (B_i, B_j) = \langle B_i B_j \rangle_l
    # - \langle B_i \rangle_l \langle B_j \rangle_l
    # is the energy density of the turbulent mag fluctuations.
    # It is a scalar quantity.
    #################################################################

    if isinstance(mag_field, str):
        mag_field = snap[mag_field]

    #####################
    # 1. Start by computing \langle e_B \rangle_l
    # = 1/(8\pi) \langle B^2 \rangle_l

    e_B_smooth, _ = extract_turbulent_var(
        snap,
        sf,
        np.sum(mag_field**2, axis=1) / (8.0 * np.pi),
        filter_length,
        weight=weight,
        filter_type=filter_type,
        iterative=iterative,
        selection=selection,
    )

    #####################
    # 2. Then compute \epsilon_{B,bulk}:

    # 2.1 compute \langle \bm B \rangle_l
    if snap_out is not None and recycle and "filt_mag_vec" in snap_out.keys():
        print("reusing...")
        filt_mag_vec = snap_out["filt_mag_vec"]
    else:
        filt_mag_vec, _ = extract_turbulent_var(
            snap,
            sf,
            mag_field,
            filter_length,
            weight=weight,
            filter_type=filter_type,
            iterative=iterative,
            selection=selection,
        )
        if snap_out is not None:
            snap_out["filt_mag_vec"] = filt_mag_vec

    # 2.2 compute \epsilon_{B,bulk} = 1/(8\pi) (\langle \bm B \rangle_l)^2
    eps_bulk = np.sum(filt_mag_vec**2, axis=1) / (8.0 * np.pi)

    #####################
    # 3. Finally compute \epsilon_{B,turb} by subtraction
    eps_turb = e_B_smooth - eps_bulk

    return e_B_smooth, eps_bulk, eps_turb


def decompose_kinetic_energy(
    snap,
    sf,
    density,
    velocity,
    filter_length,
    weight,
    filter_type,
    iterative=False,
    recycle=False,
    snap_out=None,
    selection=None,
):
    #################################################################
    # This function decomposes the filtered total kinetic energy
    # e_K according to Perrone+2026:
    # \langle e_K \rangle_l = (1/2) \langle \rho u^2 \rangle_l
    # = \epsilon_{k,bulk} + \epsilon_{k,cross} + \epsilon_{k,turb}
    # where
    # \epsilon_{k,bulk} = (1/2) \langle \rho \rangle_l ( \langle \bm u \rangle_l)^2
    # is the energy density of the mean flow,
    # \epsilon_{k,cross} =  \langle \bm u \rangle_l \cdot \mu_(\rho, \bm u)
    # with
    # \mu_(\rho, \bm u) = \langle \rho \bm u \rangle_l
    # - \langle \rho \rangle_l \times \langle \bm u \rangle_l
    # is the transport of the turbulent momentum by the mean flow
    # and
    # \epsilon_{k,turb} = \langle e_K \rangle_l - \epsilon_{k,bulk} - \epsilon_{k,cross}
    # is the energy density of the fluctuations.
    # It is a scalar quantity.
    #################################################################

    if isinstance(velocity, str):
        velocity = snap[velocity]
    if isinstance(density, str):
        density = snap[density]

    #####################
    # 1. Start by computing \langle e_K \rangle_l
    # = (1/2) \langle \rho u^2 \rangle_l

    e_K_smooth, _ = extract_turbulent_var(
        snap,
        sf,
        0.5 * density * np.sum(velocity**2, axis=1),
        filter_length,
        weight=weight,
        filter_type=filter_type,
        iterative=iterative,
        selection=selection,
    )

    #####################
    # 2. Then compute \epsilon_{k,bulk} = (1/2) \langle \rho \rangle_l ( \langle \bm u \rangle_l)^2

    # 2.1 compute \langle \bm u \rangle_l
    if snap_out is not None and recycle and "filt_vel_vec" in snap_out.keys():
        print("reusing...")
        filt_vel_vec = snap_out["filt_vel_vec"]
    else:
        filt_vel_vec, _ = extract_turbulent_var(
            snap,
            sf,
            velocity,
            filter_length,
            weight=weight,
            filter_type=filter_type,
            iterative=iterative,
            selection=selection,
        )
        if snap_out is not None:
            snap_out["filt_vel_vec"] = filt_vel_vec

    # 2.2 compute \langle \rho \rangle_l
    if (
        snap_out is not None
        and recycle
        and "rho_smooth"
        and "delta_rho" in snap_out.keys()
    ):
        print("reusing...")
        rho_smooth = snap_out["rho_smooth"]
        delta_rho = snap_out["delta_rho"]
    else:
        rho_smooth, delta_rho = extract_turbulent_var(
            snap,
            sf,
            density,
            filter_length,
            weight=weight,
            filter_type=filter_type,
            iterative=iterative,
            selection=selection,
        )
        if snap_out is not None:
            snap_out["rho_smooth"] = rho_smooth
            snap_out["delta_rho"] = delta_rho

    # 2.3 compute \epsilon_{k,bulk} = (1/2) \langle \rho \rangle_l ( \langle \bm u \rangle_l)^2
    eps_bulk = 0.5 * rho_smooth * np.sum(filt_vel_vec**2, axis=1)

    #####################
    # 3. Then compute \epsilon_{k,cross}: \langle \bm u \rangle_l \cdot \mu_(\rho, \bm u)

    # 3.1 First compute \mu_(\rho, \bm u) = \langle \rho \bm u \rangle_l
    # - \langle \rho \rangle_l \times \langle \bm u \rangle_l

    # We only need \langle \rho \bm u \rangle_l

    if snap_out is not None and recycle and "filt_mom_vec" in snap_out.keys():
        print("reusing...")
        filt_mom_vec = snap_out["filt_mom_vec"]
    else:

        filt_mom_vec, _ = extract_turbulent_var(
            snap,
            sf,
            velocity * density[:, None],
            filter_length,
            weight=weight,
            filter_type=filter_type,
            iterative=iterative,
            selection=selection,
        )
        if snap_out is not None:
            snap_out["filt_mom_vec"] = filt_mom_vec

    # 3.2 We assemble \langle \rho \rangle_l \times \langle \bm u \rangle_l
    mu2_RhoVel = filt_mom_vec - filt_vel_vec * rho_smooth[:, None]

    # 3.3 Compute dot product \langle \bm u \rangle_l \cdot \mu_(\rho, \bm u)
    eps_cross = np.sum(filt_vel_vec * mu2_RhoVel, axis=1)

    #####################
    # 4. Finally compute \epsilon_{k,turb} by subtraction
    # \epsilon_{k,turb} = \langle e_K \rangle_l - \epsilon_{k,bulk} - \epsilon_{k,cross}

    eps_turb = e_K_smooth - eps_bulk - eps_cross

    return e_K_smooth, eps_bulk, eps_cross, eps_turb
