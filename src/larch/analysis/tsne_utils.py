from sklearn.manifold import TSNE as skl_TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from cuml.manifold import TSNE as cuML_TSNE
import cupy as cp
from cuml.preprocessing import StandardScaler as cuMLScaler
from cuml.manifold import UMAP as cuML_UMAP

def compute_tsne_skl(input_vect,
                     perp=30,
                     exag=6,
                     lr='auto',
                     n_iter=2000,
                     metric='euclidean',
                     method='barnes_hut',
                     init='pca',
                     random_state=None,
                     verbose=0):        

    print("Running scikit-learn t-SNE with:",
          "perplexity =", perp,
          "early exaggeration =", exag)
        
    tsne = skl_TSNE(
        n_components=2,
        perplexity=perp,
        early_exaggeration=exag,
        learning_rate=lr,
        init=init,
        metric=metric,
        method=method,
        random_state=random_state,
        verbose=verbose
    )
    
    tsne_results = tsne.fit_transform(input_vect)

    print("Found:", tsne_results.shape[0], "points")
    return tsne_results

def run_tsne_skl(input_vect=None, zvect=None, alpha_vect=None, perp=30, exag=6,
                 lr=2000.0, n_iter=2000, ztitle="Cluster ID", save_name=None, norm=True, n_samples=None, tsne_results=None, pca=None):

    if tsne_results is None:
        tsne_results = compute_tsne_skl(input_vect,
                                        perp=perp,
                                        exag=exag,
                                        lr=lr,
                                        n_iter=n_iter,
                                        norm=norm,
                                        pca=pca)
    plot_tsne(tsne_results,
              zvect=zvect,
              alpha_vect=alpha_vect,
              ztitle=ztitle,
              ax=None,
              add_colorbar=True,
              save_name=save_name)
    return tsne_results


def _make_discrete_cmap(n_colors, cmap=None):
    """
    Construct a discrete palette where consecutive classes have strongly
    contrasting colors.
    """

    if n_colors < 1:
        raise ValueError("n_colors must be positive")

    if cmap is not None:
        source_cmap = plt.get_cmap(cmap)

        if isinstance(source_cmap, mcolors.ListedColormap):
            colors = np.asarray(source_cmap.colors)

            if n_colors <= len(colors):
                return mcolors.ListedColormap(colors[:n_colors])

        # This is less suitable for categorical colors because adjacent
        # samples from a continuous map can be similar.
        colors = source_cmap(np.linspace(0, 1, n_colors))
        return mcolors.ListedColormap(colors)

    if n_colors <= 10:
        # Different hue for every consecutive class.
        colors = list(plt.cm.tab10.colors[:n_colors])

    elif n_colors <= 20:
        tab20 = list(plt.cm.tab20.colors)

        ## Rearrange to put saturated colors first
        ordering = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
            5, 7, 9, 11, 13, 15, 17, 19, 1, 3,
        ]
        colors = [tab20[i] for i in ordering[:n_colors]]

    else:
        ## Extend the above if more clases are requested
        tab20 = list(plt.cm.tab20.colors)
        ordering = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
            5, 7, 9, 11, 13, 15, 17, 19, 1, 3,
        ]
        colors = [tab20[i] for i in ordering]

        extra_candidates = (
            list(plt.cm.Set1.colors)
            + list(plt.cm.Dark2.colors)
            + list(plt.cm.Set2.colors)
            + list(plt.cm.tab20b.colors)
            + list(plt.cm.tab20c.colors)
        )

        colors.extend(extra_candidates)
        colors = colors[:n_colors]

        # Fall back to a generated palette if the requested number exceeds
        # the combined discrete palettes.
        if len(colors) < n_colors:
            n_extra = n_colors - len(colors)
            colors.extend(
                plt.cm.hsv(
                    np.linspace(0, 1, n_extra, endpoint=False)
                )
            )

    return mcolors.ListedColormap(colors)

def plot_tsne(
    tsne_results,
    zvect,
    alpha_vect=None,
    ztitle="Cluster ID",
    ax=None,
    add_colorbar=True,
    color_mode="discrete",
    cmap=None,
    norm_type="linear",
    vmin=None,
    vmax=None,
    save_name=None,
    order_by_value=False,
):
    """
    Plot t-SNE coordinates colored by discrete, categorical, or continuous
    values.

    Parameters
    ----------
    tsne_results : array-like, shape (N, 2)
        The t-SNE coordinates.

    zvect : array-like, shape (N,)
        Values used to color points.

    color_mode : {"discrete", "categorical", "continuous"}
        discrete:
            Ordered integer-valued classes, such as particle multiplicity.
            Uses high-contrast colors. Values can be clipped using vmin/vmax.
            Endpoint labels become <=vmin and >=vmax.

        categorical:
            Unordered classes, including strings. vmin/vmax are not allowed.

        continuous:
            Continuous numerical values. Uses a continuous color scale.

    norm_type : {"linear", "log"}
        Normalization for continuous variables only.

    vmin, vmax : optional
        For discrete mode, values outside the range are assigned to the
        corresponding endpoint class.

        For continuous mode, these define the displayed color range.
    """

    tsne_results = np.asarray(tsne_results)
    zvect = np.asarray(zvect)

    if tsne_results.ndim != 2 or tsne_results.shape[1] != 2:
        raise ValueError("tsne_results must have shape (N, 2)")

    if len(zvect) < len(tsne_results):
        raise ValueError("zvect requires at least as many points as tsne_results")

    zvect = zvect[:len(tsne_results)]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ## Make the point sizes sensible
    npts = tsne_results.shape[0]
    if npts > 200_000:
        point_size = 0.2
    elif npts > 100_000:
        point_size = 0.5
    elif npts > 25_000:
        point_size = 1.0
    elif npts > 10_000:
        point_size = 2.0
    else:
        point_size = 3.0

    # Per-point alpha
    if alpha_vect is not None:
        alpha_vect = np.asarray(alpha_vect, dtype=float)

        if len(alpha_vect) != npts:
            raise ValueError("alpha_vect and tsne_results must have the same length")

        # Preserve the nonlinear alpha transformation from the original.
        alpha_vect = np.clip(alpha_vect, 0.0, 1.0) ** 3

    if color_mode == "discrete":
        # ================================================================
        # Ordered integer-valued classes, e.g. number of particles
        # ================================================================
        try:
            z_numeric = zvect.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Discrete mode requires numerical integer-valued labels"
            ) from error

        finite = np.isfinite(z_numeric)

        if not np.any(finite):
            raise ValueError("zvect contains no finite values")

        # Particle-count-style labels should be integer-valued, even if
        # stored in a floating-point array.
        if not np.allclose(
            z_numeric[finite],
            np.round(z_numeric[finite]),
        ):
            raise ValueError(
                "Discrete mode requires integer-valued labels. "
                "Use color_mode='continuous' for arbitrary floats."
            )

        z_integer = np.zeros(len(z_numeric), dtype=np.int64)
        z_integer[finite] = np.round(z_numeric[finite]).astype(np.int64)

        requested_vmin = vmin
        requested_vmax = vmax

        if vmin is None:
            discrete_min = int(np.min(z_integer[finite]))
        else:
            if not np.isclose(vmin, round(vmin)):
                raise ValueError("vmin must be integer-valued in discrete mode")
            discrete_min = int(round(vmin))

        if vmax is None:
            discrete_max = int(np.max(z_integer[finite]))
        else:
            if not np.isclose(vmax, round(vmax)):
                raise ValueError("vmax must be integer-valued in discrete mode")
            discrete_max = int(round(vmax))

        if discrete_min > discrete_max:
            raise ValueError("vmin must not be greater than vmax")

        # Clip all out-of-range values into the endpoint classes.
        z_clipped = np.clip(
            z_integer,
            discrete_min,
            discrete_max,
        )

        # Include every integer class in the displayed range, even if one
        # of those classes happens not to occur in this sample.
        levels = np.arange(
            discrete_min,
            discrete_max + 1,
            dtype=np.int64,
        )
        n_levels = len(levels)

        # Map discrete labels to consecutive color indices.
        color_indices = z_clipped - discrete_min

        discrete_cmap = _make_discrete_cmap(n_levels, cmap=cmap)

        discrete_norm = mcolors.BoundaryNorm(
            boundaries=np.arange(n_levels + 1) - 0.5,
            ncolors=n_levels,
        )

        point_colors = discrete_cmap(color_indices)

        # Invalid entries are gray.
        point_colors[~finite] = mcolors.to_rgba("lightgray")

        if alpha_vect is not None:
            point_colors[:, 3] = alpha_vect

        if order_by_value:
            # Invalid values are drawn first; higher values are drawn last.
            sorting_values = np.where(finite, z_clipped, -np.inf)
            sort_order = np.argsort(sorting_values)
        else:
            sort_order = np.arange(npts)

        ax.scatter(
            tsne_results[sort_order, 0],
            tsne_results[sort_order, 1],
            s=point_size,
            c=point_colors[sort_order],
            linewidths=0,
            rasterized=npts > 50_000,
        )

        if add_colorbar:
            scalar_mappable = plt.cm.ScalarMappable(
                norm=discrete_norm,
                cmap=discrete_cmap,
            )
            scalar_mappable.set_array([])

            cbar = fig.colorbar(
                scalar_mappable,
                ax=ax,
                ticks=np.arange(n_levels),
            )
            cbar.set_label(
                ztitle,
                rotation=270,
                labelpad=20,
            )

            tick_labels = [str(value) for value in levels]

            # Use clipped endpoint labels whenever the corresponding bound
            # was explicitly requested.
            if requested_vmin is not None:
                tick_labels[0] = rf"$\leq${discrete_min}"

            if requested_vmax is not None:
                tick_labels[-1] = rf"$\geq${discrete_max}"

            # If vmin == vmax, both clipping directions correspond to the
            # same displayed class.
            if (
                requested_vmin is not None
                and requested_vmax is not None
                and discrete_min == discrete_max
            ):
                tick_labels[0] = str(discrete_min)

            cbar.set_ticklabels(tick_labels)

    elif color_mode == "categorical":
        # ================================================================
        # Unordered categories, e.g. interaction type
        # ================================================================
        if vmin is not None or vmax is not None:
            raise ValueError("vmin and vmax are not meaningful in categorical mode")

        unique_labels, color_indices = np.unique(zvect, return_inverse=True)
        n_categories = len(unique_labels)

        discrete_cmap = _make_discrete_cmap(n_categories, cmap=cmap)

        discrete_norm = mcolors.BoundaryNorm(
            boundaries=np.arange(n_categories + 1) - 0.5,
            ncolors=n_categories,
        )

        point_colors = discrete_cmap(color_indices)

        if alpha_vect is not None:
            point_colors[:, 3] = alpha_vect

        if order_by_value:
            # This uses the category ordering returned by np.unique.
            sort_order = np.argsort(color_indices)
        else:
            sort_order = np.arange(npts)

        ax.scatter(
            tsne_results[sort_order, 0],
            tsne_results[sort_order, 1],
            s=point_size,
            c=point_colors[sort_order],
            linewidths=0,
            rasterized=npts > 50_000,
        )

        if add_colorbar:
            scalar_mappable = plt.cm.ScalarMappable(norm=discrete_norm, cmap=discrete_cmap)
            scalar_mappable.set_array([])

            cbar = fig.colorbar(
                scalar_mappable,
                ax=ax,
                ticks=np.arange(n_categories),
            )
            cbar.set_label(
                ztitle,
                rotation=270,
                labelpad=20,
            )
            cbar.set_ticklabels(
                [str(label) for label in unique_labels]
            )

    elif color_mode == "continuous":
        # ================================================================
        # Continuous numerical values, e.g. energy transfer
        # ================================================================
        try:
            z_numeric = zvect.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("Continuous mode requires numerical values") from error

        finite = np.isfinite(z_numeric)

        if norm_type == "linear":
            valid = finite

            if not np.any(valid):
                raise ValueError("zvect contains no finite values")

            color_vmin = (
                np.min(z_numeric[valid])
                if vmin is None else float(vmin)
            )
            color_vmax = (
                np.max(z_numeric[valid])
                if vmax is None else float(vmax)
            )

            if color_vmin > color_vmax:
                raise ValueError("vmin must not be greater than vmax")

            if color_vmin == color_vmax:
                color_vmax = color_vmin + max(abs(color_vmin) * 1e-6, 1e-12)

            norm = mcolors.Normalize(
                vmin=color_vmin,
                vmax=color_vmax,
                clip=True,
            )

        elif norm_type == "log":
            valid = finite & (z_numeric > 0)

            if not np.any(valid):
                raise ValueError("Log normalization requires positive values")

            color_vmin = (np.min(z_numeric[valid]) if vmin is None else float(vmin))
            color_vmax = (np.max(z_numeric[valid]) if vmax is None else float(vmax))

            if color_vmin <= 0:
                raise ValueError("vmin must be positive with log normalization")

            if color_vmax <= color_vmin:
                raise ValueError("vmax must be greater than vmin with log normalization")

            norm = mcolors.LogNorm(vmin=color_vmin, vmax=color_vmax, clip=True)

        else:
            raise ValueError("norm_type must be 'linear' or 'log'")

        continuous_cmap = plt.get_cmap(cmap or "viridis").copy()
        continuous_cmap.set_bad("lightgray")

        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=continuous_cmap)
        scalar_mappable.set_array([])

        point_colors = scalar_mappable.to_rgba(z_numeric)
        point_colors[~valid] = mcolors.to_rgba("lightgray")

        if alpha_vect is not None:
            point_colors[:, 3] = alpha_vect

        if order_by_value:
            sorting_values = np.where(valid, z_numeric, -np.inf)
            sort_order = np.argsort(sorting_values)
        else:
            sort_order = np.arange(npts)

        ax.scatter(
            tsne_results[sort_order, 0],
            tsne_results[sort_order, 1],
            s=point_size,
            c=point_colors[sort_order],
            linewidths=0,
            rasterized=npts > 50_000,
        )

        if add_colorbar:
            below_range = (vmin is not None and np.any(z_numeric[valid] < color_vmin))
            above_range = (vmax is not None and np.any(z_numeric[valid] > color_vmax))

            if below_range and above_range:
                extend = "both"
            elif below_range:
                extend = "min"
            elif above_range:
                extend = "max"
            else:
                extend = "neither"

            cbar = fig.colorbar(scalar_mappable, ax=ax, extend=extend)
            cbar.set_label(ztitle, rotation=270, labelpad=20)

    else:
        raise ValueError("color_mode must be 'discrete', 'categorical', or 'continuous'")

    ax.set_xlabel("t-SNE #0")
    ax.set_ylabel("t-SNE #1")
    ax.grid(False)

    if save_name:
        fig.savefig(
            save_name,
            dpi=200,
            bbox_inches="tight",
        )

    return ax


def plot_summary_tsne_block(tsne_results, processed, save_name=None):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    ntsne = len(tsne_results)

    nhits = np.log10(processed['nhits'][:ntsne])
    plot_tsne(tsne_results, nhits, color_mode="continuous",
              ax=axes[0][0], ztitle=r"log$_{10}$(N. hits)", norm_type="linear")
    plot_tsne(tsne_results, processed['labels']['event']['enu'][:ntsne], color_mode="continuous",
              ax=axes[0][1], ztitle=r"$E_{\nu}$ (GeV)", norm_type="linear", vmin=2, vmax=12)
    plot_tsne(tsne_results, processed['labels']['event']['q0'][:ntsne], color_mode="continuous",
              ax=axes[0][2], ztitle=r"$q_{0}$ (GeV)", norm_type="linear", vmin=0, vmax=3)

    plot_tsne(tsne_results, processed['labels']['event']['cctopology_truth'][:ntsne], color_mode="categorical",
              ax=axes[1][0], ztitle="CC topology (truth)")
    plot_tsne(tsne_results, processed['labels']['event']['cctopology_visible'][:ntsne], color_mode="categorical",
              ax=axes[1][1], ztitle="CC Topology (visible)")
    plot_tsne(tsne_results, processed['labels']['event']['edep_5mm'][:ntsne], color_mode="continuous",
              ax=axes[1][2], ztitle=r"$E_{dep}$ 5 mm", norm_type="linear", vmax=100)

    plot_tsne(tsne_results, processed['labels']['event']['edep_10mm'][:ntsne], color_mode="continuous",
              ax=axes[2][0], ztitle=r"$E_{dep}$ 10 mm", norm_type="linear", vmax=200)
    plot_tsne(tsne_results, processed['labels']['event']['edep_20mm'][:ntsne], color_mode="continuous",
              ax=axes[2][1], ztitle=r"$E_{dep}$ 20 mm", norm_type="linear", vmax=200)
    plot_tsne(tsne_results, processed['labels']['event']['edep_50mm'][:ntsne], color_mode="continuous",
              ax=axes[2][2], ztitle=r"$E_{dep}$ 50 mm", norm_type="linear", vmax=300)
    
    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    

def plot_particle_tsne_block(tsne_results, processed_labels, save_name=None):
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    ntsne = len(tsne_results)

    plot_tsne(tsne_results, processed_labels['ncharged'], order_by_value=False, vmax=5,
              ax=axes[0][0], ztitle="N. charged particles", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['nproton'][:ntsne], order_by_value=False, vmax=5,
              ax=axes[0][1], ztitle="N. protons", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['ncluster'][:ntsne], order_by_value=False, vmax=3,
              ax=axes[0][2], ztitle="N. cluster", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['npipm'][:ntsne],  order_by_value=False, vmax=3,
              ax=axes[1][0], ztitle=r"N. $\pi^{\pm}$", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['npi0'][:ntsne],  order_by_value=True, vmax=3,
              ax=axes[1][1], ztitle=r"N. $\pi^{0}$", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['nem'][:ntsne], order_by_value=False, vmax=3,
	      ax=axes[1][2], ztitle="N. EM", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['nkapm'][:ntsne], order_by_value=True, vmax=3,
              ax=axes[2][0], ztitle=r"N. $K^{\pm}$", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['nka0'][:ntsne], order_by_value=True, vmax=3,
              ax=axes[2][1], ztitle=r"N. $K^{\pm}$", norm_type="linear", color_mode="discrete")
    plot_tsne(tsne_results, processed_labels['nlambda0'][:ntsne], order_by_value=True, vmax=3,
              ax=axes[2][2], ztitle=r"N. $\Lambda^{0}$", norm_type="linear", color_mode="discrete")

    plt.tight_layout()
    if save_name: plt.savefig(save_name, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()


## Define a function for running t-SNE using the cuml version
def compute_tsne_cuml(input_vect, 
                      perp=30, 
                      exag=6, 
                      lr=None, 
                      n_iter=5000,
                      n_exag=1000,
                      verbose=True,
                      method='barnes_hut',
                      metric="euclidean"):
    
    input_vect = cp.asarray(input_vect, dtype=cp.float32)                                       
    
    n_neighbors = 3*perp
    if n_neighbors > 1024: n_neighbors = 1024

    ## LR guesstimate from Belkina et al (2019)
    if lr is None: lr = max(float(input_vect.shape[0]) / float(exag), 200)

    print("Running cuML t-SNE with:",
          "perplexity =", perp,
	  "early exaggeration =", exag,
          "lr =", lr,
          "method =", method)
    
    tsne = cuML_TSNE(n_components=2,
                     perplexity=perp,
                     n_iter=n_iter, 
                     early_exaggeration=exag,
                     exaggeration_iter=n_exag,
                     learning_rate=lr,
                     learning_rate_method=None,
                     n_neighbors=n_neighbors,
                     metric=metric,
                     method=method,
                     init='pca',
                     verbose=verbose)
    
    emb = tsne.fit_transform(input_vect)
    emb = cp.asarray(emb)

    ## Normalize
    emb = emb - emb.mean(axis=0, keepdims=True)
    scale = float(emb.std())
    if scale > 0: emb = emb * (1.0 / scale)    
    emb = cp.asnumpy(emb)

    print("Found:", emb.shape[0], "points")
    return emb

def run_umap_cuml(input_vect=None,
                  zvect=None,
                  n_neighbors=30,
                  min_distance=0.01,
                  n_epochs=800,
                  alpha_vect=0.5,
                  ztitle="Cluster ID",
                  save_name=None,
                  metric="euclidean",
                  linear_colorbar=False):

    input_vect = cp.asarray(input_vect, dtype=cp.float32)        
        
    fit = cuML_UMAP(
        negative_sample_rate=5,
        n_neighbors=n_neighbors, 
        min_dist=min_distance, 
        metric=metric, 
        #build_algo='nn_descent',
        n_epochs=n_epochs,
        init='random',
        random_state=42, 
        verbose=True
    )
    umap_results = fit.fit_transform(input_vect)    
    umap_results = cp.asnumpy(umap_results)

    x_low, x_high = np.percentile(umap_results[:,0], [0.01, 99.99])
    y_low, y_high = np.percentile(umap_results[:,1], [0.01, 99.99])
    
    unique_labels = np.unique(zvect)
    n_clusters = len(unique_labels)

    # Use a qualitative colormap with enough colors
    if linear_colorbar:
        all_colors = tuple(
            plt.cm.nipy_spectral(i / n_clusters) for i in range(n_clusters)
        )
    else:
        all_colors = (
            plt.cm.tab20.colors +
            plt.cm.tab20b.colors +
            plt.cm.tab20c.colors +
            plt.cm.tab10.colors
        )

        if n_clusters > 70:
            n_extra = n_clusters - 70
            all_colors += tuple(
                plt.cm.nipy_spectral(i / n_extra) for i in range(n_extra)
            )

    cmap = mcolors.ListedColormap(all_colors[:n_clusters])
    norm_cmap = mcolors.BoundaryNorm(
        boundaries=np.arange(n_clusters + 1),
        ncolors=n_clusters
    )

    if alpha_vect is not None:
        alpha_vect = alpha_vect**3
        rgb_colors = np.array(
            [cmap(i % n_clusters)[:3] for i in zvect]
        )
        rgb_colors = np.concatenate(
            [rgb_colors, alpha_vect[:, None]],
            axis=1
        )
    else:
        rgb_colors = [cmap(i % n_clusters) for i in zvect]

    npts = umap_results.shape[0]
    s = 0.1
    if npts <= 25000: s = 0.5
    if npts <= 10000: s = 2
    if npts > 100000: s = 0.01
    
    gr = plt.scatter(umap_results[:, 0], umap_results[:, 1], s=s, alpha=alpha_vect, c=zvect, cmap=cmap, norm=norm_cmap)
    plt.colorbar(gr, label=ztitle)
    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xlabel('UMAP #0')
    plt.ylabel('UMAP #1')
    ax = plt.gca()
    ax.grid(False)
    if save_name: plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    return
