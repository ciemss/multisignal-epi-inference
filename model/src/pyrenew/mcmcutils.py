import torch
import pandas as pd
import matplotlib.pyplot as plt

def spread_draws(posteriors, variables_names):
    """
    Get nicely shaped draws from the posterior as a Pandas DataFrame.
    """
    dfs = []
    for i_var, v in enumerate(variables_names):
        if isinstance(v, str):
            v_dims = None
        else:
            v_dims = v[1:]
            v = v[0]

        post = posteriors.get(v)
        if isinstance(post, torch.Tensor):
            post = post.numpy()

        indices = pd.MultiIndex.from_product([range(s) for s in post.shape], names=[f"{v}_dim_{i}" for i in range(len(post.shape))])
        flat_post = post.flatten()
        df = pd.Series(flat_post, index=indices, name=v).reset_index()
        dfs.append(df)

    # Combine all the data frames into one
    df_final = pd.concat(dfs, axis=1)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    return df_final

def plot_posterior(var, draws, obs_signal=None, ylab=None, xlab="Time", samples=50, figsize=(4, 5), draws_col="darkblue", obs_col="black"):
    """
    Plot the posterior distribution of a variable
    """
    if ylab is None:
        ylab = var

    fig, ax = plt.subplots(figsize=figsize)

    if obs_signal is not None:
        ax.plot(obs_signal, color=obs_col)

    sampled_draws = draws.sample(n=samples)
    for _, row in sampled_draws.iterrows():
        ax.plot(row['time'], row[var], color=draws_col, alpha=0.1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.plot([], [], color=draws_col, alpha=0.9, label="Posterior samples")

    if obs_signal is not None:
        ax.plot([], [], color=obs_col, label="Observed signal")

    ax.legend()
    return fig