push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using CairoMakie
using DataFrames
using CSV
using Statistics

results_dir = joinpath(@__DIR__, "..", "results")
plots_dir = @__DIR__
mkpath(plots_dir)

function save_fig(fig, name)
    CairoMakie.save(joinpath(plots_dir, name * ".pdf"), fig)
    CairoMakie.save(joinpath(plots_dir, name * ".png"), fig; px_per_unit=300/96)
    println("Saved $name")
end

# ── Figure 1 & 2: Erdős-Rényi ──────────────────────────────────────────────
er = CSV.read(joinpath(results_dir, "erdos_renyi.csv"), DataFrame)

function grouped_mean_std(df, group_col, val_col)
    gdf = groupby(df, group_col)
    xs = Float64[]
    mus = Float64[]
    sigs = Float64[]
    for g in gdf
        push!(xs, first(g[!, group_col]))
        push!(mus, mean(g[!, val_col]))
        push!(sigs, std(g[!, val_col]))
    end
    return xs, mus, sigs
end

# Figure 1: ER approximation ratio
fig1 = Figure(resolution=(900, 300))
titles = ["SA", "SQA", "Exact Quantum"]
cols = [:sa_approx_ratio, :sqa_approx_ratio, :eq_approx_ratio]
for (k, (title, col)) in enumerate(zip(titles, cols))
    ax = Axis(fig1[1, k], title=title, xlabel="p", ylabel="Approx. ratio",
              limits=(nothing, (0, 1.05)))
    xs, mus, sigs = grouped_mean_std(er, :p, col)
    barplot!(ax, xs, mus, color=:steelblue, width=0.06)
    errorbars!(ax, xs, mus, sigs, color=:black, linewidth=1.5)
end
save_fig(fig1, "fig1_er_approx_ratio")

# Figure 2: ER coincidence rate
fig2 = Figure(resolution=(900, 300))
cols2 = [:sa_coincidence_rate, :sqa_coincidence_rate, :eq_coincidence]
for (k, (title, col)) in enumerate(zip(titles, cols2))
    ax = Axis(fig2[1, k], title=title, xlabel="p", ylabel="Coincidence rate",
              limits=(nothing, (0, 1.05)))
    xs, mus, sigs = grouped_mean_std(er, :p, col)
    barplot!(ax, xs, mus, color=:coral, width=0.06)
    errorbars!(ax, xs, mus, sigs, color=:black, linewidth=1.5)
end
save_fig(fig2, "fig2_er_coincidence")

# ── Figure 3: Watts-Strogatz ────────────────────────────────────────────────
ws = CSV.read(joinpath(results_dir, "watts_strogatz.csv"), DataFrame)

fig3 = Figure(resolution=(900, 300))
for (k, (title, col)) in enumerate(zip(titles, cols))
    ax = Axis(fig3[1, k], title=title, xlabel="β", ylabel="Approx. ratio",
              limits=(nothing, (0, 1.05)))
    xs, mus, sigs = grouped_mean_std(ws, :beta, col)
    barplot!(ax, xs, mus, color=:mediumseagreen, width=0.04)
    errorbars!(ax, xs, mus, sigs, color=:black, linewidth=1.5)
end
save_fig(fig3, "fig3_ws_approx_ratio")

# ── Figure 4: Welfare gap ────────────────────────────────────────────────────
fig4 = Figure(resolution=(900, 400))

# ER
ax41 = Axis(fig4[1, 1], title="ER: Welfare gap", xlabel="p", ylabel="W* - W̄")
er_p_vals = sort(unique(er.p))
for (sym, col, lab) in [(:circle, :sa_mean_welfare, "SA"), (:diamond, :sqa_mean_welfare, "SQA")]
    xs, mus, sigs = grouped_mean_std(er, :p, col)
    opt_xs, opt_mus, _ = grouped_mean_std(er, :p, :optimal_welfare)
    gaps = opt_mus .- mus
    scatterlines!(ax41, xs, gaps, marker=sym, label=lab)
end
axislegend(ax41)

# WS
ax42 = Axis(fig4[1, 2], title="WS: Welfare gap", xlabel="β", ylabel="W* - W̄")
for (sym, col, lab) in [(:circle, :sa_mean_welfare, "SA"), (:diamond, :sqa_mean_welfare, "SQA")]
    xs, mus, sigs = grouped_mean_std(ws, :beta, col)
    opt_xs, opt_mus, _ = grouped_mean_std(ws, :beta, :optimal_welfare)
    gaps = opt_mus .- mus
    scatterlines!(ax42, xs, gaps, marker=sym, label=lab)
end
axislegend(ax42)
save_fig(fig4, "fig4_welfare_gap")

# ── Figure 5: Spectral gap vs welfare gap ───────────────────────────────────
fig5 = Figure(resolution=(700, 350))
ax5 = Axis(fig5[1, 1], xlabel="Spectral gap", ylabel="Welfare gap", title="Spectral gap vs Welfare gap")

combined = vcat(
    transform(er, :p => (x -> fill("ER", length(x))) => :source),
    transform(ws, :beta => (x -> fill("WS", length(x))) => :source),
)

for (col, label, color) in [(:sa_mean_welfare, "SA", :steelblue), (:sqa_mean_welfare, "SQA", :coral)]
    gaps = combined.optimal_welfare .- combined[!, col]
    scatter!(ax5, combined.spectral_gap, gaps, label=label, color=(color, 0.5), markersize=6)
end
axislegend(ax5)
save_fig(fig5, "fig5_spectral_gap_welfare_gap")

# ── Figure 6: Logical entropy ────────────────────────────────────────────────
fig6 = Figure(resolution=(900, 400))
ent_cols = [:sa_mean_logical_entropy, :sqa_mean_logical_entropy, :eq_logical_entropy]
ent_labels = ["SA", "SQA", "Exact Quantum"]

ax61 = Axis(fig6[1, 1], title="ER: Logical entropy", xlabel="p", ylabel="Mean logical entropy")
for (col, lab) in zip(ent_cols, ent_labels)
    xs, mus, _ = grouped_mean_std(er, :p, col)
    scatterlines!(ax61, xs, mus, label=lab)
end
axislegend(ax61)

ax62 = Axis(fig6[1, 2], title="WS: Logical entropy", xlabel="β", ylabel="Mean logical entropy")
for (col, lab) in zip(ent_cols, ent_labels)
    xs, mus, _ = grouped_mean_std(ws, :beta, col)
    scatterlines!(ax62, xs, mus, label=lab)
end
axislegend(ax62)
save_fig(fig6, "fig6_logical_entropy")

# ── Figure 7: Extremes ───────────────────────────────────────────────────────
ext = CSV.read(joinpath(results_dir, "extremes.csv"), DataFrame)

metrics_ext = [
    (:sa_mean_welfare, :sqa_mean_welfare, :eq_welfare, "Welfare"),
    (:sa_coincidence_rate, :sqa_coincidence_rate, :eq_coincidence, "Coincidence"),
    (:sa_mean_cooperation, :sqa_mean_cooperation, :eq_cooperation, "Cooperation rate"),
    (:sa_mean_logical_entropy, :sqa_mean_logical_entropy, :eq_logical_entropy, "Logical entropy"),
]

fig7 = Figure(resolution=(1000, 500))
graph_names = ext.graph
colors = [:steelblue, :coral, :mediumseagreen]

for (col_idx, (sa_col, sqa_col, eq_col, title)) in enumerate(metrics_ext)
    ax = Axis(fig7[1, col_idx], title=title, xticks=(1:length(graph_names), graph_names),
              xticklabelrotation=pi/6)
    for (gidx, row) in enumerate(eachrow(ext))
        vals = [row[sa_col], row[sqa_col], row[eq_col]]
        xs = gidx .+ [-0.25, 0.0, 0.25]
        barplot!(ax, xs, vals, color=colors, width=0.2)
    end
end

# Legend
elem_sa  = PolyElement(color=colors[1])
elem_sqa = PolyElement(color=colors[2])
elem_eq  = PolyElement(color=colors[3])
Legend(fig7[2, :], [elem_sa, elem_sqa, elem_eq], ["SA", "SQA", "Exact Quantum"],
       orientation=:horizontal)
save_fig(fig7, "fig7_extremes")

println("All figures saved.")
