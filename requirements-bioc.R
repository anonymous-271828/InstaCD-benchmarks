bioc_packages <- c(
	"pcalg",
    "graph"
)

requireNamespace("BiocManager", quietly = TRUE)
BiocManager::install(bioc_packages, ask=F)