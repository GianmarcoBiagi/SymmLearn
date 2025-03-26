using Documenter


push!(LOAD_PATH, "../src")


makedocs(
    sitename = "SymmLearn Documentation",
    authors = "Gianmarco Biagi",
    modules = [SymmLearn],
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
    ]
)