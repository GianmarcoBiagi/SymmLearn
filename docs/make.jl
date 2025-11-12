using Documenter
using SymmLearn



DocMeta.setdocmeta!( SymmLearn , :DocTestSetup, :(using  SymmLearn ); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers

function nice_name(file)
  file = replace(file, r"^[0-9]*-" => "")
  if haskey(page_rename, file)
    return page_rename[file]
  end
  return splitext(file)[1] |> x -> replace(x, "-" => " ") |> titlecase
end

makedocs(;
  modules = [ SymmLearn ],
  doctest = true,
  linkcheck = false, # Rely on Lint.yml/lychee for the links
  authors = "Gianmarco Biagi <biaxbiagis@gmail.com>",
  repo = "https://github.com/GianmarcoBiagi/SymmLearn",
  sitename = "SymmLearn.jl",
  format = Documenter.HTML(;
    prettyurls = true,
    canonical = "https://github.com/GianmarcoBiagi/SymmLearn",
    assets = ["assets/style.css"],
  ),
    pages    = [
        "Home"         => "index.md",
        "Installation" => "installation.md",
        "Usage"        => "usage.md",
        "Examples"     => "examples.md",
        "API"          => "api.md"
    ],
)

deploydocs(; repo = "https://github.com/GianmarcoBiagi/SymmLearn", push_preview = true)