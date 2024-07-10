import Lake
open Lake DSL

package «7_9_2024_lean_introduction» where
  -- add package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"

require LatexInLean from git "https://github.com/kcaze/LatexInLean.git"@"main"

@[default_target]
lean_exe «7_9_2024_lean_introduction» where
  root := `Main
