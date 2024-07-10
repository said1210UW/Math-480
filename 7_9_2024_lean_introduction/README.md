# Lean introduction

Lean is a mathematical formalization tool that originated from a team at Microsoft Research but now has its own organization that continues development of it. The version of Lean we are using is the latest version, Lean 4, and we are using the corresponding mathematical library that comes with it called "mathlib". Lean mostly provides the language and semantics for proof-checking whereas mathlib defines many mathematical objects that are used (such as the real numbers).

Most of the content today is in `IMO_2022_Shortlist.lean` where we formalize the proof of problem A1 in the shortlist to the 2022 IMO. Rather than understanding the Lean proof line by line, the goal is to get a sense of what a Lean proof looks like and to see that it is possible to faithfully translate a human written proof into a Lean proof.

Your assignment today will be in `Main.lean` which contains some basic tips for working in Lean. There is a single theorem to prove: that $ax + b = 0$ if and only if $x = -b/a$, assuming that $a \neq 0$. You will primarily need 4 tactics: `calc`, `rw`, `simp`, and `ring`. Some additional resources for learning more Lean on your own are: [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/) and the various Lean games on the [Lean Game Server](https://adam.math.hhu.de/).
