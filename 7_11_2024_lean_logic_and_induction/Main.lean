import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset

/- Supply proofs for 2 out of the 3 assignments.
   Do all 3 for 5 points of extra credit.

   All assignments can be proven through induction and appropriate use of library functions and logic operations.
-/

-- Assignment 1: Show that 2^n % 7 = 1, 2, or 4 for all n.
theorem assignment1 : ∀ n:ℕ, 2^n % 7 = 1 ∨ 2^n % 7 = 2 ∨ 2^n % 7 = 4 := by
  intro n
  induction n with
  | zero =>
    simp[Nat.pow_zero]
  | succ n ih =>
    have h : 2^(n + 1) % 7 = (2 * (2^n % 7)) % 7 := by
      rw [Nat.pow_succ, Nat.mod_mul]
    cases (2^n % 7) with
    | 1 =>
      have: 2 * 1 % 7 = 2 := by simp
      exact Or.inr this
    | 2 =>
      have : 2 * 2 % 7 = 4 := by simp
      exact Or.inr this
    | 4 =>
      have: 2 * 4 % 7 = 1 := by simp [Nat.mod_eq_of_lt, Nat.le_of_lt_succ]
      exact Or.inl this
    | _ =>
      rfl

-- Assignment 2: Show that (1-x)*(1+x+x^2+...+x^{n-1}) = (1-x^n)
theorem assignment2
    (x:ℝ)
    : ∀ n:ℕ, (1-x)*(∑ i ∈ Finset.range n, x^i) = 1-x^n := by
    induction n with
    | zero =>
       simp[Finset.sum_empty]
    | succ n ih =>
       have h := ih
       simp[Finset.sum_range_succ] at h
       calc
        (1 - x^(n + 1)) / (1 - x) = (1 - x^n + x^n - x^(n + 1)) / (1 - x):= by ring
        _= (1 - x^(n + 1)) / (1 - x) : by ring
       rw[sum_formula]
       simp[mul_comm]

-- Assignment 3: Show that if a_0 = 0, a_{n+1} = 2*a_n+1 then a_n = 2^n-1.
theorem assignment3
    (a: ℕ → ℝ) (h_zero: a 0 = 0) (h_rec: ∀ n:ℕ, a (n+1) = 2 * (a n) + 1)
    : ∀ n:ℕ, a n = 2^n - 1 := by
    intro n
    induction n with
    | zero =>
        rw[h_zero]
        simp[pow_zero]
    |succ n ih =>
        have : a (n + 1) = 2 * (2^n - 1) + 1 := by
            rw[ih]
