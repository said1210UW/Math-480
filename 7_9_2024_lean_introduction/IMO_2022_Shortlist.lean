import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import LatexInLean
show_panel_widgets [latex]

set_option linter.unusedTactic false

/-! Let $(a_n)_{n \ge 1}$ be a sequence of real positive numbers with the property that $(a_{n+1})^2 + a_na_{n-2} ≤ a_n + a_{n+2}$ for all positive integers $n$. Show that $a_{2022} ≤ 1$.-/
theorem A1 (a: Nat → ℝ) (h_pos: ∀ n, a n > 0) (h_cond: ∀ n, (a (n+1))^2 + (a n)*(a (n+2)) ≤ a n + a (n+2)) : a 2022 ≤ 1 := by
  latex r"We begin by observing that $a_{n+1}^2 - 1 ≤ a_n + a_{n+2} - a_na_{n+2} - 1$, which is equivalent to $(a_{n+1})^2 - 1 ≤ (1-a_n)(a_{n+2} - 1)$."
  have h_cond' : ∀ n, (a (n+1))^2 - 1 ≤ (1 - a n) * (a (n+2) - 1) := by
    intro n
    calc (a (n+1))^2 - 1 = (a (n+1))^2 + (a n)*(a (n+2)) - (a n)*(a (n+2)) - 1 := by ring
    _ ≤ a n + a (n+2) - (a n) * (a (n+2)) - 1 := by rel [h_cond n]
    _ = (1 - a n) * (a (n+2) - 1) := by ring

  latex r"We will show that we cannot have two consecutive terms, except maybe $a_1$ and $a_2$, strictly greater than 1."
  have h_no_cons_gt_one : ¬∃ n, a (n+1) > 1 ∧ a (n+2) > 1 := by
    latex r"Suppose for contradiction that there exists a positive integer $n$ such that $a_{n+1} > 1$ and $a_{n+2} > 1$."
    by_contra h_contra
    rcases h_contra with ⟨n, h_gt_one⟩


    let A := (a (n+1))^2 - 1
    let B := a (n+2) - 1
    let C := 1 - a n
    have hA_pos : 0 < A := by simp [A, one_lt_sq_iff, lt_abs, h_gt_one]
    have hB_pos : 0 < B := by simp [B, h_gt_one]
    have hB_nn : 0 ≤ B := by linarith
    have hB_ne_zero : B ≠ 0 := by linarith
    have hC_pos : 0 < C := by
      calc 0 < A/B := by apply div_pos hA_pos hB_pos
           _ ≤ C*B/B := by simp [A, B, C, h_cond', hB_nn,div_le_div_of_nonneg_right]
           _ = C := by exact mul_div_cancel_right₀ C hB_ne_zero

    latex r"Since $a_n$ and $a_{n+2}$ are positive, we deduced that $1 - a_n < 1 + a_{n+2}$."
    have hC_lt : C < 1 + a (n+2) := by
      calc C < 1 := by simp [C, h_pos]
           _ < 1 + a (n+2) := by simp [h_pos]

    latex r"Since $(a_{n+1})^2 - 1 ≤ (1-a_n)(a_{n+2}-1)$ and $1-a_n < 1+a_{n+2}$, we deduce that $(a_{n+1})^2 - 1 ≤ (a_{n+2}+1)(a_{n+2}-1) = (a_{n+2})^2 - 1$."
    have hA_lt : A < (a (n+2))^2 - 1 := by
      calc A ≤ C*B := by simp [h_cond']
           _ < (1 + a (n+2)) * B := by simp [hB_pos, hC_lt, mul_lt_mul_of_pos_left]
           _ = (a (n+2))^2 - 1 := by ring

    latex r"On the other hand, $(a_{n+2})^2-1 ≤ (1-a_{n+3})(a_{n+1}-1) < (1+a_{n+1})(a_{n+1}-1) = (a_{n+1})^2 - 1$."
    have h_lt_A : (a (n+2))^2 - 1 < (a (n+1))^2 - 1 := by
      calc (a (n+2))^2 - 1 ≤ (1 - a (n+1))*(a (n+3) - 1) := by simp [h_cond']
            _ = (1 - a (n+3))*(a (n+1) -1) := by ring
            _ < (1 + a (n+1))*(a (n+1) - 1) := by
              have h : 1 - a (n+3) < 1 + a (n+1) := by
                calc 1 - a (n+3) < 1 := by simp [h_pos]
                  _ < 1 + a (n+1) := by simp [h_pos]
              have h' : 0 < a (n+1) - 1 := by simp [h_gt_one]
              rel [h, h']
            _ = (a (n+1))^2 - 1 := by ring

    latex r"Therefore, $(a_{n+1})^2 - 1 < (a_{n+2})^2 - 1 < (a_{n+1})^2 - 1$ which is a contradiction."
    have h_contradiction : (a (n+1))^2 - 1 < (a (n+1))^2 - 1 := by
      calc (a (n+1))^2 - 1 < (a (n+2))^2 - 1 := by exact hA_lt
           _ < (a (n+1))^2 - 1 := by exact h_lt_A
    exact (lt_self_iff_false ((a (n+1))^2 - 1)).mp h_contradiction

  latex r"Finally, suppose for contradiction that $a_{2022} > 1$."
  by_contra h_a_2022_gt_one
  simp [not_le, ← gt_iff_lt] at h_a_2022_gt_one

  latex r"This implies that $a_{2021} ≤ 1$ and $a_{2023} ≤ 1$."
  have h_a_2021_le_one : a 2021 ≤ 1 := by
    by_contra h_a_2021_gt_one
    simp [not_le, ← gt_iff_lt] at h_a_2021_gt_one
    exact h_no_cons_gt_one ⟨2020, ⟨h_a_2021_gt_one, h_a_2022_gt_one⟩⟩
  have h_a_2023_le_one : a 2023 ≤ 1 := by
    by_contra h_a_2023_gt_one
    simp [not_le, ← gt_iff_lt] at h_a_2023_gt_one
    exact h_no_cons_gt_one ⟨2021, ⟨h_a_2022_gt_one, h_a_2023_gt_one⟩⟩

  latex r"Therefore $0 < (a_{2022})^2-1 ≤ (1-a_{2021})(a_{2023}-1) ≤ 0$, a contradiction."
  have h_contradiction : (0:ℝ) < 0 := by
    calc 0 < (a 2022)^2 - 1 := by simp [h_a_2022_gt_one, lt_abs]
         _ ≤ (1 - a 2021) * (a 2023 - 1) := by simp [h_cond']
         _ ≤ 0 := by simp [mul_nonpos_iff, h_a_2021_le_one, h_a_2023_le_one]
  exact (lt_self_iff_false 0).mp h_contradiction
