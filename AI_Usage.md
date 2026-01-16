# AI Usage Disclosure

## Summary
ChatGPT was used as a support tool mainly for:
- understanding the assignment requirements and expected deliverables,
- brainstorming architecture choices and constraint-enforcement strategies,
- getting help with a few difficult implementation details (especially quantum metrics and PSD/trace constraints),
- improving clarity of documentation.

All final code was written, assembled, and tested by me, and I verified correctness by running training/evaluation and checking outputs.

---

## Where AI assistance was used 

### 1) Concept clarification 
- Understanding how to ensure a predicted density matrix is **Hermitian, PSD, and trace-1**
- Deciding an approach to enforce constraints (e.g., Cholesky factorization)

### 2) Difficult code components
ChatGPT was used for guidance on:
- implementing **Uhlmann fidelity** in a numerically stable way (batched eigendecomposition + matrix square root idea)
- implementing **trace distance** via eigenvalues of a Hermitian difference matrix
- structuring the model output so that the reconstructed matrix remains physical (PSD + normalization)


---

## Example prompts used (for record)
- “Explain how Cholesky factorization guarantees PSD and trace normalization for a density matrix.”
- “Provide a stable way to compute fidelity between two density matrices using PyTorch.”
- “How to compute trace distance for Hermitian matrices efficiently?”
- “Suggest a clean repo structure and replication guide for this project.”

---

## Verification steps I performed
- Verified **trace ≈ 1** for predicted ρ across random batches.
- Verified **Hermiticity** by checking ||ρ − ρ†|| is near zero.
- Verified **PSD** by checking eigenvalues are non-negative up to a small tolerance.
- Confirmed **fidelity ∈ [0,1]** and trace distance is non-negative.
- Ran training and evaluation scripts successfully and saved logs/metrics.
