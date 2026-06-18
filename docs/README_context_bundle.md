# Codex Context Bundle for Verifier-Aware Budget-Conditioned dLLM Decoding

## Where to put these files

Unzip this bundle at the **root of the RL unmasking-policy repository**, preferably the repository based on the paper **Learning Unmasking Policies for Diffusion Language Models**.

Example:

```bash
cd /path/to/RL
unzip /path/to/codex_llada_context_bundle.zip
```

After unzipping, the repo should look like:

```text
/path/to/RL/
  AGENTS.md
  docs/
    README_context_bundle.md
    paper_notes_learning_unmasking_policies.md
    project_plan_verifier_budget_remasking.md
    reward_design_and_tests.md
    codex_implementation_prompt.md
    server_run_commands_template.md
    papers/
      README.md
```

Optional but recommended: place the original PDFs here:

```text
docs/papers/2512.09106v3.pdf
docs/papers/proposal_verifier_budget_rl.pdf
```

The Markdown files are the important part. The PDFs are just backup context.

## How to use with Codex

1. Unzip this bundle at the repository root.
2. Open Codex in that repository.
3. Tell Codex:

```text
Read AGENTS.md first, then docs/codex_implementation_prompt.md.
Follow the no-run constraint: do not run training, evaluation, pytest, import checks, model loading, or dataset downloads.
Implement the requested changes and give me server commands to run later.
```

## Important reward note

This bundle tells Codex to implement **two** reward variants:

1. `additive_proposal`
   - Comes from the project proposal.
   - Formula: `TaskScore - lambda * ComputeCost`.
   - Should be implemented and tested as a baseline/ablation.
   - Not recommended as the default final RL training reward.

2. `multiplicative_budget`
   - Recommended default.
   - Formula:
     `exact_match * exp(-beta * over_budget) * exp(-mu * nfe/max_steps) * exp(-rho * remask_count)`.
   - Wrong answers receive zero reward regardless of speed.

Both should appear in implementation and tests.
