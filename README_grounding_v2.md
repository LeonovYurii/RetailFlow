# Agent1 grounded fine-tuning v2 pilot

Extract this archive into your Agent1 project root. It adds new paths only:
- data/training/grounding_v2/train_cases.jsonl (240 synthetic cases)
- data/training/grounding_v2/validation_cases.jsonl (48 synthetic cases)
- notebooks/10_prepare_grounding_v2.ipynb (local VS Code)
- notebooks/11_finetune_grounding_v2_colab.ipynb (Colab GPU)

The existing prompt builder is authoritative: notebook 10 uses it to produce train_v2.jsonl and validation_v2.jsonl, plus its exact source snapshot and data hashes. No hard-coded Windows root and no new inference prompt are introduced.

Eight response types each have 30 train / 6 validation cases. Synthetic scenario groups are disjoint and exact benchmark inputs excluded. Validation shares generation templates: this checks limited within-template generalization, not broad production readiness. Many wording patterns recur. Review the data and later add independently authored final tests. Cancel, refund, stock, invoice and address tool dictionaries are training fixtures, NOT implementations of those Agent1 tools. Adapt their schema to actual ToolResult during integration.

The dataset targets the observed failures: unsupported branch choice, wrong explanations of unknown facts, unperformed actions, redundant questions, invented procedures and verbosity. It includes correct direct-answer examples to preserve base capabilities. All dates, addresses, stock quantities and policies are fictional.

Training: original base -> fresh LoRA; no v1 continuation. Rank16/alpha32/all-linear; lr5e-5; up to3epochs; completion-only loss; batch1 x accumulation8. Both data and LR differ from v1, so this is not a single-factor causal experiment. Start small, compare epoch checkpoints, do not assume3epochs is optimal. Run0 preview is the fresh adapter before training, which initially leaves base outputs unchanged under the given configuration.

Use Colab for training, your existing local CUDA setup for evaluation. No training has been executed for you. Source syntax/data integrity are checked; runtime APIs and full GPU execution must be verified in your session. The conditional private TRL T4 workaround is version-sensitive; if a different error occurs, retain the traceback instead of repeatedly calling train().

Keep v1 and the 24-case diagnostic benchmark intact. Do not mix benchmark cases into training. New independently authored test scenarios are not included in this pilot.
