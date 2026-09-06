# Acceptance review — 2026-09-06

Overall: INCOMPLETE. This review supersedes earlier claims that remaining
physics differences are proven irreducible or that repository integration
constitutes final acceptance.

## Established defects repaired in this review

- Playback hard-coded an 18-action default network and could not open adaptive
  checkpoints. It now constructs both task and policy from validated checkpoint
  configuration. A real GPU run played 40 steps from the existing 24-action
  `integration-adaptive-gains/seed-1/model_2.pt`.
- Resume reconstructed default hyperparameters instead of the saved training
  configuration. The CLI restores that configuration; the runner rejects a
  conflicting configuration before changing its model. A 32-environment GPU
  run resumed the adaptive checkpoint without an adaptive CLI flag and saved
  `artifacts/acceptance-review/adaptive-resume/model_4.pt` after iterations 2/3.
- Repeated `learn()` calls restarted episodes. The runner now retains the next
  observations; an exact-parameter regression compares split and continuous
  learning through both DAgger and PPO.
- Adaptive torque supervision allowed negative proportional gain while actual
  control clamps it. Both now apply the same positive lower bound.
- The rollout comparator could pass identical infinities, empty arrays, or
  mutually missing required metadata. Those inputs now fail validation.
- Independent legacy algorithm execution now covers ordinary PPO, DAgger and
  fixed-gain torque-supervised PPO with 5 epochs and 3 mini-batches each.
  Their losses and network parameters agree at atol 1e-7 / rtol 1e-6.

## Rechecked quantitative evidence

The existing legal-finger matched traces were re-compared with the repaired
comparator and unchanged tolerance SHA-256
`ff4b828c97673367e530b31aa8cc43f134eb54cf70e263e059f14ad7b9f6b1b0`.
This was a new comparison of existing recorded arrays, not a new simulation.
Schema checks pass; overall B–D fails. Maximum joint-velocity error is
12.925866 rad/s (first failing step 0), root-state error 1.490456 (step 0),
EE-state error 1.732170 (step 0), and done flags first disagree at step 18.
Action error is exactly zero. Observation, joint-position and summed reward
fields pass their pre-existing tolerances. Inputs remain under the preserved
worktree's `artifacts/{legacy,isaaclab}/matched-limit-valid/seed-1`.

## Open acceptance requirements

1. Matched physical trajectories still fail the frozen thresholds. Evidence
   localizes the first divergence to contact-related simulation; it does not
   establish an unavoidable PhysX-version limitation. Remaining material,
   collision, articulation and solver differences require investigation.
2. Native-acceleration wrench reconstruction has not matched the old sensor.
   It supplies observations and rewards, so this remains an implementation
   acceptance issue even though pure reward formulas pass same-state tests.
3. Resume is explicitly optimizer resume with new simulation episodes. The
   checkpoint does not include task histories, delay buffers, goals, local
   generators, randomized body properties or simulator state. Exact stateful
   resume is not implemented or accepted.
4. Full learning quality has not been evaluated. Six 20-update runs establish
   execution, not convergence. The original legacy config specifies 512
   environments and 40,000 updates, but the migration plan never froze a Gate F
   training horizon. The previous 32-environment / 40,000-update claim was an
   unjustified interpretation. Gate F still needs a declared sample budget,
   complete legacy baseline and prospectively fixed acceptance intervals.
5. Adaptive-gain training has no executable legacy end-to-end oracle under the
   frozen old configuration; its repaired semantics require their own research
   validation. Collision geometry reference coverage remains incomplete.

These are open tasks, not accepted exceptions. No final all-gates PASS is issued.
