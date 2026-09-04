# WidowGo1 Action and Observation Contract

The policy emits 18 actions in external hardware order: FR, FL, RR, RL (hip, thigh, calf), then waist, shoulder, elbow, forearm roll, wrist angle, and wrist rotate. The articulation has two additional state-only finger joints. Runtime code resolves articulation indices from names and never treats a simulator index as semantic identity.

The current proprioception is 76 columns: orientation 2, base angular velocity 3, joint position 20, joint velocity 20, previous action 18, foot contacts 4, command 3, end-effector goal 3, and end-effector orientation error 3. It is serialized as current proprioception 76, privileged values 24, then ten prior proprioception frames 760, totaling 860 columns. Quaternions are `xyzw` at module boundaries.
