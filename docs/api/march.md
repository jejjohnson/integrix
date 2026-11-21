# Marches

A march steps a solution in time, $\vect{y}_{n+1} = \Phi(\vect{y}_n, t_n, h)$. The single-step schemes are diffrax solvers (design decision 8), so `diffrax.diffeqsolve` runs them with its step-size controllers, `SaveAt` and adjoints. Schemes without an embedded error estimate (`ForwardEulerDfx`, `RK4Classic`, `SSP_RK104`, `AB3Solver`) run under constant steps only, and an adaptive controller raises. The schemes that carry history between steps (`AB2Solver`, `LeapfrogRAFSolver`, `SplitExplicitRKSolver`) are `AbstractStepper`s instead. `run` marches them with `jax.lax.scan`, keeping the history in the loop carry, never on the module, and it also accepts the diffrax solvers, so one entry point covers every stepper. Its `adjoint=` takes `integrix.adjoint.Checkpointed` or `Truncated` for long rollouts. Each scheme also has a pure step function for an autonomous right-hand side $f(\vect{y})$.

For a model with a stiff term, `imex_solver` returns diffrax's `KenCarp3` with a matrix-free Newton–GMRES implicit stage, `imex_stepsize_controller` the PID controller to run it under, and `stable_step` the forward-Euler CFL bound $h \le \text{CFL} / (\sum_i \lvert u_i \rvert / \Delta x_i + 2 \sum_i K_i / \Delta x_i^2)$, which traces.

Semi-Lagrangian advection reads the old field at departure points: `departure_points` integrates the trajectory back from each node with the midpoint rule, $\vect{x}_d = \vect{x} - h \vect{u}(t + h/2, (\vect{x} + \vect{x}_d)/2)$, and `semi_lagrangian_step` (or `SemiLagrangianSolver` under `diffeqsolve`) reads the field there with any functax interpolation plan the caller passes in. It has no CFL limit and no grid of its own.

The steppers moved in from finitevolX at `82090a3`, and the IMEX solver and step control from sesmax-core at `f08f59d`, with their interfaces unchanged; the source libraries' tests run against them (gate G1-port).

The theory is in the [Marches](../marches/) chapter of the mathematical reference.

::: integrix.march
