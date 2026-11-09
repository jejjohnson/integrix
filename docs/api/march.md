# Marches

A march steps a solution in time, $\vect{y}_{n+1} = \Phi(\vect{y}_n, t_n, h)$. The single-step schemes are diffrax solvers (design decision 8), so `diffrax.diffeqsolve` runs them with its step-size controllers, `SaveAt` and adjoints. Schemes without an embedded error estimate (`ForwardEulerDfx`, `RK4Classic`, `SSP_RK104`, `AB3Solver`) run under constant steps only, and an adaptive controller raises. The schemes that carry history between steps (`AB2Solver`, `LeapfrogRAFSolver`, `SplitExplicitRKSolver`) are `AbstractStepper`s instead. `run` marches them with `jax.lax.scan`, keeping the history in the loop carry, never on the module, and it also accepts the diffrax solvers, so one entry point covers every stepper. Each scheme also has a pure step function for an autonomous right-hand side $f(\vect{y})$.

The steppers moved in from finitevolX at `82090a3` with their interfaces unchanged; finitevolX's tests run against them (gate G1-port).

::: integrix.march
