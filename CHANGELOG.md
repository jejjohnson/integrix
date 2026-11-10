# Changelog

## [0.0.4](https://github.com/jejjohnson/integrix/compare/v0.0.3...v0.0.4) (2026-11-10)


### Features

* **rule:** accept a batch of weight functions in weighted_mean ([#62](https://github.com/jejjohnson/integrix/issues/62)) ([b3921c9](https://github.com/jejjohnson/integrix/commit/b3921c9316333f104e9aa38c8647f4aaa2ad3ceb))
* **rule:** add where= and nan_policy= to integrate and weighted_mean ([#61](https://github.com/jejjohnson/integrix/issues/61)) ([4f2e4ce](https://github.com/jejjohnson/integrix/commit/4f2e4ce4a4b3d62fd9507d4e0af4a7675fc8cb1c))

## [0.0.3](https://github.com/jejjohnson/integrix/compare/v0.0.2...v0.0.3) (2026-11-10)


### Features

* **adapt:** add adaptive Gauss–Kronrod with a static buffer and a custom JVP ([#43](https://github.com/jejjohnson/integrix/issues/43)) ([4d7773f](https://github.com/jejjohnson/integrix/commit/4d7773fdef402073092a4f5709072c5e4d39593f)), closes [#24](https://github.com/jejjohnson/integrix/issues/24)
* **adapt:** add tanh–sinh, Romberg and cumulative adaptive integrals ([#44](https://github.com/jejjohnson/integrix/issues/44)) ([1c7abf3](https://github.com/jejjohnson/integrix/commit/1c7abf35a90836a5add200b3d68c91b3af5f919d)), closes [#25](https://github.com/jejjohnson/integrix/issues/25)
* **adjoint:** add Checkpointed and checkpointed_scan, and wire adjoint= into run ([#41](https://github.com/jejjohnson/integrix/issues/41)) ([3d83215](https://github.com/jejjohnson/integrix/commit/3d83215f014a01de449a7b69ba57e4ef5e11ef5a)), closes [#23](https://github.com/jejjohnson/integrix/issues/23)
* **adjoint:** add Truncated and to_diffrax_adjoint from pipekit-jax ([#39](https://github.com/jejjohnson/integrix/issues/39)) ([7d9045e](https://github.com/jejjohnson/integrix/commit/7d9045e1370837bce6881de82d76170a919c54b1))
* **expect:** add Monte Carlo, randomised quasi-Monte Carlo and importance sampling ([#45](https://github.com/jejjohnson/integrix/issues/45)) ([ce5aa00](https://github.com/jejjohnson/integrix/commit/ce5aa001cf2f1e4f2f0dc4d31313428682a9545a)), closes [#27](https://github.com/jejjohnson/integrix/issues/27)

## [0.0.2](https://github.com/jejjohnson/integrix/compare/v0.0.1...v0.0.2) (2026-11-09)


### Features

* **march:** add AbstractStepper and run, with the AB2, leapfrog-RAF and split-explicit loops ([#35](https://github.com/jejjohnson/integrix/issues/35)) ([d68f87c](https://github.com/jejjohnson/integrix/commit/d68f87c079f6e7bcb032e97edffeac288fb4a850)), closes [#20](https://github.com/jejjohnson/integrix/issues/20)
* **march:** add the eight diffrax-solver steppers and their pure step functions ([#34](https://github.com/jejjohnson/integrix/issues/34)) ([0b0c4d0](https://github.com/jejjohnson/integrix/commit/0b0c4d0b18404b83a7218be85372789031556ff5)), closes [#19](https://github.com/jejjohnson/integrix/issues/19)
* **march:** add the IMEX solver, step-size control and the CFL bound from sesmax-core ([#36](https://github.com/jejjohnson/integrix/issues/36)) ([a985b85](https://github.com/jejjohnson/integrix/commit/a985b85c56400a15cac0986e6c73fb24a013f587)), closes [#21](https://github.com/jejjohnson/integrix/issues/21)
* **rule:** add integrate and weighted_mean, weights against samples over any axes ([#30](https://github.com/jejjohnson/integrix/issues/30)) ([e17b2f6](https://github.com/jejjohnson/integrix/commit/e17b2f6d6855068ca4776bd73e61575464148725)), closes [#16](https://github.com/jejjohnson/integrix/issues/16)
* **scan:** add cumulative, running integrals forward and reverse by associative_scan ([#31](https://github.com/jejjohnson/integrix/issues/31)) ([d356c11](https://github.com/jejjohnson/integrix/commit/d356c1117c101e1222c9bc0a1d9f8b9841997a63)), closes [#17](https://github.com/jejjohnson/integrix/issues/17)

## Changelog
