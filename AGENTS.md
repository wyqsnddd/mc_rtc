# Repository Guidelines

## Project Structure & Module Organization

Public C++ headers live in `include/`, grouped by namespace (`mc_rtc`, `mc_control`, `mc_rbdyn`, and others); matching implementations are under `src/`. Loadable components are organized in `controllers/`, `observers/`, `plugins/`, and `robots/`. Python bindings and their tests live in `binding/python/`. General C++ and controller integration tests are in `tests/`, documentation sources in `doc/`, command-line and GUI tools in `utils/`, and optional performance targets in `benchmarks/`. Treat `3rd-party/` as vendored code and avoid changing it unless the dependency itself is the subject of the change.

## Build, Test, and Development Commands

The project requires CMake 3.22+, C++17, and the dependencies listed by the top-level `CMakeLists.txt`. With dependencies installed, use an out-of-tree build:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run one test while iterating with `ctest --test-dir build -R testConfiguration --output-on-failure`. Enable optional benchmarks with `-DBUILD_BENCHMARKS=ON`; `-DENABLE_FAST_TESTS=ON` shortens controller tests. Run `pre-commit run --all-files` before submitting to apply all repository checks.

## Coding Style & Naming Conventions

Follow `.clang-format`: two-space indentation, no tabs, Allman braces, and a 120-column limit. Use `clang-format` through pre-commit rather than hand-formatting. CMake files are checked by `cmake-format`; Python uses Black and Flake8 with an 88-character target. Follow nearby code for naming: types and public component files commonly use `PascalCase`, while functions and variables use `camelCase` or established `snake_case` APIs. Preserve existing namespace-based include paths and keep public declarations paired with their source implementation.

## Testing Guidelines

C++ tests use Boost.Test and are registered with CTest. Add focused tests beside related suites in `tests/`; conventional filenames are `testFeature.cpp` or `TestControllerName.cpp`. Register new executables in the nearest `CMakeLists.txt`. There is no stated numeric coverage threshold, but changes should exercise success and failure paths and leave the complete CTest suite passing.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects, often with a category and optional scope: `feat(observer): ...`, `fix ...`, `doc: ...`, `ci: ...`, or `chore: ...`. Keep commits logically focused. Pull requests should explain motivation and behavior, identify affected modules, link relevant issues, and report exact tests run. Include screenshots or recordings for GUI/documentation visual changes and call out configuration or compatibility impacts.
