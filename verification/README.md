# Antithesis: We Won, what now?

- build reliable software by any means necessary
- **vibe quality**: cost of something goes down drastically, can do a lot more of this. (similar to Jevons paradox)
- controversial decisions: software infra vs physical infra. physical infra doesn't actually collapse, etc.

## Software Correctness

- We need to care about correctness.
- Property Based Testing has become popular.
- Same for formal verification.
- It's because of AI.
  - conventional: ai doesn't write reliable software. people feel like they need backup. THIS IS FALSE, since we've had unreliable agents for creating software FOR DECADES i.E HUMANS! industry has been okay with it.
  - more obvious story: Amdahl's law. fact about perf optimisation. Basically: if you're trying to make an overall system fast, worry about the part that is slow, not fast. for eg: testing.

- More gaps in tests when building with agents.
- example based testing
  - aka unit tests
  - single walk through your space.
  - only end up testing a specific thing, and leads to gaps.
- property based testing
  - rather than scenario, find properties/invariants about system, and through examples to find violations to these invariants.
  - seatbelt that protects you.

- antithesis skill: research
  - teaches agents how to think in terms of properties
  - write to understand topology of system,
  - and high quality test design to explore state-space and ran better tests.

- no pbt libraries in all languages
  - hegel
    - client-server arch. based on hypothesis(python)
    - shipping frontends for all languages
  - bombadil
    - style of testing to web interface.
    - how to auto explore user interfaces and find violations.
    - think it's based on playwright.
