# Navier PINN — Satellite and Tower-Sensor Emission Detection for Ecological Monitoring

Emission monitoring fails at the same two points everywhere: satellites see a column, not a source, and ground sensors see a source, not an area. The physics that connects them is plume transport — advection and diffusion of a scalar in a turbulent boundary layer.

This project puts that transport equation into the model, so a sparse network of gas sensors mounted on telecom towers that already exist, combined with satellite column measurements, resolves where the emission actually came from.

**Status.** Concept stage. No code and no measurement campaign in this repository yet.

## The physics

Advection–diffusion with a source term, constrained by wind field and stability class. The inverse problem — recovering source location and rate from downwind concentrations — is exactly the case where a physics-informed network earns its place: sparse, noisy sensors, and a governing equation that is known.

## Why towers

Telecom towers are already powered, already connected, already maintained, and already distributed across the terrain in a pattern designed for coverage. Emission monitoring built on them costs the sensors and nothing else.

## Source

Not public.
