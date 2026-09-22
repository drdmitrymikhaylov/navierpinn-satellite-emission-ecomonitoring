# Navier PINN - Satellite and Tower-Sensor Emission Detection for Ecological Monitoring

## Idea

Emission monitoring fails at the same two points everywhere. Satellites see a column, not a source, and ground sensors see a source, not an area. The physics that connects them is plume transport - advection and diffusion of a scalar in a turbulent boundary layer.

This project puts that transport equation into the model. A sparse network of gas sensors mounted on telecom towers that already exist, combined with satellite column measurements, then resolves where the emission actually came from.

### Why towers

Telecom towers are already powered, already connected, already maintained, and already distributed across the terrain in a pattern designed for coverage. Emission monitoring built on them costs the sensors and nothing else.

## Planned method

Advection-diffusion with a source term, constrained by wind field and stability class. The inverse problem is recovering source location and rate from downwind concentrations. That is exactly the case where a physics-informed network earns its place - sparse, noisy sensors, and a governing equation that is known.

## Status

This is a project page for work at concept stage, and the source is not public. There is no code and no measurement campaign in this repository yet.

## Licence

Documentation, figures and result files in this repository: CC BY 4.0. Source code is held in a private repository, all rights reserved, and is available under NDA.
