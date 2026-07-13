# Gripper Hook V2 Drawing Notes

Units: mm.

Coordinate convention follows the current MuJoCo `gripper_hook` body:

- `+Y`: hook forward direction, away from the palm/base.
- `+Z`: upper side of the hook profile.
- `+X`: lateral width direction.

## Design Intent

The hook is not a sharp fishing-hook shape. It should be a wide-mouth C/J capture hook for a lever or bar door handle. The door handle collision diameter is about 36 mm, so the hook pocket should give geometric capture with clearance instead of depending on high friction.

## Side Profile Centerline

Use these points as the centerline of the main hook profile in the Y-Z plane:

| Point | Y | Z | Purpose |
| --- | ---: | ---: | --- |
| A | 42 | 18 | shank start after palm |
| B | 132 | 18 | shank end / pocket rear upper |
| C | 166 | 42 | upper guide |
| D | 204 | 8 | rounded nose / farthest point |
| E | 190 | -46 | lower nose return |
| F | 130 | -52 | inner lip rear |
| G | 98 | -42 | entry guide rear |

Recommended centerline segments:

- A to B: straight shank.
- B to C: rising upper guide.
- C to D: rounded outer nose.
- D to E: lower return.
- E to F: inner lip.
- F to G: entry guide.

## SolidWorks Modeling Recommendation

Recommended method:

1. Keep the existing mounting base geometry unchanged.
2. Sketch the side profile above in the Y-Z plane.
3. Build the hook as a solid rounded C/J plate, or sweep a rounded section along the centerline and merge the bodies.
4. Make the total lateral width in X about 60 mm.
5. Fillet all outer and inner contact edges. No sharp inner lip.

Recommended section and fillets:

| Region | Suggested Section / Radius |
| --- | ---: |
| Shank and upper guide | 24 mm diameter equivalent, or R12 |
| Nose and lower return | 28 mm diameter equivalent, or R14 |
| Inner lip and entry guide | 22-24 mm diameter equivalent, or R11-R12 |
| All hard CAD edges | at least R2, preferably R4-R6 |

Critical clearances:

- Total forward reach from mounting origin: about 204 mm.
- Effective pocket center: around `Y=162, Z=-8`.
- Useful pocket vertical clearance: about 56-62 mm.
- Entry mouth should not be smaller than 55 mm.
- Total lateral width: 55-65 mm. Target 60 mm.

## MuJoCo Collision Approximation

The current simulation uses two side rails at:

- `X = -18 mm`
- `X = +18 mm`

with capsule radii of 11-14 mm and short bridge capsules across X. This gives a practical lateral capture envelope of about 60-64 mm.

## STL Export Notes

- Export in millimeters.
- Preserve the current gripper mounting base coordinate system if possible.
- If preserving the coordinate system is inconvenient, export the STL anyway and keep the mounting face flat and identifiable; the mesh can be aligned in MuJoCo afterward.
- Do not export an ultra-high triangle count mesh. A clean medium-resolution STL is better for loading and inspection.

