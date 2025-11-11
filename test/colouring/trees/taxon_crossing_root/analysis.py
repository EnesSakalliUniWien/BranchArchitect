"""
Manual analysis of tree comparison:

Tree 1: ((A,B),(C,(D,E)))
  Root
  ├── (A,B)
  │   ├── A
  │   └── B
  └── (C,(D,E))
      ├── C
      └── (D,E)
          ├── D
          └── E

Tree 2: ((C,(A,E)),(B,D))
  Root
  ├── (C,(A,E))
  │   ├── C
  │   └── (A,E)
  │       ├── A
  │       └── E
  └── (B,D)
      ├── B
      └── D

Analysis:
- In Tree1: A is with B on the left | C,D,E on the right
- In Tree2: C,A,E on the left | B,D on the right

Comparing positions across the root:
- A: moves from left side (with B) to left side (with C,E) ✓ stays left
- B: stays left side (with A) to right side (with D) ✗ CROSSES ROOT
- C: right side to left side ✗ CROSSES ROOT
- D: right side to right side ✓ stays right
- E: right side to left side ✗ CROSSES ROOT

Wait, let me reconsider the root position...

Actually in unrooted terms or considering the minimal RF distance:
- If we keep A,B together and C,E together, then D must move
- If we keep C,D,E together, then A must move
- The algorithm found: A moves OR D moves (both minimal solutions)

The solution [A, D] means both A AND D jump simultaneously.
Let me verify if this is actually minimal...

Actually, looking at the algorithm output:
"Solutions: [[((A)), ((D))], [((B)), ((C)), ((E))]]"

This means there are TWO alternative solutions:
1. Move A and D
2. Move B, C, and E

The first solution (A+D = 2 taxa) is smaller than the second (B+C+E = 3 taxa),
so the algorithm correctly chose [A, D] as the minimal solution.

My original expected solution of just [A] was WRONG!
"""

print(__doc__)
