# Test Cases Solution Verification

## Summary

Running the algorithm on all test cases to determine actual solutions vs. expected:

### Test Results:

| Test Case                 | Status | Actual Solution  | Expected Solution                                                    | Issue                             |
| ------------------------- | ------ | ---------------- | -------------------------------------------------------------------- | --------------------------------- |
| taxon_crossing_root.json  | ✅ PASS | [('A',), ('D',)] | [('A',), ('D',)] or [('B',), ('C',), ('E',)]                         | Fixed                             |
| symmetric_clade_swap.json | ❌ FAIL | []               | [('A',), ('B',), ('C',), ('D',)] or [('E',), ('F',), ('G',), ('H',)] | Trees are topologically IDENTICAL |

### Analysis:

#### symmetric_clade_swap.json - NOT A VALID TEST
- Tree 1: `(((A,B),(C,D)),((E,F),(G,H)))`
- Tree 2: `(((E,F),(G,H)),((A,B),(C,D)))`
- **Problem**: These trees are identical in unrooted topology - just different rooting
- **Solution**: Empty [] because trees are the same
- **Action**: Need to create actual topology difference, not just re-rooting

The jumping taxa algorithm works on UNROOTED trees. Swapping children at the root doesn't create a topological difference!

## Fix Strategy:

I need to manually create trees with ACTUAL topological differences, not just re-rootings. Let me redesign these test cases with real structural changes.
