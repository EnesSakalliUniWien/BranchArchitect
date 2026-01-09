import pytest
import copy
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick


class TestPruningDesign:
    @pytest.fixture
    def simple_chain_tree(self):
        """
        Creates A -> B(1.0) -> C(2.0).
        Total length A->C is 3.0.
        """
        encoding = {"C": 0}
        leaf_c = Node(
            name="C", length=2.0, split_indices=Partition.from_bitmask(1, encoding)
        )
        node_b = Node(
            children=[leaf_c],
            name="B",
            length=1.0,
            split_indices=Partition.from_bitmask(1, encoding),
        )
        root_a = Node(
            children=[node_b],
            name="A",
            length=0.0,
            split_indices=Partition.from_bitmask(1, encoding),
        )
        root_a.taxa_encoding = encoding
        return root_a, node_b, leaf_c

    @pytest.fixture
    def branching_tree(self):
        """
        Creates Root -> (A, B)
        """
        encoding = {"A": 0, "B": 1}
        leaf_a = Node(
            name="A", length=1.0, split_indices=Partition.from_bitmask(1, encoding)
        )
        leaf_b = Node(
            name="B", length=1.0, split_indices=Partition.from_bitmask(2, encoding)
        )
        root = Node(children=[leaf_a, leaf_b], name="Root", length=0.0)
        root.initialize_split_indices(encoding)
        return root, leaf_a, leaf_b

    def test_prune_internal_chain_preserves_length(self, simple_chain_tree):
        """
        Scenario: A -> B -> C. Remove empty B (simulated by B having only C).
        Actually, we test removing a child if it causes a chain.
        Let's construct A -> B -> (C, D). Remove D.
        Result: A -> B -> C. B has 1 child C. B should be compressed.
        Result should be A -> C with length = len(A->B) + len(B->C).
        """
        # Construction: Root -> Mid(1.0) -> (Left(2.0), Right(1.0))
        # Remove Right. Mid becomes single-child. Mid should be compressed.
        # Expect Root -> Left with length 1.0 + 2.0 = 3.0.
        encoding = {"Left": 0, "Right": 1}
        leaf_left = Node(name="Left", length=2.0)
        leaf_right = Node(name="Right", length=1.0)
        mid = Node(children=[leaf_left, leaf_right], name="Mid", length=1.0)
        root = Node(children=[mid], name="Root", length=0.0)
        root.initialize_split_indices(encoding)

        # Action: Remove 'Right' subtree
        root.remove_subtree(leaf_right, mode="stable", preserve_lengths=True)

        # Verification
        # 1. Root should have 1 child (because Mid was compressed into Root->Left link?
        #    Wait, Root->Mid->Left. Mid has 1 child (Left).
        #    Mid is compressed. Root now points to Left.
        assert len(root.children) == 1
        new_child = root.children[0]
        assert new_child.name == "Left"

        # 2. Check Length Preservation
        # Original path Root->Mid(1.0)->Left(2.0) = 3.0
        # New child length should be 3.0 (inherits Mid's length)
        # Note: Root length is usually 0 or irrelevant for finding children,
        # but Mid's length (1.0) should be added to Left's length (2.0).
        assert abs(new_child.length - 3.0) < 1e-9

    def test_remove_subtree_stable_mode_updates_splits(self, branching_tree):
        """
        Scenario: Root -> (A, B). Remove B.
        Mode: Stable.
        Expect: Root -> A.
        Root split mask should be just A's bit (1), not A|B (3).
        B's bit (2) is effectively gone from root.
        """
        root, leaf_a, leaf_b = branching_tree

        root.remove_subtree(leaf_b, mode="stable")

        assert len(root.children) == 1
        assert root.children[0] is leaf_a

        # Check root split indices
        # Should be same as leaf_a (bit 1)
        assert root.split_indices.bitmask == 1
        # Original A is 1, B is 2. 1 | 2 = 3.
        # After removal, root sum is just 1.

    def test_remove_subtree_shrink_mode_rebuilds_encoding(self, branching_tree):
        """
        Scenario: Root -> (A, B). Remove B.
        Mode: Shrink.
        Expect: A is re-encoded to index 0. Root mask is 1.
        """
        root, leaf_a, leaf_b = branching_tree

        # Before: A=0, B=1.
        assert root.taxa_encoding["B"] == 1

        root.remove_subtree(leaf_b, mode="shrink")

        # After: only A remains. A should be 0.
        assert "B" not in root.taxa_encoding
        assert root.taxa_encoding["A"] == 0
        assert root.split_indices.bitmask == 1

    def test_remove_root_error(self, branching_tree):
        root, _, _ = branching_tree
        with pytest.raises(ValueError, match="Cannot remove root node"):
            root.remove_subtree(root)

    def test_find_leaf_by_name(self, branching_tree):
        root, leaf_a, _ = branching_tree
        found = root.find_leaf_by_name("A")
        assert found is leaf_a

        assert root.find_leaf_by_name("NonExistent") is None

    def test_complex_isomorphism_with_shrink(self):
        """
        complex scenario:
        T1: ((A, B), (C, D))
        T2: ((A, B), (C, (D, E)))  <-- E is extra sibling of D

        Pruning E from T2 should result in ((A,B), (C, D)) which is isomorphic to T1.
        Must use 'shrink' mode so E is removed from encoding and D is re-indexed.
        """
        # Build T1: ((A:1, B:1):1, (C:1, D:1):1)
        enc1 = {"A": 0, "B": 1, "C": 2, "D": 3}
        t1_root = Node(name="Root1", length=0.0)
        t1_ab = Node(name="AB", length=1.0)
        t1_cd = Node(name="CD", length=1.0)
        t1_root.append_child(t1_ab)
        t1_root.append_child(t1_cd)
        t1_ab.append_child(Node(name="A", length=1.0))
        t1_ab.append_child(Node(name="B", length=1.0))
        t1_cd.append_child(Node(name="C", length=1.0))
        t1_cd.append_child(Node(name="D", length=1.0))
        t1_root.initialize_split_indices(enc1)

        # Build T2: ((A:1, B:1):1, (C:1, (D:1, E:1):1):1)
        enc2 = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        t2_root = Node(name="Root2", length=0.0)
        t2_ab = Node(name="AB", length=1.0)
        t2_c_de = Node(name="C_DE", length=1.0)  # Matches T1 CD structure
        t2_root.append_child(t2_ab)
        t2_root.append_child(t2_c_de)

        t2_ab.append_child(Node(name="A", length=1.0))
        t2_ab.append_child(Node(name="B", length=1.0))
        t2_c_de.append_child(Node(name="C", length=1.0))

        # DE Group: Initially holds D and E
        # Length 0 means it acts as a structural pivot.
        # If E is removed, D remains. D length is 1.0.
        # C_DE -> DE_Group(0) -> D(1.0). Merges to C_DE -> D(1.0). Matches T1.
        t2_de = Node(name="DE_Group", length=0.0)
        t2_c_de.append_child(t2_de)

        node_e = Node(name="E", length=1.0)
        t2_de.append_child(Node(name="D", length=1.0))
        t2_de.append_child(node_e)

        t2_root.initialize_split_indices(enc2)

        # Action: Remove E
        t2_root.remove_subtree(node_e, mode="shrink")

        # Comparison
        # 1. Encodings match (sorted leaves)
        assert len(t2_root.taxa_encoding) == 4
        assert "E" not in t2_root.taxa_encoding
        assert t1_root.taxa_encoding == t2_root.taxa_encoding

        # 2. Topological Equality
        # This asserts that the structure ((A,B),(C,D)) is identical in terms of splits
        assert t1_root == t2_root

    def test_pruning_error_conditions(self, branching_tree):
        root, leaf_a, _ = branching_tree

        # 1. Target not in children (Consistency check)
        # Create a floating node pointing to root as parent, but not in root.children
        floating = Node(name="Ghost")
        floating.parent = root
        # Do NOT add to root.children

        with pytest.raises(ValueError, match="not found in parent's children"):
            root.remove_subtree(floating)

        # 2. Target is root (redundant check but good to have)
        with pytest.raises(ValueError, match="Cannot remove root node"):
            root.remove_subtree(root)

    def test_complex_dataset_pruning(self):
        """
        Verify pruning on a real complex tree from datasets/52_bootstrap.newick.
        Test equivalence between:
        1. Removing a whole subtree (internal node).
        2. Removing all leaves of that subtree individually.
        Both should result in isomorphic trees.
        """

        # Tree 1 from 52_bootstrap (approximate snippet for robust unit testing)
        complex_newick = "(Sclerospora_graminicola:0.085,(((((Plasmopara_densa:0.013,Plasmopara_obducens:0.021):0.004,((Plasmopara_baudysii:0.008,Plasmopara_umbelliferarum:0.008):0.004,Plasmopara_pimpinellae:0.028):0.009):0.001,Plasmopara_pusilla:0.046):0.003,(Plasmopara_viticola:0.028,Plasmopara_megasperma:0.023):0.007):0.014,(((Plasmoverna_pygmaea:0.035,(Paraperonospora_leptosperma:0.018,Bremia_lactucae:0.072):0.015):0.009,(Peronospora_sanguisorbae:0.019,(((Peronospora_verna:0.004,Peronospora_arvensis:0.004):0.030,(Peronospora_aquatica:0.068,(Peronospora_sordida:0.022,Peronospora_conglomerata:0.050):0.005):0.008):0.007,((Phytophthora_infestans:0.008,(Phytophthora_arecae:0.003,Phytophthora_litchii:0.002):0.006):0.008,((((Pythium_undulatum:0.088,Pythium_monospermum:0.085):0.118,Peronospora_alta:0.093):0.002,(Pseudoperonospora_cubensis:0.001,Pseudoperonospora_humuli:0.002):0.009):0.005,((Peronospora_potentillae_sterilis:0.026,Peronospora_lamii:0.082):0.002,((((Peronospora_trivialis:0.024,Peronospora_variabilis:0.020):0.007,(Peronospora_boni_henrici:0.005,((Peronospora_aparines:0.007,Peronospora_calotheca:0.000):0.013,(Peronospora_aestivalis:0.026,((Peronospora_trifolii_alpestris:0.007,Peronospora_trifolii_repentis:0.009):0.007,Peronospora_trifoliorum:0.000):0.016):0.008):0.003):0.000):0.003,((Peronospora_pulveracea:0.002,(Peronospora_alpicola:0.001,Peronospora_hiemalis:0.015):0.002):0.006,Peronospora_rumicis:0.023):0.003):0.002,(Peronospora_aff._dentariae_MG_18_6:0.026,(Hyaloperonospora_parasitica:0.054,(Hyaloperonospora_niessleana:0.005,((Hyaloperonospora_thlaspeos_perfoliati:0.004,Hyaloperonospora_erophilae:0.005):0.012,(Peronospora_dentariae:0.016,(Hyaloperonospora_brassicae:0.000,Hyaloperonospora_lunariae:0.017):0.011):0.032):0.003):0.003):0.003):0.014):0.051):0.003):0.004):0.003):0.002):0.010):0.012):0.004,(Graminivora_graminicola:0.035,Viennotia_oplismeni:0.025):0.036):0.000):0.004,Basidiophora_entospora:0.061):0.0;"

        tree_orig = parse_newick(complex_newick)
        # Deep copies for testing
        tree_a = copy.deepcopy(tree_orig)
        tree_b = copy.deepcopy(tree_orig)

        # Target: The trio ((Peronospora_trifolii_alpestris, Peronospora_trifolii_repentis), Peronospora_trifoliorum)
        leaf_alpestris = tree_a.find_leaf_by_name("Peronospora_trifolii_alpestris")
        assert leaf_alpestris is not None

        parent_node = leaf_alpestris.parent
        assert len(parent_node.children) == 2

        target_subtree = (
            parent_node.parent
        )  # The group containing (alpestris, repentis) and trifoliorum
        assert len(target_subtree.children) == 2

        # --- Strategy A: Remove the internal node `target_subtree` ---
        # Before removal check
        aestivalis = tree_a.find_leaf_by_name("Peronospora_aestivalis")
        common_ancestor = target_subtree.parent

        # REMOVE SUBTREE
        common_ancestor.remove_subtree(target_subtree, mode="shrink")

        # --- Strategy B: Remove leaves individually from tree_b ---
        leaves_to_remove = [
            "Peronospora_trifolii_alpestris",
            "Peronospora_trifolii_repentis",
            "Peronospora_trifoliorum",
        ]

        for name in leaves_to_remove:
            l = tree_b.find_leaf_by_name(name)
            assert l is not None
            tree_b.remove_subtree(l, mode="shrink")

        # --- Verification ---

        # 1. Structure Equality
        assert tree_a == tree_b

        # 2. Encoding match
        assert tree_a.taxa_encoding == tree_b.taxa_encoding

        # 3. Check Taxa Count
        leaves_a = [l.name for l in tree_a.get_leaves()]
        leaves_b = [l.name for l in tree_b.get_leaves()]
        assert len(leaves_a) == len(leaves_b)
        assert set(leaves_a) == set(leaves_b)
        assert "Peronospora_trifolii_alpestris" not in leaves_a
