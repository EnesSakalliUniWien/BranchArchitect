import unittest
from brancharchitect.io import parse_newick
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.leaforder.anchor_order import derive_order_for_pair


class TestSmallExampleCaiman(unittest.TestCase):
    def test_caiman_movement(self):
        # Tree 2 and Tree 3 from small_example copy 3.tree
        tree2_str = "(Emu:0.02928094160087241563,((BrownKiwi:0.01208596668760406057,(LSKiwi:0.00314962313441840636,gskiwi:0.00270734043020991189):0.00764186023120266063):0.05039757266303425409,(Ostrich:0.07554099803468590502,(((GreatRhea:0.01221036124990559045,LesserRhea:0.00994055926538009622):0.06454373927766506036,((((Gtinamou:0.08777908753262356201,Crypturellus:0.13182009559880414340):0.02674515860488506716,ECtinamou:0.10652448799039977712):0.03235830085031921044,(Caiman:0.23979163606190970204,Alligator:0.09333012840290719203):1.40024316140611704284):0.04292721260659913135,(((BrushTurkey:0.09117773684389550437,Chicken:0.12324034412627636603):0.02537968028337391629,(duck:0.11984382721393661841,magpiegoose:0.05726357836766533815):0.01428531651339515034):0.02659428953343255519,((GaviaStellata:0.05368949984387591523,LBPenguin:0.07254133283515638853):0.01047536695104974401,(turnstone:0.06913707892878373507,oystercatcher:0.07222177279361213786):0.01039358137820301647):0.02544019788595633380):0.05911893167471141824):0.00763870637300718969):0.01065886081346077187,(Dinornis:0.00838143307922432218,(EasternMoa:0.00387464654604225463,lbmoa:0.00382452830711423226):0.00567219309882202594):0.06503560661154839107):0.00730789750250917473):0.00770929693660993461):0.02679030034022620091,Cassowary:0.02328296840890944214):0.0;"
        tree3_str = "(Emu:0.02689732852636610508,((BrownKiwi:0.01024674327497765089,(LSKiwi:0.00381244065879114951,gskiwi:0.00401405770130664960):0.00929246902320924814):0.04857657492258232734,(Ostrich:0.08168109129441572525,((((Caiman:0.25813423305613053538,Alligator:0.11668792884391929188):1.45402491322467009027,(((BrushTurkey:0.08522374936585715155,Chicken:0.11843781191247268469):0.02204763669088290537,(duck:0.12266413344738807956,magpiegoose:0.05858353635430775530):0.01542833979917960542):0.03419954773131092529,((oystercatcher:0.06631336456300103432,turnstone:0.06862877567080867547):0.01095444366945961916,(GaviaStellata:0.05365069704011184160,LBPenguin:0.07407051854195442764):0.00891012952685564345):0.02808256267446461799):0.03240909284696128628):0.03480629906582367872,((Dinornis:0.00774095951326196369,(EasternMoa:0.00431146334519920578,lbmoa:0.00340906190327597974):0.00500553471875614213):0.05165656726875401950,((Gtinamou:0.09240063914243092091,Crypturellus:0.12811600083962526586):0.02882400146510060227,ECtinamou:0.12169840727400109415):0.07595819838813527847):0.01268572199285467744):0.01035144126754939178,(GreatRhea:0.01175369186265239757,LesserRhea:0.01056703799406636182):0.07173135862496562987):0.00657466367220502137):0.00573028868520841957):0.02528560389969182823,Cassowary:0.02624394259668589441):0.0;"

        # Temporary parse to get taxa
        temp_t2 = parse_newick(tree2_str)
        if isinstance(temp_t2, list):
            temp_t2 = temp_t2[0]

        temp_t3 = parse_newick(tree3_str)
        if isinstance(temp_t3, list):
            temp_t3 = temp_t3[0]

        taxa = sorted(
            list(
                set(
                    list(temp_t2.get_current_order())
                    + list(temp_t3.get_current_order())
                )
            )
        )
        encoding = {name: i for i, name in enumerate(taxa)}

        # Real parse with shared encoding
        tree2 = parse_newick(tree2_str, encoding=encoding)
        if isinstance(tree2, list):
            tree2 = tree2[0]

        tree3 = parse_newick(tree3_str, encoding=encoding)
        if isinstance(tree3, list):
            tree3 = tree3[0]

        # 1. Check Lattice Algorithm Output
        solutions, _ = LatticeSolver(tree2, tree3).solve_iteratively()

        print("\n--- Lattice Solutions ---")
        found_caiman_alligator = False
        for sol in solutions:
            print(f"Solution: {sol}")
            if "Caiman" in sol.taxa or "Alligator" in sol.taxa:
                print("!!! Caiman or Alligator found in solution !!!")
                found_caiman_alligator = True

        if not found_caiman_alligator:
            print("Caiman and Alligator are NOT in the jumping taxa solution.")
        else:
            print("Caiman and Alligator ARE in the jumping taxa solution.")

        # 2. Check Anchor Ordering
        print("\n--- Anchor Ordering ---")
        derive_order_for_pair(tree2, tree3, circular=True)

        print(f"Tree 2 Order: {list(tree2.get_current_order())}")
        print(f"Tree 3 Order: {list(tree3.get_current_order())}")

        # 3. Check Interpolation Pipeline
        print("\n--- Interpolation Pipeline ---")
        from brancharchitect.tree_interpolation.pair_interpolation import (
            process_tree_pair_interpolation,
        )

        # Use fresh copies for interpolation
        t2_interp = tree2.deep_copy()
        t3_interp = tree3.deep_copy()

        # NOTE: We are NOT calling derive_order_for_pair here to see what the raw pipeline does

        result = process_tree_pair_interpolation(t2_interp, t3_interp)

        print(f"Interpolation generated {len(result.trees)} trees.")
        if result.trees:
            first_tree = result.trees[0]
            last_tree = result.trees[-1]
            print(
                f"First Interpolated Tree Order: {list(first_tree.get_current_order())}"
            )
            print(
                f"Last Interpolated Tree Order:  {list(last_tree.get_current_order())}"
            )

            # Check if last tree matches Tree 3's order (which might be different if derive_order_for_pair wasn't called)
            print(
                f"Matches Tree 3 Raw Order? {list(last_tree.get_current_order()) == list(tree3.get_current_order())}"
            )

            # Check if it matches the Anchor Order (which we printed above)
            # We need to capture the anchor order from the previous step
            # But wait, derive_order_for_pair modifies the trees in-place!
            # So 'tree2' and 'tree3' ABOVE are already modified by derive_order_for_pair.

            print(
                f"Matches Anchor-Ordered Tree 3? {list(last_tree.get_current_order()) == list(tree3.get_current_order())}"
            )


if __name__ == "__main__":
    unittest.main()
