import unittest

from main import print_truth_table, build_pcnf, minimize_expression_scnf, format_expression_scnf, build_pdnf, \
    minimize_expression_sdnf, format_expression_sdnf, build_table_sdnf, print_table, find_redundant_terms, \
    remove_redundant_terms_sdnf, karno_map, build_table_scnf, remove_redundant_terms_scnf


class MyTestCase(unittest.TestCase):
    def test_cnf(self):
        expression = "(a & b) ~ ((!c | d) | e)"
        truth_table_results, variables = print_truth_table(expression)

        scnf = build_pcnf(truth_table_results, variables)
        print("SKNF:", scnf)
        cnf5 = minimize_expression_scnf(scnf, 5)
        cnf4 = minimize_expression_scnf(cnf5, 4)
        cnf3 = minimize_expression_scnf(cnf4, 3)
        cnf2 = minimize_expression_scnf(cnf3, 2)
        cnf = format_expression_scnf(cnf2)
        temp = "(!a|!b|!c|d|e) & (a|!d) & (a|!e) & (a|c) & (b|!d) & (b|!e) & (b|c)"
        print(f"Минимизированая КНФ: {cnf}")
        print("")

        table2 = build_table_scnf(scnf, cnf)
        print_table(table2)
        result = find_redundant_terms(table2)

        filtered_cnf = remove_redundant_terms_scnf(cnf, result)
        print(filtered_cnf)

        print("")
        karno_map(truth_table_results)


        self.assertEqual(filtered_cnf, temp)  # add assertion here

    def test_dnf(self):
        expression = "(a & b) - (!c & d) "
        truth_table_results, variables = print_truth_table(expression)

        sdnf = build_pdnf(truth_table_results, variables)
        print("SDNF:", sdnf)
        dnf5 = minimize_expression_sdnf(sdnf, 5)
        dnf4 = minimize_expression_sdnf(dnf5, 4)
        dnf3 = minimize_expression_sdnf(dnf4, 3)
        dnf2 = minimize_expression_sdnf(dnf3, 2)
        dnf = format_expression_sdnf(dnf2)

        print(f"Минимизированая ДНФ: {dnf}")
        print("")
        temp = "(!a) | (!b) | (!c&d)"

        table = build_table_sdnf(sdnf, dnf)
        print_table(table)
        result = find_redundant_terms(table)

        filtered_dnf = remove_redundant_terms_sdnf(dnf, result)
        print(filtered_dnf)

        print("")
        karno_map(truth_table_results)
        self.assertEqual(filtered_dnf, temp)  # add assertion here


if __name__ == '__main__':
    unittest.main()
