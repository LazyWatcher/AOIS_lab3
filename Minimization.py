import itertools
import re


def minimize_expression_scnf(expression, quantity):
    parentheses = [parenthese.strip() for parenthese in expression.replace('(', '').replace(')', '').split('&')]

    minimized_parentheses = set()

    parenthese_status = [(parenthese, False) for parenthese in parentheses]

    for i in range(len(parentheses)):

        parenthese1 = parentheses[i].strip()
        vars1 = parse_parenthese_scnf(parenthese1)

        for j in range(i + 1, len(parentheses)):

            parenthese2 = parentheses[j].strip()
            vars2 = parse_parenthese_scnf(parenthese2)

            minimized = try_minimize_scnf(vars1, vars2, quantity)

            if minimized:
                minimized_parentheses.add(minimized)
                parenthese_status[i] = (parenthese1, True)
                parenthese_status[j] = (parenthese2, True)

    remaining_parentheses = [parenthese for parenthese, minimized in parenthese_status if not minimized]

    for parenthese1 in remaining_parentheses[:]:

        vars1 = parse_parenthese_scnf(parenthese1)

        for parenthese2 in parentheses:

            if parenthese1 == parenthese2:
                continue

            vars2 = parse_parenthese_scnf(parenthese2)

            minimized = try_minimize_scnf(vars1, vars2, quantity)

            if minimized:

                minimized_parentheses.add(minimized)

                if minimized and len(remaining_parentheses) == 1:
                    remaining_parentheses = ""
                if minimized and len(remaining_parentheses) > 1:
                    remaining_parentheses.remove(parenthese1)

    final_result = minimized_parentheses
    final_result.update(remaining_parentheses)

    minimized_expression = ' & '.join(sorted(final_result))
    return minimized_expression


def try_minimize_scnf(vars1, vars2, quantity):
    match_count = 0
    opposite_count = 0
    common_vars = []

    for var1, val1 in vars1:
        for var2, val2 in vars2:
            if var1 == var2:
                if val1 == val2:
                    match_count += 1
                    common_vars.append((var1, val1))
                else:
                    opposite_count += 1

    if match_count == 4 and opposite_count == 1 and quantity == 5:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '|'.join(minimized_vars)
        return minimized_expr

    if match_count == 3 and opposite_count == 1 and quantity == 4:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '|'.join(minimized_vars)
        return minimized_expr

    if match_count == 2 and opposite_count == 1 and quantity == 3:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '|'.join(minimized_vars)
        return minimized_expr

    if match_count == 1 and opposite_count == 1 and quantity == 2:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '|'.join(minimized_vars)
        return minimized_expr

    return None

def minimize_expression_sdnf(expression, quantity):
    parentheses = [parenthese.strip() for parenthese in expression.replace('(', '').replace(')', '').split('|')]

    minimized_parentheses = set()

    parenthese_status = [(parenthese, False) for parenthese in parentheses]

    for i in range(len(parentheses)):

        parenthese1 = parentheses[i].strip()
        vars1 = parse_parenthese_sdnf(parenthese1)

        for j in range(i + 1, len(parentheses)):

            parenthese2 = parentheses[j].strip()
            vars2 = parse_parenthese_sdnf(parenthese2)

            minimized = try_minimize_sdnf(vars1, vars2, quantity)

            if minimized:
                minimized_parentheses.add(minimized)
                parenthese_status[i] = (parenthese1, True)
                parenthese_status[j] = (parenthese2, True)

    remaining_parentheses = [parenthese for parenthese, minimized in parenthese_status if not minimized]

    for parenthese1 in remaining_parentheses[:]:

        vars1 = parse_parenthese_sdnf(parenthese1)

        for parenthese2 in parentheses:

            if parenthese1 == parenthese2:
                continue

            vars2 = parse_parenthese_sdnf(parenthese2)

            minimized = try_minimize_sdnf(vars1, vars2, quantity)

            if minimized:

                minimized_parentheses.add(minimized)

                if minimized and len(remaining_parentheses) == 1:
                    remaining_parentheses = ""
                if minimized and len(remaining_parentheses) > 1:
                    remaining_parentheses.remove(parenthese1)

    final_result = minimized_parentheses
    final_result.update(remaining_parentheses)

    minimized_expression = ' | '.join(sorted(final_result))
    return minimized_expression


def try_minimize_sdnf(vars1, vars2, quantity):
    match_count = 0
    opposite_count = 0
    common_vars = []

    for var1, val1 in vars1:
        for var2, val2 in vars2:
            if var1 == var2:
                if val1 == val2:
                    match_count += 1
                    common_vars.append((var1, val1))
                else:
                    opposite_count += 1

    if match_count == 4 and opposite_count == 1 and quantity == 5:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '&'.join(minimized_vars)
        return minimized_expr

    if match_count == 3 and opposite_count == 1 and quantity == 4:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '&'.join(minimized_vars)
        return minimized_expr

    if match_count == 2 and opposite_count == 1 and quantity == 3:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '&'.join(minimized_vars)
        return minimized_expr

    if match_count == 1 and opposite_count == 1 and quantity == 2:

        minimized_vars = []
        for var, val in common_vars:
            minimized_vars.append(var if val else '!' + var)
        minimized_expr = '&'.join(minimized_vars)
        return minimized_expr

    return None


def parse_parenthese_sdnf(parenthese):
    vars_in_parenthese = []

    terms = parenthese.split('&')

    for term in terms:
        var = term.strip()
        if var.startswith('!'):
            vars_in_parenthese.append((var[1:], False))
        else:
            vars_in_parenthese.append((var, True))

    return vars_in_parenthese

def parse_parenthese_scnf(parenthese):
    vars_in_parenthese = []

    terms = parenthese.split('|')

    for term in terms:
        var = term.strip()
        if var.startswith('!'):
            vars_in_parenthese.append((var[1:], False))
        else:
            vars_in_parenthese.append((var, True))

    return vars_in_parenthese


def format_expression_sdnf(expression):
    parentheses = expression.split('|')

    formatted_parentheses = []
    for parenthese in parentheses:
        parenthese = parenthese.strip()
        formatted_parenthese = f"({parenthese})"
        formatted_parentheses.append(formatted_parenthese)

    formatted_expression = ' | '.join(formatted_parentheses)

    return formatted_expression

def format_expression_scnf(expression):
    parentheses = expression.split('&')

    formatted_parentheses = []
    for parenthese in parentheses:
        parenthese = parenthese.strip()
        formatted_parenthese = f"({parenthese})"
        formatted_parentheses.append(formatted_parenthese)

    formatted_expression = ' & '.join(formatted_parentheses)

    return formatted_expression



def evaluate(op, a, b=None):
    if b is None:
        if op == '!':
            return not a
        return False
    if op == '&':
        return a and b
    if op == '|':
        return a or b
    if op == '~':
        return a == b
    if op == '-':
        return not a or b
    return False

def precedence(op):
    if op in ('!', '~'):
        return 3
    if op in ('&'):
        return 2
    if op in ('|', '-'):
        return 1
    return 0

def apply_operator(operators, operands):
    op = operators.pop()
    if op == '!':
        a = operands.pop()
        operands.append(evaluate(op, a))
    else:
        b = operands.pop()
        a = operands.pop()
        operands.append(evaluate(op, a, b))

def evaluate_expression(expression, values, variables):
    operands = []
    operators = []

    i = 0
    while i < len(expression):
        if expression[i] == ' ':
            i += 1
            continue
        if expression[i] == '(':
            operators.append(expression[i])
        elif expression[i] == ')':
            while operators and operators[-1] != '(':
                apply_operator(operators, operands)
            operators.pop()
        elif expression[i] in variables:
            operands.append(values[variables.index(expression[i])])
        elif expression[i] == '!':
            operators.append(expression[i])
        else:
            while (operators and precedence(operators[-1]) >= precedence(expression[i])):
                apply_operator(operators, operands)
            operators.append(expression[i])
        i += 1

    while operators:
        apply_operator(operators, operands)

    return operands[-1]

def extract_variables(expression):
    return sorted(set(re.findall(r'[a-z]', expression)))

def print_truth_table(expression):
    variables = extract_variables(expression)
    num_vars = len(variables)
    table_results = []

    for values in itertools.product([False, True], repeat=num_vars):
        for value in values:
            print(int(value), end=" ")
        result = evaluate_expression(expression, list(values), variables)
        print(" |", int(result))
        table_results.append(result)

    return table_results, variables

def get_num_vars(truth_table_size):
    num_vars = int(truth_table_size).bit_length() - 1
    if (1 << num_vars) != truth_table_size:
        raise ValueError("Неправильный размер таблицы истинности")
    return num_vars

def build_pcnf(truth_table_results, variables):
    scnf = ""
    num_vars = get_num_vars(len(truth_table_results))

    for i, result in enumerate(truth_table_results):
        if result:
            continue

        binary = f"{i:0{num_vars}b}"

        conjunction = ""
        for j in range(num_vars):
            if binary[j] == '0':
                conjunction += f"{variables[j]}"
            else:
                conjunction += f"!{variables[j]}"
            conjunction += "|"
        conjunction = conjunction[:-1]
        if scnf:
            scnf += "&"
        scnf += f"({conjunction})"

    return scnf

def build_pdnf(truth_table_results, variables):
    sdnf = ""
    num_vars = get_num_vars(len(truth_table_results))

    for i, result in enumerate(truth_table_results):
        if not result:
            continue

        binary = f"{i:0{num_vars}b}"

        disjunction = ""
        for j in range(num_vars):
            if binary[j] == '0':
                disjunction += f"!{variables[j]}"
            else:
                disjunction += f"{variables[j]}"
            disjunction += "&"
        disjunction = disjunction[:-1]

        if sdnf:
            sdnf += "|"
        sdnf += f"({disjunction})"

    return sdnf



def parse_expression_sdnf(expression):
    terms = expression.split('|')
    parsed_terms = []
    for term in terms:
        parsed_term = []
        literals = term.strip().strip('()').split('&')
        for literal in literals:
            literal = literal.strip()
            if literal.startswith('!'):
                parsed_term.append((literal[1:], False))
            else:
                parsed_term.append((literal, True))
        parsed_terms.append(parsed_term)
    return parsed_terms


def parse_expression_scnf(expression):
    terms = expression.split('&')
    parsed_terms = []
    for term in terms:
        parsed_term = []
        literals = term.strip().strip('()').split('|')
        for literal in literals:
            literal = literal.strip()
            if literal.startswith('!'):
                parsed_term.append((literal[1:], False))
            else:
                parsed_term.append((literal, True))
        parsed_terms.append(parsed_term)
    return parsed_terms


def check_intersection(term_sdnf, term_dnf):
    term_sdnf_dict = {var: val for var, val in term_sdnf}
    for var, val in term_dnf:
        if var not in term_sdnf_dict or term_sdnf_dict[var] != val:
            return False
    return True


def build_table_sdnf(sdnf, dnf):
    parsed_sdnf = parse_expression_sdnf(sdnf)
    parsed_dnf = parse_expression_sdnf(dnf)

    table = []
    for term_dnf in parsed_dnf:
        row = []
        for term_sdnf in parsed_sdnf:
            if check_intersection(term_sdnf, term_dnf):
                row.append(1)
            else:
                row.append(0)
        table.append(row)
    return table

def build_table_scnf(scnf, cnf):
    parsed_scnf = parse_expression_scnf(scnf)
    parsed_cnf = parse_expression_scnf(cnf)

    table = []
    for term_dnf in parsed_cnf:
        row = []
        for term_sdnf in parsed_scnf:
            if check_intersection(term_sdnf, term_dnf):
                row.append(1)
            else:
                row.append(0)
        table.append(row)
    return table


def print_table(table):
    for row in table:
        print(" ".join(map(str, row)))


def find_redundant_terms(table):
    rows = len(table)
    cols = len(table[0])

    result = []

    for row in range(rows):
        essential = False
        for col in range(cols):
            if table[row][col] == 1:
                count_ones_in_col = sum(table[r][col] for r in range(rows))
                if count_ones_in_col == 1:
                    essential = True
                    break

        if essential:
            result.append('1')
        else:
            result.append('0')
            table[row] = [0] * cols

    return ''.join(result)


def remove_redundant_terms_sdnf(dnf, result):
    terms = dnf.split('|')
    filtered_terms = []

    for i, term in enumerate(terms):
        if result[i] == '1':
            filtered_terms.append(term.strip())

    return ' | '.join(filtered_terms)

def remove_redundant_terms_scnf(cnf, result):
    terms = cnf.split('&')
    filtered_terms = []

    for i, term in enumerate(terms):
        if result[i] == '1':
            filtered_terms.append(term.strip())

    return ' & '.join(filtered_terms)


def karno_map(truth_table):
    n = len(truth_table)
    if n not in [4, 8, 16, 32]:
        raise ValueError("Таблица истинности должна содержать 4, 8, 16 или 32 значений.")

    if n == 4:

        kmap = [[''] * 2 for _ in range(2)]
        indices = [
            (0, 0), (0, 1),
            (1, 0), (1, 1)
        ]
        col_labels = ['0', '1']
        row_labels = ['0', '1']

    elif n == 8:

        kmap = [[''] * 4 for _ in range(2)]
        indices = [
            (0, 0), (0, 1), (0, 3), (0, 2),
            (1, 0), (1, 1), (1, 3), (1, 2)
        ]
        col_labels = ['00', '01', '11', '10']
        row_labels = ['0', '1']

    elif n == 16:

        kmap = [[''] * 4 for _ in range(4)]
        indices = [
            (0, 0), (0, 1), (0, 3), (0, 2),
            (1, 0), (1, 1), (1, 3), (1, 2),
            (3, 0), (3, 1), (3, 3), (3, 2),
            (2, 0), (2, 1), (2, 3), (2, 2)
        ]
        col_labels = ['00', '01', '11', '10']
        row_labels = ['00', '01', '11', '10']

    elif n == 32:

        kmap = [[''] * 8 for _ in range(4)]
        indices = [
            (0, 0), (0, 1), (0, 3), (0, 2), (0, 7), (0, 6), (0, 5), (0, 4),
            (1, 0), (1, 1), (1, 3), (1, 2), (1, 7), (1, 6), (1, 5), (1, 4),
            (3, 0), (3, 1), (3, 3), (3, 2), (3, 7), (3, 6), (3, 5), (3, 4),
            (2, 0), (2, 1), (2, 3), (2, 2), (2, 7), (2, 6), (2, 5), (2, 4)
        ]
        col_labels = ['000', '001', '011', '010', '110', '111', '101', '100']
        row_labels = ['00', '01', '11', '10']
    else:
        print("Error")



    for index, value in enumerate(truth_table):
        row, col = indices[index]
        kmap[row][col] = '1' if value else '0'



    print("   " + ' '.join(col_labels))

    for i, row_label in enumerate(row_labels):
        row = ' '.join(kmap[i])
        print(f"{row_label} | {row}")








#expression = '(!a&!b&!c) | (!a&!b&c) | (!a&b&c) | (a&!b&c) | (a&b&c)'

expression = "(a & b) - (!c & d) "
truth_table_results, variables = print_truth_table(expression)

sdnf = build_pdnf(truth_table_results, variables)
print("SDNF:", sdnf)

scnf = build_pcnf(truth_table_results, variables)
print("SKNF:", scnf)


dnf5 = minimize_expression_sdnf(sdnf,5)
dnf4 = minimize_expression_sdnf(dnf5, 4)
dnf3 = minimize_expression_sdnf(dnf4, 3)
dnf2 = minimize_expression_sdnf(dnf3, 2)
dnf = format_expression_sdnf(dnf2)

print(f"Минимизированая ДНФ: {dnf}")
print("")


cnf5 = minimize_expression_scnf(scnf,5)
cnf4 = minimize_expression_scnf(cnf5, 4)
cnf3 = minimize_expression_scnf(cnf4, 3)
cnf2 = minimize_expression_scnf(cnf3, 2)
cnf = format_expression_scnf(cnf2)

print(f"Минимизированая КНФ: {cnf}")
print("")

table = build_table_sdnf(sdnf, dnf)
print_table(table)
result = find_redundant_terms(table)


filtered_dnf = remove_redundant_terms_sdnf(dnf, result)
print(filtered_dnf)

print("")

table2 = build_table_scnf(scnf, cnf)
print_table(table2)
result = find_redundant_terms(table2)


filtered_cnf = remove_redundant_terms_scnf(cnf, result)
print(filtered_cnf)

print("")

karno_map(truth_table_results)

print(filtered_dnf)

print("")




