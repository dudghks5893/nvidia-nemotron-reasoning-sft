"""Core parser, solver, message, and split utilities extracted from v16_85_custom.ipynb.

This module intentionally keeps the notebook solver logic close to the original Colab version,
but removes notebook-only evaluation cells so it can be imported from scripts.
"""
from __future__ import annotations
import ast
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ===== Extracted from notebook cell 14: pattern detection =====
import re

def detect_symbolic_subtype(prompt):
    """
    symbolic_transformation 내부 하위 유형을 분류하는 함수

    subtype:
        - numeric_symbol: 숫자와 연산기호가 섞인 문제
        - pure_symbol: 특수문자 중심 문제
    """
    if re.search('\\d', str(prompt)):
        return 'numeric_symbol'
    return 'pure_symbol'

def detect_pattern(prompt):
    """
    prompt 문자열을 분석하여 문제 패턴을 분류하는 함수
    """
    p = prompt.lower()
    if 'bit manipulation' in p or '8-bit binary' in p:
        return 'bit_manipulation'
    if 'unit conversion' in p or 'convert the following measurement' in p:
        return 'unit_conversion'
    if 'numeral system' in p or 'roman' in p:
        return 'roman_numeral'
    if 'decrypt the following text' in p or 'encryption rules' in p:
        return 'cipher'
    if 'gravitational constant' in p or 'falling distance' in p or 'd = 0.5*g*t^2' in p:
        return 'formula_based'
    if 'transformation rules' in p or 'determine the result for' in p or re.search('\\d', prompt):
        return detect_symbolic_subtype(prompt)
    return 'unknown'

import re


# ===== Extracted from notebook cell 20 =====
import re

import pandas as pd

def parse_unit_conversion(prompt):
    """
    unit_conversion 문제에서 예시와 query 값을 추출하는 함수

    예:
    45.95 m becomes 80.36
    Now, convert the following measurement: 43.36 m
    """
    examples = re.findall('([0-9]+(?:\\.[0-9]+)?)\\s*m\\s*becomes\\s*([0-9]+(?:\\.[0-9]+)?)', prompt)
    query_match = re.search('convert the following measurement:\\s*([0-9]+(?:\\.[0-9]+)?)\\s*m', prompt, re.IGNORECASE)
    query = query_match.group(1) if query_match else None
    return (examples, query)

def parse_roman_numeral(prompt):
    """
    roman_numeral 문제에서 숫자 변환 예시와 query 숫자를 추출하는 함수

    예:
    11 -> XI
    Now, write the number 38 in the Wonderland numeral system.
    """
    examples = re.findall('\\b(\\d+)\\s*->\\s*([IVXLCDM]+)\\b', prompt)
    query_match = re.search('write the number\\s+(\\d+)', prompt, re.IGNORECASE)
    query = query_match.group(1) if query_match else None
    return (examples, query)

def parse_cipher(prompt):
    """
    cipher 문제에서 암호문 -> 평문 예시와 decrypt 대상 문장을 추출하는 함수

    예:
    abc def -> cat dog
    Now, decrypt the following text: xyz
    """
    examples = re.findall('^(.+?)\\s*->\\s*(.+)$', prompt, flags=re.MULTILINE)
    query_match = re.search('decrypt the following text:\\s*(.+)$', prompt, re.IGNORECASE)
    query = query_match.group(1).strip() if query_match else None
    return (examples, query)

def parse_bit_manipulation(prompt):
    """
    bit_manipulation 문제에서 8-bit input -> output 예시와 query binary를 추출하는 함수

    예:
    01010001 -> 11011101
    Now, determine the output for: 00110100
    """
    examples = re.findall('\\b([01]{8})\\s*->\\s*([01]{8})\\b', prompt)
    query_match = re.search('determine the output for:\\s*([01]{8})', prompt, re.IGNORECASE)
    query = query_match.group(1) if query_match else None
    return (examples, query)

def parse_formula_based(prompt):
    """
    formula_based 문제에서 t, distance 예시와 query t 값을 추출하는 함수

    예:
    For t = 4.43s, distance = 127.88 m
    Now, determine the falling distance for t = 1.48s
    """
    examples = re.findall('t\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)s,\\s*distance\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)\\s*m', prompt, flags=re.IGNORECASE)
    query_match = re.search('falling distance for t\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)s', prompt, re.IGNORECASE)
    query = query_match.group(1) if query_match else None
    return (examples, query)

def parse_symbolic_transformation(prompt):
    """
    symbolic_transformation 문제에서 식 변환 예시와 query 식을 추출하는 함수

    예:
    66$29 = 62$
    Now, determine the result for: 77/68
    """
    lines = prompt.splitlines()
    examples = []
    for line in lines:
        line = line.strip()
        if not line or 'Alice' in line or 'Below' in line or ('Now,' in line):
            continue
        if '=' in line:
            left, right = line.split('=', 1)
            examples.append((left.strip(), right.strip()))
    query_match = re.search('determine the result for:\\s*(.+)$', prompt, re.IGNORECASE)
    query = query_match.group(1).strip() if query_match else None
    return (examples, query)


# ===== Extracted from notebook cell 21 =====
def parse_prompt_by_pattern(row):
    """
    pattern 값에 따라 적절한 parser를 선택하는 함수
    """
    pattern = row['pattern']
    prompt = row['prompt']
    if pattern == 'unit_conversion':
        return parse_unit_conversion(prompt)
    elif pattern == 'roman_numeral':
        return parse_roman_numeral(prompt)
    elif pattern == 'cipher':
        return parse_cipher(prompt)
    elif pattern == 'bit_manipulation':
        return parse_bit_manipulation(prompt)
    elif pattern == 'formula_based':
        return parse_formula_based(prompt)
    elif pattern == 'numeric_symbol':
        return parse_symbolic_transformation(prompt)
    elif pattern == 'pure_symbol':
        return parse_symbolic_transformation(prompt)
    else:
        return ([], None)


# ===== Extracted from notebook cell 25 =====
BOXED_OUTPUT_PREFIX = 'The final answer must be written inside \\boxed{}.'

def prepend_boxed_prefix(solution: str) -> str:
    """
    모든 pattern solver의 solution 시작 부분에 boxed 관련 안내 문장을 추가한다.

    처리:
        - solution 시작 전에:
          I will put my final answer inside \\boxed{}.

    중복 방지:
        - 이미 같은 prefix가 있으면 다시 추가하지 않음
    """
    if solution is None:
        return BOXED_OUTPUT_PREFIX
    solution = str(solution).strip()
    if BOXED_OUTPUT_PREFIX in solution:
        return solution
    return BOXED_OUTPUT_PREFIX + '\n\n' + solution

def make_solver_result(solved=False, answer=None, solution=None, rule_name=None, solver_name=None):
    """
    모든 solver 결과를 동일한 형태로 반환한다.

    solved      : rule-based로 풀었는지 여부
    answer      : solver가 계산한 정답
    solution    : 사용한 규칙/공식/풀이 과정
    rule_name   : 사용한 규칙/공식/풀이 과정의 이름
    solver_name : 어떤 solver가 풀었는지 기록
    """
    return {'solved': bool(solved), 'answer': str(answer).strip() if answer is not None else None, 'solution': solution, 'rule_name': rule_name, 'solver_name': solver_name}


# ===== Extracted from notebook cell 27 =====
import ast

ROMAN_MAP = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

def safe_parse_roman_examples_if_string(examples):
    """
    examples가 문자열이면 ast.literal_eval로 복원한다.
    이미 list/tuple이면 그대로 반환한다.
    """
    if isinstance(examples, str):
        try:
            return ast.literal_eval(examples)
        except Exception:
            return examples
    return examples

def int_to_roman(num):
    """
    정수를 standard Roman numeral 문자열로 변환한다.

    지원 범위:
        - Standard Roman numeral only

    반환:
        roman_result: 변환된 Roman numeral 문자열
        steps: 계산 로그용 step list
        original_num: 원래 입력 숫자
    """
    num = int(num)
    original_num = num
    result = ''
    steps = []
    for value, symbol in ROMAN_MAP:
        while num >= value:
            before = num
            after = num - value
            result += symbol
            num = after
            steps.append({'before': before, 'value': value, 'symbol': symbol, 'remaining': after, 'roman_part': symbol})
    return (result, steps, original_num)

def format_roman_value_table_log():
    """
    Standard Roman value table만 출력한다.
    Extended Roman numeral은 출력하지 않는다.
    """
    return 'Reference value table\n  1000=M, 900=CM, 500=D, 400=CD\n  100=C, 90=XC, 50=L, 40=XL\n  10=X, 9=IX, 5=V, 4=IV, 1=I'

def format_roman_compact_conversion_log(input_num, roman_result, expected_roman=None):
    """
    examples/query 공통 Roman 변환 로그.

    출력 형태:
        Converting 55 -> LV:
        55 < 1000 skip
        55 < 900 skip
        55 < 500 skip
        55 < 400 skip
        55 < 100 skip
        55 < 90 skip
        55 >= 50 -> L, remainder 5
        5 < 40 skip
        5 < 10 skip
        5 < 9 skip
        5 >= 5 -> V, remainder 0
        L  V -> LV
    """
    input_num = int(input_num)
    remaining = input_num
    parts = []
    lines = []
    if expected_roman is None:
        lines.append(f'Converting {input_num}:')
    else:
        lines.append(f'Converting {input_num} -> {expected_roman}:')
    for value, symbol in ROMAN_MAP:
        if remaining >= value:
            while remaining >= value:
                before = remaining
                after = remaining - value
                parts.append(symbol)
                remaining = after
                lines.append(f'{before} >= {value} -> {symbol}, remainder {remaining}')
        else:
            lines.append(f'{remaining} < {value} skip')
        if remaining == 0:
            break
    if len(parts) == 0:
        lines.append(f'{roman_result}')
    else:
        lines.append(f"{'  '.join(parts)} -> {roman_result}")
    return '\n'.join(lines)

def build_roman_examples_conversion_log(examples):
    """
    examples에도 query와 같은 compact value-check 템플릿을 적용한다.
    """
    examples = safe_parse_roman_examples_if_string(examples)
    if examples is None or len(examples) == 0:
        return ''
    blocks = []
    for ex in examples:
        if isinstance(ex, dict):
            input_num = ex.get('input')
            expected_roman = ex.get('output')
        else:
            input_num, expected_roman = ex
        predicted_roman, steps, original_num = int_to_roman(input_num)
        block = format_roman_compact_conversion_log(input_num=original_num, roman_result=predicted_roman, expected_roman=expected_roman)
        blocks.append(block)
    return '\n\n'.join(blocks)

def build_roman_query_conversion_log(query_num, answer):
    """
    query 변환 계산 로그를 compact value-check 형태로 만든다.
    """
    return format_roman_compact_conversion_log(input_num=query_num, roman_result=answer, expected_roman=None)

def solve_roman_numeral(examples, query):
    """
    roman_numeral 문제를 해결하는 함수.

    처리 방식:
        1. Standard Roman numeral conversion rule 사용
        2. Extended Roman numeral은 사용하지 않음
        3. examples와 query 모두 compact value-check 템플릿으로 출력
        4. 최종 출력은 Roman numeral 문자열만 사용
    """
    rule_name = 'int_to_roman_standard_compact_value_check'
    solver_name = 'roman_numeral_solver_standard_compact_value_check'
    if query is None:
        return make_solver_result(solved=False, answer=None, rule_name=rule_name, solver_name=solver_name, solution='This is Arabic to Roman numeral conversion.\nstatus=unsolved\nreason=query_missing')
    try:
        answer, steps, original_num = int_to_roman(query)
        examples_log = build_roman_examples_conversion_log(examples=examples)
        query_log = build_roman_query_conversion_log(query_num=original_num, answer=answer)
        solution_parts = []
        solution_parts.append('This is Arabic to Roman numeral conversion.')
        solution_parts.append('')
        solution_parts.append(format_roman_value_table_log())
        if examples_log.strip() != '':
            solution_parts.append('')
            solution_parts.append(examples_log)
        solution_parts.append('')
        solution_parts.append(query_log)
        solution_parts.append('')
        solution_parts.append('I will now return the answer in \\boxed{}')
        solution_parts.append(f'The answer in \\boxed{{–}} is \\boxed{{{answer}}}')
        solution = '\n'.join(solution_parts)
        return make_solver_result(solved=True, answer=answer, solution=solution, rule_name=rule_name, solver_name=solver_name)
    except Exception as e:
        return make_solver_result(solved=False, answer=None, rule_name=rule_name, solver_name=solver_name, solution=f'This is Arabic to Roman numeral conversion.\nstatus=solver_exception\nerror={type(e).__name__}: {e}')


# ===== Extracted from notebook cell 32 =====
import math

from decimal import Decimal, ROUND_HALF_UP

def decimal_to_scaled_int(value, decimals=2):
    """
    소수 값을 정수 비율 계산용으로 변환한다.

    예:
        18.46 -> 1846
        17.56 -> 1756
    """
    value = Decimal(str(value))
    scale = Decimal(10) ** decimals
    return int((value * scale).to_integral_value(rounding=ROUND_HALF_UP))

def truncate_float(value, decimals=3):
    """
    고득점자 unit log 스타일에 맞춰 소수점 decimals 자리에서 버림 처리한다.

    예:
        37.79396 -> 37.793
        53.44050 -> 53.440
        0.920279 -> 0.920
    """
    value = float(value)
    scale = 10 ** decimals
    if value >= 0:
        return math.floor(value * scale) / scale
    return math.ceil(value * scale) / scale

def format_truncated_decimal(value, decimals=3):
    """
    소수점 decimals 자리까지 버림 후 고정 자리수 문자열로 출력한다.
    """
    return f'{truncate_float(value, decimals=decimals):.{decimals}f}'

def format_factor_template_value(value, decimals=3):
    """
    factor를 고득점자 스타일로 소수점 3자리까지 표시한다.
    """
    return format_truncated_decimal(value, decimals=decimals)

def get_factor_template_float(value, decimals=3):
    """
    계산에 사용할 factor template float.
    """
    return truncate_float(value, decimals=decimals)

def format_accumulated_value(value, decimals):
    """
    누적 factor 값을 현재 자리수에 맞게 출력한다.
    """
    return f'{value:.{decimals}f}'

def format_place_value(pos):
    """
    자리값을 고득점자 스타일로 출력한다.

    pos=1 -> 0.1
    pos=2 -> 0.01
    pos=3 -> 0.001
    """
    return f'{10 ** (-pos):.{pos}f}'

def build_highscore_style_division_trace(numerator, denominator, factor_decimals=3):
    """
    고득점자 unit COT와 유사한 factor 나눗셈 로그를 만든다.

    출력 예:
        = 0 + 1 * 1846 / 1756
        = 1 + 1 * 90 / 1756
        = 1.0 + 0.1 * 900 / 1756
        = 1.00 + 0.01 * 9000 / 1756
        = 1.01 + 0.01 * 7244 / 1756
        ...
        = 1.051

    목적:
        - 모델이 factor = output / input 계산 패턴을 반복적으로 학습하게 한다.
        - 고득점자 데이터셋처럼 계산 템플릿을 길게 보여준다.
    """
    numerator = int(numerator)
    denominator = int(denominator)
    lines = []
    if denominator == 0:
        lines.append('= undefined_division_by_zero')
        return lines
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    n = abs(numerator)
    d = abs(denominator)
    integer_part = n // d
    remainder = n % d
    lines.append(f'= 0 + 1 * {numerator} / {denominator}')
    current_value = integer_part * sign
    if integer_part != 0:
        signed_remainder = remainder if sign > 0 else -remainder
        lines.append(f'= {current_value} + 1 * {signed_remainder} / {denominator}')
    for pos in range(1, factor_decimals + 1):
        place = 10 ** (-pos)
        place_text = format_place_value(pos)
        expanded = remainder * 10
        current_text = format_accumulated_value(current_value, pos)
        signed_expanded = expanded if sign > 0 else -expanded
        lines.append(f'= {current_text} + {place_text} * {signed_expanded} / {denominator}')
        digit = expanded // d
        new_remainder = expanded % d
        for step in range(1, int(digit) + 1):
            current_value += sign * place
            remaining_after_step = expanded - step * d
            signed_remaining = remaining_after_step if sign > 0 else -remaining_after_step
            current_text = format_accumulated_value(current_value, pos)
            lines.append(f'= {current_text} + {place_text} * {signed_remaining} / {denominator}')
        remainder = new_remainder
    approx_value = numerator / denominator
    lines.append(f'= {format_factor_template_value(approx_value, decimals=factor_decimals)}')
    return lines

def build_unit_examples_log(examples):
    """
    Compact example list.
    """
    lines = ['Examples:']
    if examples is None or len(examples) == 0:
        lines.append('none')
        return '\n'.join(lines)
    for idx, (x, y) in enumerate(examples):
        lines.append(f'{idx}: {float(x):.2f} -> {float(y):.2f}')
    return '\n'.join(lines)

def build_unit_factor_calculation_log(item, factor_decimals=3):
    """
    example 하나에서 factor = output / input 계산 로그를 만든다.
    고득점자 데이터셋 스타일과 최대한 유사하게 출력한다.
    """
    x = item['input']
    y = item['output']
    scaled_y = decimal_to_scaled_int(y, decimals=2)
    scaled_x = decimal_to_scaled_int(x, decimals=2)
    factor_template = format_factor_template_value(item['raw_ratio'], decimals=factor_decimals)
    lines = []
    lines.append(f'{x:.2f} -> {y:.2f}')
    lines.append(f'Casting input to 2 decimal places, output to 2 decimal places: {x:.2f} -> {y:.2f}')
    lines.append(f'factor = {y:.2f} / {x:.2f}')
    division_trace = build_highscore_style_division_trace(numerator=scaled_y, denominator=scaled_x, factor_decimals=factor_decimals)
    lines.extend(division_trace)
    lines.append(f'factor_template={factor_template}')
    return '\n'.join(lines)

def median_from_sorted_values(sorted_values):
    """
    정렬된 값 리스트에서 median을 계산한다.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

def build_factor_summary_log(factor_templates):
    """
    고득점자 데이터셋 스타일의 factor summary.
    """
    factor_templates_sorted = sorted(factor_templates)
    median_factor = median_from_sorted_values(factor_templates_sorted)
    factor_values_text = ', '.join((f'{v:.3f}' for v in factor_templates))
    sorted_values_text = ', '.join((f'{v:.3f}' for v in factor_templates_sorted))
    lines = []
    lines.append(f'factor values: {factor_values_text}')
    lines.append(f'factor values (sorted): {sorted_values_text}')
    lines.append(f'The median factor is {median_factor:.3f}.')
    return ('\n'.join(lines), median_factor)

def decompose_decimal_multiplier(value, decimals=3):
    """
    multiplier를 자리값 단위로 분해한다.

    예:
        1.051 -> [1.000, 0.050, 0.001]
        0.920 -> [0.900, 0.020]
        1.549 -> [1.000, 0.500, 0.040, 0.009]
    """
    value = float(value)
    sign = -1 if value < 0 else 1
    value_abs = abs(value)
    text = f'{value_abs:.{decimals}f}'
    integer_part, decimal_part = text.split('.')
    terms = []
    for pos, digit_char in enumerate(integer_part):
        digit = int(digit_char)
        if digit == 0:
            continue
        power = len(integer_part) - pos - 1
        place_value = digit * 10 ** power
        terms.append(sign * place_value)
    for pos, digit_char in enumerate(decimal_part, start=1):
        digit = int(digit_char)
        if digit == 0:
            continue
        place_value = digit * 10 ** (-pos)
        terms.append(sign * place_value)
    if len(terms) == 0:
        terms.append(0.0)
    return terms

def format_multiplier_term(term):
    """
    query multiplication 로그용 term 포맷.
    """
    return f'{term:.3f}'

def build_unit_query_calculation_log(query_value, median_factor, factor_decimals=3):
    """
    query * median factor 계산 로그.
    고득점자 데이터셋처럼 자리값별 곱셈과 누적합을 보여준다.
    """
    terms = decompose_decimal_multiplier(median_factor, decimals=factor_decimals)
    terms_for_display = sorted(terms, key=lambda x: abs(x))
    prediction = query_value * median_factor
    answer = format_truncated_decimal(prediction, decimals=factor_decimals)
    lines = []
    lines.append(f'Converting {query_value:g}:')
    lines.append(f'{query_value:g} * {median_factor:.3f}:')
    partial_values = []
    for term in terms_for_display:
        partial = query_value * term
        partial_values.append(partial)
        lines.append(f'{query_value:g} * {format_multiplier_term(term)} = {partial:.5f}')
    running_sum = 0.0
    for idx, partial in enumerate(partial_values):
        if idx == 0:
            running_sum = partial
            continue
        previous = running_sum
        running_sum += partial
        lines.append(f'{previous:.5f} + {partial:.5f} = {running_sum:.5f}')
    lines.append(f'= {answer}')
    return ('\n'.join(lines), answer)

def solve_unit_conversion(examples, query):
    """
    Unit Conversion 문제를 해결하는 함수.

    고득점자 데이터셋 방식:
        1. 각 example에서 factor = output / input 계산
        2. factor를 소수점 3자리 템플릿으로 변환
        3. factor values를 정렬
        4. median factor 선택
        5. query에 median factor를 곱함
        6. 최종 output은 소수점 3자리까지 버림 출력

    v15:
        - assistant.reasoning_content에는 계산 템플릿형 solution 저장
        - assistant.content에는 \\boxed{answer}만 저장
    """
    rule_name = 'median_3_decimal_factor_template'
    solver_name = 'unit_conversion_solver'
    if query is None or examples is None or len(examples) == 0:
        return make_solver_result(solved=False, rule_name=rule_name, solver_name=solver_name, solution='Task: unit_conversion\nstatus=unsolved\nreason=query_or_examples_missing')
    try:
        parsed_examples = []
        factor_templates = []
        for example_idx, (x, y) in enumerate(examples):
            x = float(x)
            y = float(y)
            if x == 0:
                continue
            raw_ratio = y / x
            factor_template = get_factor_template_float(raw_ratio, decimals=3)
            parsed_examples.append({'idx': example_idx, 'input': x, 'output': y, 'raw_ratio': raw_ratio, 'factor_template': factor_template})
            factor_templates.append(factor_template)
        if len(parsed_examples) == 0:
            return make_solver_result(solved=False, rule_name=rule_name, solver_name=solver_name, solution='Task: unit_conversion\nstatus=unsolved\nreason=no_valid_examples')
        factor_summary_log, median_factor = build_factor_summary_log(factor_templates=factor_templates)
        query_value = float(query)
        query_calculation_log, answer = build_unit_query_calculation_log(query_value=query_value, median_factor=median_factor, factor_decimals=3)
        factor_logs = '\n\n'.join((build_unit_factor_calculation_log(item=item, factor_decimals=3) for item in parsed_examples))
        solution = f'Task: unit_conversion\nQuery: {query_value:g}\n\n{build_unit_examples_log(examples)}\n\n{factor_logs}\n\n{factor_summary_log}\n\n{query_calculation_log}\n\nFinal output:\n{answer}\n\nFinal boxed answer:\n\\boxed{{{answer}}}'
        return make_solver_result(solved=True, answer=answer, solution=solution, rule_name=rule_name, solver_name=solver_name)
    except Exception as e:
        return make_solver_result(solved=False, rule_name=rule_name, solver_name=solver_name, solution=f'Task: unit_conversion\nstatus=solver_exception\nerror={type(e).__name__}: {e}')


# ===== Extracted from notebook cell 37 =====
def metric_like_match(true, pred):
    """
    Kaggle metric과 유사한 방식으로 정답 여부를 판단하는 함수

    숫자형:
        상대 오차 1e-2, 절대 오차 1e-5 허용

    문자열:
        대소문자 무시 후 exact match
    """
    true = str(true).strip()
    pred = str(pred).strip()
    if re.fullmatch('[01]+', true):
        return pred.lower() == true.lower()
    try:
        true_num = float(true)
        pred_num = float(pred)
        return math.isclose(true_num, pred_num, rel_tol=0.01, abs_tol=1e-05)
    except:
        return pred.lower() == true.lower()


# ===== Extracted from notebook cell 40 =====
def truncate_float(value, decimals=3):
    """
    고득점자 gravity COT 스타일에 맞춰 소수점 decimals 자리에서 버림 처리한다.

    예:
        156.3470771 -> 156.347
        8.339812    -> 8.339
    """
    value = float(value)
    scale = 10 ** decimals
    if value >= 0:
        return math.floor(value * scale) / scale
    return math.ceil(value * scale) / scale

def format_truncated_decimal(value, decimals=3):
    """
    소수점 decimals 자리까지 버림 후 고정 자리수 문자열로 출력한다.
    """
    return f'{truncate_float(value, decimals=decimals):.{decimals}f}'

def decimal_to_scaled_int_truncate(value, decimals=3):
    """
    고득점자 gravity 로그 스타일에 맞춰 소수점 decimals 자리까지 버림 후 정수화한다.

    예:
        177.99   -> 177990
        21.3444  -> 21344   # 21.344로 버림
        8.8209   -> 8820
    """
    value = Decimal(str(value))
    scale = Decimal(10) ** decimals
    if value >= 0:
        return int(value * scale)
    return int(value * scale)

def format_number_compact(value, decimals=2):
    """
    prompt에 나온 숫자처럼 불필요한 trailing zero를 줄인다.
    """
    text = f'{float(value):.{decimals}f}'
    text = text.rstrip('0').rstrip('.')
    if text == '-0':
        text = '0'
    return text

def decompose_decimal_number(value, decimals=2):
    """
    value를 자리값 단위로 분해한다.

    예:
        4.62 -> [4.00, 0.60, 0.02]
        18.7489 -> [10.0000, 8.0000, 0.7000, 0.0400, 0.0080, 0.0009]
    """
    value = float(value)
    sign = -1 if value < 0 else 1
    value_abs = abs(value)
    text = f'{value_abs:.{decimals}f}'
    integer_part, decimal_part = text.split('.')
    terms = []
    for pos, digit_char in enumerate(integer_part):
        digit = int(digit_char)
        if digit == 0:
            continue
        power = len(integer_part) - pos - 1
        place_value = digit * 10 ** power
        terms.append(sign * place_value)
    for pos, digit_char in enumerate(decimal_part, start=1):
        digit = int(digit_char)
        if digit == 0:
            continue
        place_value = digit * 10 ** (-pos)
        terms.append(sign * place_value)
    if len(terms) == 0:
        terms.append(0.0)
    return terms

def format_term_for_square(term):
    """
    t^2 계산 로그용 term 포맷.

    고득점자 예:
        4.62 * 0.02
        4.62 * 0.60
        4.62 * 4.00
    """
    return f'{term:.2f}'

def format_term_for_distance(term):
    """
    d = k * t^2 계산 로그용 term 포맷.

    고득점자 예:
        8.339 * 0.0009
        8.339 * 0.0080
        8.339 * 10.0000
    """
    return f'{term:.4f}'

def build_square_calculation_log(t):
    """
    t^2 = t * t 계산 로그를 고득점자 스타일로 만든다.

    예:
        t^2 = 4.62 * 4.62:
        4.62 * 0.02 = 0.0924
        4.62 * 0.60 = 2.7720
        4.62 * 4.00 = 18.4800
        0.0924 + 2.7720 = 2.8644
        2.8644 + 18.4800 = 21.3444
    """
    t = float(t)
    terms = decompose_decimal_number(value=t, decimals=2)
    terms_for_display = sorted(terms, key=lambda x: abs(x))
    partial_values = [t * term for term in terms_for_display]
    lines = []
    lines.append(f't^2 = {t:.2f} * {t:.2f}:')
    for term, partial in zip(terms_for_display, partial_values):
        lines.append(f'{t:.2f} * {format_term_for_square(term)} = {partial:.4f}')
    running_sum = 0.0
    for idx, partial in enumerate(partial_values):
        if idx == 0:
            running_sum = partial
            continue
        previous = running_sum
        running_sum += partial
        lines.append(f'{previous:.4f} + {partial:.4f} = {running_sum:.4f}')
    t_sq = t ** 2
    return ('\n'.join(lines), t_sq)

def format_accumulated_value(value, decimals):
    """
    k 누적값을 현재 자리수에 맞게 출력한다.
    """
    return f'{value:.{decimals}f}'

def format_place_value(pos):
    """
    pos=1 -> 0.1
    pos=2 -> 0.01
    pos=3 -> 0.001
    """
    return f'{10 ** (-pos):.{pos}f}'

def build_k_division_trace(numerator, denominator, k_decimals=3):
    """
    고득점자 gravity COT와 유사하게 k = d / t^2 나눗셈 로그를 만든다.

    예:
        = 0 + 1 * 177990 / 21344
        = 1 + 1 * 156646 / 21344
        ...
        = 8.339
    """
    numerator = int(numerator)
    denominator = int(denominator)
    lines = []
    if denominator == 0:
        lines.append('= undefined_division_by_zero')
        return lines
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    n = abs(numerator)
    d = abs(denominator)
    current_value = 0.0
    remainder = n
    lines.append(f'= 0 + 1 * {numerator} / {denominator}')
    integer_digit = n // d
    for step in range(1, int(integer_digit) + 1):
        current_value += sign * 1.0
        remainder = n - step * d
        signed_remainder = remainder if sign > 0 else -remainder
        current_text = str(int(current_value))
        lines.append(f'= {current_text} + 1 * {signed_remainder} / {denominator}')
    remainder = n % d
    for pos in range(1, k_decimals + 1):
        place = 10 ** (-pos)
        place_text = format_place_value(pos)
        expanded = remainder * 10
        current_text = format_accumulated_value(current_value, pos)
        signed_expanded = expanded if sign > 0 else -expanded
        lines.append(f'= {current_text} + {place_text} * {signed_expanded} / {denominator}')
        digit = expanded // d
        new_remainder = expanded % d
        for step in range(1, int(digit) + 1):
            current_value += sign * place
            remaining_after_step = expanded - step * d
            signed_remaining = remaining_after_step if sign > 0 else -remaining_after_step
            current_text = format_accumulated_value(current_value, pos)
            lines.append(f'= {current_text} + {place_text} * {signed_remaining} / {denominator}')
        remainder = new_remainder
    raw_value = numerator / denominator
    lines.append(f'= {format_truncated_decimal(raw_value, decimals=k_decimals)}')
    return lines

def build_formula_examples_log(examples):
    """
    Compact example list.
    """
    lines = ['Examples:']
    if examples is None or len(examples) == 0:
        lines.append('none')
        return '\n'.join(lines)
    for idx, (t, d) in enumerate(examples):
        lines.append(f'{idx}: t={float(t):.2f}, d={float(d):.2f}')
    return '\n'.join(lines)

def build_formula_k_calculation_log(item, k_decimals=3):
    """
    example 하나에서 k = d / t^2 계산 로그를 만든다.
    고득점자 gravity 데이터셋 스타일과 최대한 유사하게 출력한다.
    """
    t = item['t']
    d = item['d']
    t_sq = item['t_sq']
    scaled_d = decimal_to_scaled_int_truncate(d, decimals=3)
    scaled_t_sq = decimal_to_scaled_int_truncate(t_sq, decimals=3)
    k_template = format_truncated_decimal(item['raw_k'], decimals=k_decimals)
    square_log, _ = build_square_calculation_log(t)
    lines = []
    lines.append(f't = {t:.2f}s, d = {d:.2f}m:')
    lines.append(square_log)
    lines.append(f'k = {d:.2f} / {t:.2f}^2 = {d:.2f} / {t_sq:.4f} = {d:.3f} / {truncate_float(t_sq, decimals=3):.3f}')
    division_trace = build_k_division_trace(numerator=scaled_d, denominator=scaled_t_sq, k_decimals=k_decimals)
    lines.extend(division_trace)
    lines.append(f'k_template={k_template}')
    return '\n'.join(lines)

def median_from_sorted_values(sorted_values):
    """
    정렬된 값 리스트에서 median을 계산한다.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2

def build_k_summary_log(k_templates):
    """
    고득점자 데이터셋 스타일의 k summary.
    """
    k_templates_sorted = sorted(k_templates)
    median_k = median_from_sorted_values(k_templates_sorted)
    k_values_text = ', '.join((f'{v:.3f}' for v in k_templates))
    sorted_values_text = ', '.join((f'{v:.3f}' for v in k_templates_sorted))
    lines = []
    lines.append(f'k values: {k_values_text}')
    lines.append(f'k values (sorted): {sorted_values_text}')
    lines.append(f'The median k is {median_k:.3f}.')
    return ('\n'.join(lines), median_k)

def build_query_distance_calculation_log(query_t, median_k, answer_decimals=3):
    """
    query t에 대해 t^2를 계산하고 d = median_k * t^2를 계산하는 로그.
    고득점자 gravity 데이터셋처럼 자리값 곱셈과 부분합을 보여준다.
    """
    query_t = float(query_t)
    square_log, query_t_sq = build_square_calculation_log(query_t)
    terms = decompose_decimal_number(value=query_t_sq, decimals=4)
    terms_for_display = sorted(terms, key=lambda x: abs(x))
    partial_values = [median_k * term for term in terms_for_display]
    prediction = median_k * query_t_sq
    answer = format_truncated_decimal(prediction, decimals=answer_decimals)
    lines = []
    lines.append(f'For t = {query_t:g}:')
    lines.append(square_log)
    lines.append(f'= {query_t_sq:.4f}')
    lines.append('')
    lines.append(f'd = {median_k:.3f} * {query_t_sq:.4f}:')
    for term, partial in zip(terms_for_display, partial_values):
        lines.append(f'{median_k:.3f} * {format_term_for_distance(term)} = {partial:.7f}')
    running_sum = 0.0
    for idx, partial in enumerate(partial_values):
        if idx == 0:
            running_sum = partial
            continue
        previous = running_sum
        running_sum += partial
        lines.append(f'{previous:.7f} + {partial:.7f} = {running_sum:.7f}')
    lines.append(f'= {answer}')
    return ('\n'.join(lines), answer)

def solve_formula_based(examples, query):
    """
    Falling distance formula 문제 전용 solver.

    고득점자 데이터셋 방식:
        1. d = k * t^2 로 놓는다.
           원래 공식 d = 0.5*g*t^2 이므로 k = 0.5*g.
        2. 각 example에서 t^2를 계산한다.
        3. 각 example에서 k = d / t^2를 계산한다.
        4. k를 소수점 3자리 템플릿으로 만든다.
        5. k values의 median을 선택한다.
        6. query t에 대해 t^2를 계산한다.
        7. d = median_k * t^2 를 계산한다.
        8. 최종 output은 소수점 3자리까지 버림 출력한다.

    v15:
        - assistant.reasoning_content에는 계산 템플릿형 solution 저장
        - assistant.content에는 \\boxed{answer}만 저장
    """
    rule_name = 'falling_distance_median_3_decimal_k_template'
    solver_name = 'formula_based_solver'
    if query is None or examples is None or len(examples) == 0:
        return make_solver_result(solved=False, rule_name=rule_name, solver_name=solver_name, solution='Task: formula_based\nstatus=unsolved\nreason=query_or_examples_missing')
    try:
        parsed_examples = []
        k_templates = []
        for example_idx, (t, d) in enumerate(examples):
            t = float(t)
            d = float(d)
            if t == 0:
                continue
            t_sq = t ** 2
            raw_k = d / t_sq
            k_template = get_factor_template_float(raw_k, decimals=3)
            parsed_examples.append({'idx': example_idx, 't': t, 'd': d, 't_sq': t_sq, 'raw_k': raw_k, 'k_template': k_template})
            k_templates.append(k_template)
        if len(parsed_examples) == 0:
            return make_solver_result(solved=False, rule_name=rule_name, solver_name=solver_name, solution='Task: formula_based\nstatus=unsolved\nreason=no_valid_examples')
        k_summary_log, median_k = build_k_summary_log(k_templates=k_templates)
        query_t = float(query)
        query_calculation_log, answer = build_query_distance_calculation_log(query_t=query_t, median_k=median_k, answer_decimals=3)
        k_logs = '\n\n'.join((build_formula_k_calculation_log(item=item, k_decimals=3) for item in parsed_examples))
        solution = f'Task: formula_based\nQuery: {query_t:g}\n\n{build_formula_examples_log(examples)}\n\nRule identification:\nformula=d=k*t^2\nhidden_parameter=k\nrelationship_to_original_formula=k=0.5*g\nparameter_template=3_decimal_k_from_each_example\nparameter_selection=median_k\n\n{k_logs}\n\n{k_summary_log}\n\n{query_calculation_log}\n\nFinal output:\n{answer}\n\nFinal boxed answer:\n\\boxed{{{answer}}}'
        return make_solver_result(solved=True, answer=answer, solution=solution, rule_name=rule_name, solver_name=solver_name)
    except Exception as e:
        return make_solver_result(solved=False, rule_name=rule_name, solver_name=solver_name, solution=f'Task: formula_based\nstatus=solver_exception\nerror={type(e).__name__}: {e}')


# ===== Extracted from notebook cell 49 =====
import random

from collections import Counter

cipher_answer_vocab = set()

cipher_answer_vocab = set(sorted(cipher_answer_vocab))

def clean_word_token(word):
    word = str(word).lower().strip()
    word = re.sub('[^a-z]', '', word)
    return word

cipher_external_vocab = set()

cipher_noun_vocab = set()

cipher_display_vocab = set(sorted(cipher_answer_vocab))

cipher_selection_vocab = set(sorted(cipher_answer_vocab))

def format_vocab_table(words, cols=8):
    words = sorted(list(words))
    lines = []
    for i in range(0, len(words), cols):
        row = words[i:i + cols]
        lines.append(' | '.join(row))
    return '\n'.join(lines)

CIPHER_STYLE_VOCAB_TABLE_TEXT = format_vocab_table(cipher_selection_vocab, cols=8)

cipher_unigram_counts = Counter()

cipher_bigram_counts = Counter()


# ===== Extracted from notebook cell 50 =====
def safe_parse_examples_if_string(examples):
    """
    examples가 문자열이면 ast.literal_eval로 복원한다.
    이미 list/tuple이면 그대로 반환한다.
    """
    if isinstance(examples, str):
        try:
            return ast.literal_eval(examples)
        except Exception:
            return examples
    return examples

def format_chars_with_dash(text):
    return '–'.join(list(str(text)))

def pattern_matches(pattern, candidate):
    pattern = str(pattern).lower()
    candidate = str(candidate).lower()
    if len(pattern) != len(candidate):
        return False
    for p_char, c_char in zip(pattern, candidate):
        if p_char == '?':
            continue
        if p_char != c_char:
            return False
    return True


def cipher_word_pattern(word):
    """
    High-score reference style word pattern.

    Example:
        queen -> (0, 1, 2, 2, 3)
    This is used by the cipher candidate search so that repeated letters in the
    cipher word and candidate word have the same structural pattern.
    """
    seen = {}
    pattern = []
    for char in str(word).lower():
        if char not in seen:
            seen[char] = len(seen)
        pattern.append(seen[char])
    return tuple(pattern)
def get_vocab_candidates(pattern, vocab):
    """
    '?'가 포함된 decoded pattern에 대해 valid 후보 단어를 만든다.

    조건:
        1. vocab에서만 찾는다.
        2. 길이가 같아야 한다.
        3. fixed letter가 같은 위치에서 일치해야 한다.
    """
    pattern = str(pattern).lower()
    if '?' not in pattern:
        return []
    candidates = [word for word in sorted(vocab) if pattern_matches(pattern, word)]
    return candidates

def get_same_length_vocab_words(pattern, vocab, max_candidates=None):
    """
    같은 길이의 vocab 단어를 반환한다.

    이번 버전에서는 고득점자 스타일처럼 정답 vocab 전체를 보여주는 방향이므로
    max_candidates=None이면 같은 길이 단어를 전부 반환한다.
    """
    pattern = str(pattern).lower()
    target_len = len(pattern)
    same_length_words = [word for word in sorted(vocab) if len(word) == target_len]
    if max_candidates is not None:
        return same_length_words[:max_candidates]
    return same_length_words

def score_candidate_in_sentence(candidate, sentence_words, word_idx):
    """
    현재 cipher solver에서는 문맥 점수를 사용하지 않는다.
    """
    return 0.0

def choose_best_candidate(pattern, candidates, sentence_words=None, word_idx=None):
    """
    후보 중 최종 단어를 선택한다.

    정책:
        1. candidates는 이미 strict validity check를 통과한 후보여야 한다.
        2. 문맥 점수는 사용하지 않는다.
        3. valid 후보가 없으면 pattern 그대로 반환한다.
        4. valid 후보가 있으면 alphabetic order로 deterministic하게 선택한다.
    """
    if len(candidates) == 0:
        return pattern
    candidates = sorted(set(candidates))
    return candidates[0]


# ===== Extracted from notebook cell 51 =====
def build_reverse_mapping(mapping):
    """
    plain character -> cipher character reverse mapping을 만든다.

    mapping:
        cipher_char -> plain_char

    reverse_mapping:
        plain_char -> cipher_char
    """
    reverse_mapping = {}
    for cipher_char, plain_char in mapping.items():
        if plain_char not in reverse_mapping:
            reverse_mapping[plain_char] = cipher_char
    return reverse_mapping

def get_fixed_position_checks(decoded_pattern, candidate):
    """
    decoded_pattern에서 이미 확정된 문자 위치와 candidate를 비교한다.
    """
    decoded_pattern = str(decoded_pattern).lower()
    candidate = str(candidate).lower()
    fixed_checks = []
    mismatch_checks = []
    max_len = min(len(decoded_pattern), len(candidate))
    for pos in range(max_len):
        expected_char = decoded_pattern[pos]
        actual_char = candidate[pos]
        if expected_char == '?':
            continue
        item = {'position': pos + 1, 'expected': expected_char, 'actual': actual_char, 'match': expected_char == actual_char}
        fixed_checks.append(item)
        if not item['match']:
            mismatch_checks.append(item)
    return (fixed_checks, mismatch_checks)

def format_fixed_position_summary(decoded_pattern):
    decoded_pattern = str(decoded_pattern).lower()
    items = []
    for idx, char in enumerate(decoded_pattern, start=1):
        if char != '?':
            items.append(f"position {idx} = '{char}'")
    if len(items) == 0:
        return 'none'
    return ', '.join(items)

def check_candidate_mapping_consistency(cipher_word, decoded_pattern, candidate, mapping):
    """
    candidate word가 현재 mapping과 양방향으로 일관적인지 확인한다.

    조건:
        1. 길이가 정확히 같아야 한다.
        2. decoded_pattern에서 이미 해독된 글자는 candidate와 같은 위치에서 일치해야 한다.
        3. '?' 위치의 original cipher character는 아직 mapping에 없어야 한다.
        4. 새 plain letter가 이미 다른 cipher character의 output으로 사용 중이면 충돌이다.
        5. 같은 unknown cipher character가 반복되면 같은 plain letter로만 매핑되어야 한다.
    """
    cipher_word = str(cipher_word).lower()
    decoded_pattern = str(decoded_pattern).lower()
    candidate = str(candidate).lower()
    expected_len = len(decoded_pattern)
    candidate_len = len(candidate)
    cipher_len = len(cipher_word)
    if candidate_len != expected_len:
        return {'valid': False, 'reason': 'length mismatch', 'new_mappings': [], 'details': {'expected_len': expected_len, 'candidate_len': candidate_len}}
    if cipher_len != candidate_len:
        return {'valid': False, 'reason': 'cipher word length mismatch', 'new_mappings': [], 'details': {'cipher_len': cipher_len, 'candidate_len': candidate_len}}
    fixed_checks, fixed_mismatches = get_fixed_position_checks(decoded_pattern=decoded_pattern, candidate=candidate)
    if len(fixed_mismatches) > 0:
        return {'valid': False, 'reason': 'fixed letters mismatch', 'new_mappings': [], 'details': {'fixed_checks': fixed_checks, 'fixed_mismatches': fixed_mismatches, 'first_mismatch': fixed_mismatches[0]}}
    reverse_mapping = build_reverse_mapping(mapping)
    new_mappings = []
    local_new_mapping = {}
    for pos, (cipher_char, decoded_char, candidate_char) in enumerate(zip(cipher_word, decoded_pattern, candidate), start=1):
        if decoded_char != '?':
            if cipher_char in mapping and mapping[cipher_char] != candidate_char:
                return {'valid': False, 'reason': 'known mapping conflict', 'new_mappings': [], 'details': {'position': pos, 'cipher_char': cipher_char, 'known_plain': mapping[cipher_char], 'candidate_plain': candidate_char}}
            continue
        if cipher_char in mapping:
            return {'valid': False, 'reason': 'unknown position already has mapping', 'new_mappings': [], 'details': {'position': pos, 'cipher_char': cipher_char, 'known_plain': mapping[cipher_char], 'candidate_plain': candidate_char}}
        if cipher_char in local_new_mapping:
            if local_new_mapping[cipher_char] != candidate_char:
                return {'valid': False, 'reason': 'local repeated unknown conflict', 'new_mappings': [], 'details': {'position': pos, 'cipher_char': cipher_char, 'first_plain': local_new_mapping[cipher_char], 'candidate_plain': candidate_char}}
        else:
            local_new_mapping[cipher_char] = candidate_char
        if candidate_char in reverse_mapping and reverse_mapping[candidate_char] != cipher_char:
            return {'valid': False, 'reason': 'reverse mapping conflict', 'new_mappings': [], 'details': {'position': pos, 'candidate_plain': candidate_char, 'existing_cipher': reverse_mapping[candidate_char], 'new_cipher': cipher_char}}
    for cipher_char, plain_char in local_new_mapping.items():
        new_mappings.append((cipher_char, plain_char))
    return {'valid': True, 'reason': 'length match, fixed letters match, reverse mapping consistent', 'new_mappings': new_mappings, 'details': {'expected_len': expected_len, 'candidate_len': candidate_len, 'fixed_checks': fixed_checks}}

def classify_candidate_status(cipher_word, decoded_pattern, candidate, mapping):
    """
    candidate를 solution 출력용 한 줄 summary로 분류한다.

    출력 원칙:
        - 실제 검사 순서처럼 보이게 한다.
        - length mismatch는 바로 invalid.
        - fixed mismatch는 length match 이후 첫 번째 mismatch만 보여준다.
        - reverse mapping conflict는 length match, fixed letters match 이후 보여준다.
    """
    cipher_word = str(cipher_word).lower()
    decoded_pattern = str(decoded_pattern).lower()
    candidate = str(candidate).lower()
    expected_len = len(decoded_pattern)
    candidate_len = len(candidate)
    if candidate_len != expected_len:
        return {'valid': False, 'reason_key': 'length mismatch', 'summary': f'{candidate}: length mismatch, expected {expected_len} but got {candidate_len}; invalid', 'new_mappings': []}
    fixed_checks, fixed_mismatches = get_fixed_position_checks(decoded_pattern=decoded_pattern, candidate=candidate)
    if len(fixed_mismatches) > 0:
        first = fixed_mismatches[0]
        return {'valid': False, 'reason_key': 'fixed letters mismatch', 'summary': f"{candidate}: length match; fixed letters mismatch, position {first['position']} expected '{first['expected']}' but got '{first['actual']}'; invalid", 'new_mappings': []}
    consistency = check_candidate_mapping_consistency(cipher_word=cipher_word, decoded_pattern=decoded_pattern, candidate=candidate, mapping=mapping)
    if consistency['valid']:
        new_mapping_text = ', '.join((f'{c}->{p}' for c, p in consistency['new_mappings']))
        if new_mapping_text == '':
            new_mapping_text = 'none'
        return {'valid': True, 'reason_key': 'valid', 'summary': f'{candidate}: length match; fixed letters match; reverse mapping consistent; new mappings = {new_mapping_text}; valid', 'new_mappings': consistency['new_mappings']}
    reason = consistency['reason']
    details = consistency.get('details', {})
    if reason == 'known mapping conflict':
        summary = f"{candidate}: length match; fixed letters match; known mapping conflict at position {details['position']}, cipher '{details['cipher_char']}' already maps to '{details['known_plain']}' but candidate proposes '{details['candidate_plain']}'; invalid"
    elif reason == 'unknown position already has mapping':
        summary = f"{candidate}: length match; fixed letters match; unknown-position conflict at position {details['position']}, cipher '{details['cipher_char']}' already maps to '{details['known_plain']}' but candidate proposes '{details['candidate_plain']}'; invalid"
    elif reason == 'local repeated unknown conflict':
        summary = f"{candidate}: length match; fixed letters match; repeated unknown conflict at position {details['position']}, cipher '{details['cipher_char']}' was first proposed as '{details['first_plain']}' but later proposed as '{details['candidate_plain']}'; invalid"
    elif reason == 'reverse mapping conflict':
        summary = f"{candidate}: length match; fixed letters match; reverse mapping conflict at position {details['position']}, plain letter '{details['candidate_plain']}' is already produced by cipher '{details['existing_cipher']}'; invalid"
    else:
        summary = f'{candidate}: length match; fixed letters match; {reason}; invalid'
    return {'valid': False, 'reason_key': reason, 'summary': summary, 'new_mappings': []}

def get_consistent_vocab_candidates(cipher_word, decoded_pattern, vocab, mapping):
    """
    High-score reference candidate filter.

    It follows tonghuikang/nemotron reasoners/cipher.py behavior:
      1. same length only
      2. same repeated-letter word pattern as the cipher word
      3. fixed decoded letters must match
      4. only forward cipher->plain consistency is checked here

    Reverse/plain-target usage is not checked in this first filter; it is shown
    during left-to-right scan as `untargeted` and then removed before final
    selection, matching the reference reasoning flow.
    """
    cipher_word = str(cipher_word).lower()
    decoded_pattern = str(decoded_pattern).lower()
    target_len = len(decoded_pattern)
    target_pattern = cipher_word_pattern(cipher_word)
    candidates = []
    for candidate in sorted(vocab):
        candidate = str(candidate).lower()
        if len(candidate) != target_len:
            continue
        if cipher_word_pattern(candidate) != target_pattern:
            continue
        ok = True
        for pos, decoded_char in enumerate(decoded_pattern):
            if decoded_char != '?' and decoded_char != candidate[pos]:
                ok = False
                break
        if not ok:
            continue
        for c_char, p_char in zip(cipher_word, candidate):
            if c_char in mapping and mapping[c_char] != p_char:
                ok = False
                break
        if ok:
            candidates.append(candidate)
    return candidates

# ===== Extracted from notebook cell 52 =====
def cipher_dash_chars(text):
    """
    Convert a string into dash-separated characters.

    Example:
        knight -> k–n–i–g–h–t
    """
    return '–'.join(list(str(text)))

def cipher_dash_tokens(tokens):
    """
    Convert a token list into dash-separated text.

    Example:
        ['(j)', 'n', 'i'] -> (j)–n–i
    """
    return '–'.join((str(tok) for tok in tokens))

def get_unmapped_target_letters(mapping):
    """
    Return plain target letters that are not used by the current cipher->plain mapping.
    """
    mapped_targets = set(mapping.values())
    return [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in mapped_targets]

def get_query_unknown_letters(query_words, mapping):
    """
    Return cipher letters in the query that are still unmapped.
    """
    unknown_letters = set()
    for word in query_words:
        for c in str(word):
            if c not in mapping:
                unknown_letters.add(c)
    return sorted(unknown_letters)

def decode_query_word_with_display(word, mapping):
    """
    Decode one query word using the current mapping.

    Returns:
        decoded_pattern:
            Internal pattern for candidate matching.
            Unknown cipher characters are represented as '?'.

        display_pattern:
            Human-readable pattern.
            In this version, it is the same as decoded_pattern.
    """
    decoded_chars = []
    display_chars = []
    for char in str(word):
        if char in mapping:
            decoded_char = mapping[char]
            decoded_chars.append(decoded_char)
            display_chars.append(decoded_char)
        else:
            decoded_chars.append('?')
            display_chars.append('?')
    decoded_pattern = ''.join(decoded_chars)
    display_pattern = ''.join(display_chars)
    return (decoded_pattern, display_pattern)

def format_unknown_display_tokens(cipher_word, mapping):
    """
    Format one query word using the current mapping.

    Known:
        cipher char -> plain char

    Unknown:
        cipher char -> (cipher char)
    """
    tokens = []
    for c_char in str(cipher_word):
        if c_char in mapping:
            tokens.append(mapping[c_char])
        else:
            tokens.append(f'({c_char})')
    return tokens

def format_sentence_state_with_mapping(query_words, mapping):
    """
    Format the whole query sentence using the current mapping.

    Rule:
        - Fully decoded words are printed as normal words.
        - Words with unknown characters are printed as dash-separated tokens.

    Example:
        dragon (j)–a–t–c–h–e–s through (a)–o–r–e–s–t
    """
    sentence_parts = []
    for cipher_word in query_words:
        display_tokens = []
        for c_char in str(cipher_word):
            if c_char in mapping:
                display_tokens.append(mapping[c_char])
            else:
                display_tokens.append(f'({c_char})')
        has_unknown = any((str(token).startswith('(') and str(token).endswith(')') for token in display_tokens))
        if has_unknown:
            sentence_parts.append(cipher_dash_tokens(display_tokens))
        else:
            sentence_parts.append(''.join(display_tokens))
    return ' '.join(sentence_parts)

def decode_sentence_with_mapping(query_words, mapping):
    """
    Decode the whole query sentence using the current mapping.

    Unknown characters are represented as '?'.
    """
    decoded_words = []
    for word in query_words:
        decoded_chars = []
        for char in str(word):
            if char in mapping:
                decoded_chars.append(mapping[char])
            else:
                decoded_chars.append('?')
        decoded_words.append(''.join(decoded_chars))
    return ' '.join(decoded_words)

def format_cipher_mapping_alphabet(mapping):
    """
    Print cipher -> plain mapping for a-z.
    Unknown mappings are represented as '?'.
    """
    lines = []
    for c in 'abcdefghijklmnopqrstuvwxyz':
        lines.append(f"{c}->{mapping.get(c, '?')}")
    return '\n'.join(lines)

def format_plain_inverse_alphabet(mapping):
    """
    Print plain -> cipher inverse mapping for a-z.
    Unknown inverse mappings are represented as '?'.
    """
    reverse_mapping = build_reverse_mapping(mapping)
    lines = []
    for c in 'abcdefghijklmnopqrstuvwxyz':
        lines.append(f"{c}->{reverse_mapping.get(c, '?')}")
    return '\n'.join(lines)

def format_fixed_letters_line(decoded_pattern):
    """
    Format fixed decoded letters in the current word.

    Example:
        ?i?ard -> Fixed letters are 1〖i〗, 3〖a〗, 4〖r〗, 5〖d〗.

    Position index:
        0-based, matching the candidate scan lines.
    """
    decoded_pattern = str(decoded_pattern).lower()
    fixed_items = []
    for pos, char in enumerate(decoded_pattern):
        if char != '?':
            fixed_items.append(f'{pos}〖{char}〗')
    if len(fixed_items) == 0:
        return 'Fixed letters are none.'
    return 'Fixed letters are ' + ', '.join(fixed_items) + '.'

def format_left_to_right_candidate_detail(cipher_word, decoded_pattern, candidate, mapping):
    """
    High-score reference style candidate scan detail.

    This intentionally uses the wording used by the reference implementation:
      - known/fixed match:      `match`
      - known/fixed mismatch:   `unmatchable`
      - unknown new target:     `matchable`
      - unknown target already produced by another cipher char: `untargeted`
      - repeated local unknown with same proposed plain char: `consistent`
      - repeated local unknown with different proposed plain char: `contradiction`
    """
    cipher_word = str(cipher_word).lower()
    decoded_pattern = str(decoded_pattern).lower()
    candidate = str(candidate).lower()
    comparisons = []
    mismatch_found = False
    tentative = {}
    mapped_plain = set(mapping.values())
    for pos, (candidate_char, cipher_char) in enumerate(zip(candidate, cipher_word)):
        if cipher_char in mapping:
            plain_char = mapping[cipher_char]
            if candidate_char == plain_char:
                comparisons.append(f'{pos}〖{candidate_char}〗〖{plain_char}〗match')
            else:
                comparisons.append(f'{pos}〖{plain_char}〗〖{candidate_char}〗unmatchable')
                mismatch_found = True
                break
        else:
            if cipher_char in tentative:
                if tentative[cipher_char] == candidate_char:
                    comparisons.append(f'{pos}〖{candidate_char}〗〖({cipher_char})〗consistent')
                else:
                    comparisons.append(f'{pos}〖{candidate_char}〗〖({cipher_char})〗contradiction')
                    mismatch_found = True
                    break
            else:
                if candidate_char in mapped_plain:
                    comparisons.append(f'{pos}〖{candidate_char}〗〖({cipher_char})〗untargeted')
                    mismatch_found = True
                    break
                tentative[cipher_char] = candidate_char
                comparisons.append(f'{pos}〖{candidate_char}〗〖({cipher_char})〗matchable')
    detail = ', '.join(comparisons)
    if not mismatch_found:
        detail += f', {len(cipher_word)} all match'
    return {'valid': not mismatch_found, 'detail': detail, 'reason': 'valid' if not mismatch_found else 'mismatch'}

def format_reverse_mapping_conflict_detail(cipher_word, decoded_pattern, candidate, mapping):
    """
    Backward-compatible reverse mapping conflict formatter.

    This function is kept for compatibility, but candidate display now mainly uses
    format_left_to_right_candidate_detail().
    """
    result = format_left_to_right_candidate_detail(cipher_word=cipher_word, decoded_pattern=decoded_pattern, candidate=candidate, mapping=mapping)
    if not result['valid']:
        return result['detail']
    return 'mapped mismatch'

def format_valid_candidate_all_match_detail(cipher_word, decoded_pattern, candidate, mapping):
    """
    Format all position-level checks for a valid candidate.
    """
    result = format_left_to_right_candidate_detail(cipher_word=cipher_word, decoded_pattern=decoded_pattern, candidate=candidate, mapping=mapping)
    return result['detail']

def build_redecode_after_mapping_update_log(query_text, query_words, mapping, previous_mapping=None):
    """
    Print the current sentence state after a mapping update.

    If previous_mapping is given:
        The sentence currently is
        previous state -> current state

    If previous_mapping is not given:
        The sentence currently is
        current state
    """
    current_sentence = format_sentence_state_with_mapping(query_words=query_words, mapping=mapping)
    if previous_mapping is not None:
        previous_sentence = format_sentence_state_with_mapping(query_words=query_words, mapping=previous_mapping)
        if previous_sentence != current_sentence:
            return f'The sentence currently is\n{previous_sentence} -> {current_sentence}'
    return f'The sentence currently is\n{current_sentence}'

def build_unknown_letters_after_mapping_update_log(query_words, mapping):
    """
    After adding new mappings, print remaining unknown query letters and
    currently unmapped target letters.

    Output:
        The unknown letters
        a
        x

        Unmapped target letters
        b
        f
        ...
    """
    unknown_letters = get_query_unknown_letters(query_words=query_words, mapping=mapping)
    unmapped_target_letters = get_unmapped_target_letters(mapping)
    lines = []
    lines.append('The unknown letters')
    if len(unknown_letters) > 0:
        lines.append('\n'.join(unknown_letters))
    else:
        lines.append('none')
    lines.append('')
    lines.append('Unmapped target letters')
    if len(unmapped_target_letters) > 0:
        lines.append('\n'.join(unmapped_target_letters))
    else:
        lines.append('none')
    return '\n'.join(lines)

def build_cipher_highscore_style_header(examples, query):
    """
    Build high-score style header, input word listing, and character breakdown.
    """
    lines = []
    lines.append('We need to find the encryption mapping from the examples. It looks like a substitution cipher.')
    lines.append('I will put my final answer inside \\boxed{}.')
    lines.append('')
    lines.append('Listing the input words:')
    for cipher_text, plain_text in examples:
        cipher_text = str(cipher_text)
        cipher_words = cipher_text.split()
        lines.append('')
        lines.append(f'〖{cipher_text}〗')
        for idx, word in enumerate(cipher_words):
            prefix = '' if idx == 0 else ' '
            lines.append(f'{prefix}{word}')
    query_text = str(query)
    query_words = query_text.split()
    lines.append('')
    lines.append(f'〖 {query_text}〗')
    for word in query_words:
        lines.append(f' {word}')
    lines.append('')
    lines.append('Breaking down into characters:')
    for cipher_text, plain_text in examples:
        cipher_text = str(cipher_text)
        lines.append('')
        lines.append(f'〖{cipher_text}〗')
        for word in cipher_text.split():
            lines.append(cipher_dash_chars(word))
    lines.append('')
    lines.append(f'〖 {query_text}〗')
    for word in query_words:
        lines.append(cipher_dash_chars(word))
    return '\n'.join(lines)

def build_cipher_mapping_highscore_style(examples):
    """
    Build mapping from examples and create high-score style mapping log.

    Actual mapping:
        cipher character -> plain character

    Returns:
        mapping, log_text
    """
    mapping = {}
    lines = []
    for cipher_text, plain_text in examples:
        cipher_text = str(cipher_text)
        plain_text = str(plain_text)
        cipher_words = cipher_text.split()
        plain_words = plain_text.split()
        plain_quoted = ' '.join((f'〖{word}〗' for word in plain_words))
        lines.append('')
        lines.append('')
        lines.append(f'〖{cipher_text}〗 -> 〖{plain_text}〗 / {plain_quoted}:')
        lines.append('')
        for word_idx, (cipher_word, plain_word) in enumerate(zip(cipher_words, plain_words)):
            if len(cipher_word) != len(plain_word):
                continue
            new_pairs = []
            for c_char, p_char in zip(cipher_word, plain_word):
                if c_char not in mapping:
                    mapping[c_char] = p_char
                    new_pairs.append(f'{c_char}->{p_char}')
            cipher_word_display = f' {cipher_word}' if word_idx > 0 else cipher_word
            lines.append(f'〖{cipher_word_display}〗->〖{plain_word}〗\n{cipher_dash_chars(cipher_word)}->{cipher_dash_chars(plain_word)}')
            if len(new_pairs) > 0:
                lines.append('\n'.join(new_pairs))
            if word_idx < len(cipher_words) - 1:
                lines.append('')
        lines.append('')
        lines.append('Mapping so far')
        lines.append(format_cipher_mapping_alphabet(mapping))
    lines.append('Inverse mapping')
    lines.append(format_plain_inverse_alphabet(mapping))
    unknown_chars = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in mapping]
    unmapped_targets = get_unmapped_target_letters(mapping)
    lines.append('Unknown characters')
    lines.append('\n'.join(unknown_chars))
    lines.append('Unmapped target letters')
    lines.append('\n'.join(unmapped_targets))
    return (mapping, '\n'.join(lines))

def decode_cipher_query_highscore_style(query, mapping):
    """
    Decode the query using the current mapping and create high-score style log.

    Unknown characters are displayed as (cipher_char).

    Returns:
        decoded_words, unknown_word_records, unknown_chars, log_text
    """
    query_text = str(query)
    query_words = query_text.split()
    decoded_words = [''] * len(query_words)
    unknown_word_records = []
    all_unknown_chars = set()
    lines = []
    lines.append('')
    lines.append(f'Now decrypting 〖 {query_text}〗:')
    for word_idx, cipher_word in enumerate(query_words):
        mapping_steps = []
        display_tokens = []
        decoded_chars = []
        has_unknown = False
        for c_char in cipher_word:
            if c_char in mapping:
                p_char = mapping[c_char]
                mapping_steps.append(f'{c_char}->{p_char}')
                display_tokens.append(p_char)
                decoded_chars.append(p_char)
            else:
                mapping_steps.append(f'{c_char}->?')
                display_tokens.append(f'({c_char})')
                decoded_chars.append('?')
                all_unknown_chars.add(c_char)
                has_unknown = True
        decoded_pattern = ''.join(decoded_chars)
        display_dashed = cipher_dash_tokens(display_tokens)
        if word_idx > 0:
            lines.append('')
        if has_unknown:
            lines.append(f'〖 {cipher_word}〗\n{cipher_dash_chars(cipher_word)}\n{chr(10).join(mapping_steps)}\n{display_dashed}->〖{display_dashed}〗-> {display_dashed}')
            unknown_word_records.append({'word_idx': word_idx, 'cipher_word': cipher_word, 'decoded_pattern': decoded_pattern, 'display_tokens': display_tokens, 'display_dashed': display_dashed})
        else:
            lines.append(f'〖 {cipher_word}〗\n{cipher_dash_chars(cipher_word)}\n{chr(10).join(mapping_steps)}\n{display_dashed}->〖{decoded_pattern}〗-> {decoded_pattern}')
            decoded_words[word_idx] = decoded_pattern
    sentence_parts = []
    for word_idx, cipher_word in enumerate(query_words):
        if decoded_words[word_idx]:
            sentence_parts.append(decoded_words[word_idx])
        else:
            record = next((r for r in unknown_word_records if r['word_idx'] == word_idx))
            sentence_parts.append(record['display_dashed'])
    lines.append('')
    lines.append('The sentence currently is')
    lines.append(' '.join(sentence_parts))
    lines.append('')
    unknown_mapping_chars = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in mapping]
    if len(all_unknown_chars) > 0:
        lines.append('Iterating over the unknown letters to see if they are in the question')
        for c in unknown_mapping_chars:
            yes_no = 'yes' if c in all_unknown_chars else 'no'
            lines.append(f'{c} {yes_no}')
        lines.append('')
        lines.append('The unknown letters')
        lines.append('\n'.join(sorted(all_unknown_chars)))
        unmapped_target_letters = get_unmapped_target_letters(mapping)
        lines.append('')
        lines.append('Unmapped target letters')
        lines.append('\n'.join(unmapped_target_letters))
        lines.append('')
        lines.append('Let me find the best matching wonderland words:')
    else:
        lines.append('Iterating over the unknown letters to see if they are in the question: no unknown letters')
    return (decoded_words, unknown_word_records, sorted(all_unknown_chars), '\n'.join(lines))

def collect_cipher_candidate_display_words(cipher_word, decoded_pattern, mapping, valid_candidates, selected_word):
    """
    Build candidate display list from cipher_answer_vocab or current display vocab.

    Core rule:
        - Keep the user's current candidate pool behavior.
        - If cipher_answer_vocab exists, use it.
        - Otherwise, use cipher_display_vocab.
    """
    if 'cipher_answer_vocab' in globals():
        base_vocab = cipher_answer_vocab
    else:
        base_vocab = cipher_display_vocab
    candidate_pool = sorted({str(word).lower() for word in base_vocab if re.fullmatch('[a-z]+', str(word).lower())})
    if selected_word is not None and selected_word != decoded_pattern and re.fullmatch('[a-z]+', str(selected_word).lower()):
        selected_word = str(selected_word).lower()
        if selected_word not in candidate_pool:
            candidate_pool.append(selected_word)
            candidate_pool = sorted(set(candidate_pool))
    return candidate_pool

def build_candidate_word_scan_highscore_style(
    cipher_word,
    decoded_pattern,
    mapping,
    valid_candidates,
    selected_word,
    initial_mapping=None,
    orig_display_dashed=None,
):
    """
    High-score reference style candidate scan.

    It prints the original partially decoded word, accumulated new mappings,
    current partially decoded word, then scans every display-vocab word with
    the reference wording (`length`, `match`, `matchable`, `untargeted`, ...).
    """
    lines = []
    display_tokens = format_unknown_display_tokens(cipher_word=cipher_word, mapping=mapping)
    display_dashed = cipher_dash_tokens(display_tokens)
    if orig_display_dashed is None:
        orig_display_dashed = display_dashed
    lines.append('')
    lines.append(f'〖{orig_display_dashed}〗')
    if initial_mapping is not None:
        accumulated_new = [
            f'〖({c_char})〗->〖{mapping[c_char]}〗'
            for c_char in sorted(mapping)
            if c_char not in initial_mapping
        ]
        if accumulated_new:
            lines.append(f"New mappings: {', '.join(accumulated_new)}")
        else:
            lines.append('New mappings: none')
    lines.append(f'〖{display_dashed}〗')
    lines.append(f'The length of the word is {len(cipher_word)}.')

    displayed_words = collect_cipher_candidate_display_words(
        cipher_word=cipher_word,
        decoded_pattern=decoded_pattern,
        mapping=mapping,
        valid_candidates=valid_candidates,
        selected_word=selected_word,
    )
    if len(displayed_words) == 0:
        lines.append('No wonderland candidates were found.')
        lines.append('Best match: none')
        return '\n'.join(lines)

    for candidate in displayed_words:
        candidate = str(candidate).lower()
        candidate_dashed = cipher_dash_chars(candidate)
        candidate_len = len(candidate)
        if candidate_len != len(cipher_word):
            lines.append(f'{candidate} {candidate_len} length')
            continue
        left_to_right_result = format_left_to_right_candidate_detail(
            cipher_word=cipher_word,
            decoded_pattern=decoded_pattern,
            candidate=candidate,
            mapping=mapping,
        )
        lines.append(f"{candidate} {candidate_len} 〖{candidate_dashed}〗, {left_to_right_result['detail']}")
    if selected_word is None or selected_word == decoded_pattern:
        lines.append('Best match: none')
    else:
        lines.append(f'Best match: 〖{selected_word}〗')
    return '\n'.join(lines)

def build_selected_word_update_highscore_style(cipher_word, selected_word, mapping):
    """
    High-score reference style mapping update for the selected word.
    """
    lines = []
    new_mappings = []
    pending_mappings = []
    current_display_tokens = format_unknown_display_tokens(cipher_word=cipher_word, mapping=mapping)
    lines.append(f'〖{cipher_dash_tokens(current_display_tokens)}〗->〖{cipher_dash_chars(selected_word)}〗')
    for c_char, p_char in zip(cipher_word, selected_word):
        if c_char in mapping:
            known_plain = mapping[c_char]
            if known_plain == p_char:
                lines.append(f'〖{known_plain}〗->〖{p_char}〗same')
            else:
                lines.append(f'〖{known_plain}〗->〖{p_char}〗conflict')
        else:
            lines.append(f'〖({c_char})〗->〖{p_char}〗 new')
            if (c_char, p_char) not in pending_mappings:
                pending_mappings.append((c_char, p_char))
                new_mappings.append(f'〖({c_char})〗->〖{p_char}〗')
    if new_mappings:
        lines.append('Added mappings')
        lines.append('\n'.join(new_mappings))
    return ('\n'.join(lines), pending_mappings)

# ===== Extracted from notebook cell 53 =====
def solve_cipher(examples, query):
    """
    Cipher solver using the high-score reference-style reasoning flow.

    This version intentionally follows tonghuikang/nemotron's cipher reasoning
    text more closely than the previous no-reverse variant:
      - word-pattern candidate filtering
      - wonderland-style candidate scan wording
      - `untargeted` handling for already-mapped plain letters
      - deterministic first remaining valid candidate
    """
    base_rule_name = 'char_substitution_with_wonderland_vocab_scan'
    solver_name = 'cipher_solver_answer_vocab_wonderland_highscore_style'
    rule_name = base_rule_name
    try:
        if query is None or examples is None or len(examples) == 0:
            return make_solver_result(solved=False, answer=None, solution='Task: cipher\nstatus=unsolved\nreason=query_or_examples_missing', rule_name=rule_name, solver_name=solver_name)
        examples = safe_parse_examples_if_string(examples)
        normalized_examples = []
        for ex in examples:
            if isinstance(ex, dict):
                cipher_text = ex.get('input')
                plain_text = ex.get('output')
            else:
                cipher_text, plain_text = ex
            normalized_examples.append((str(cipher_text), str(plain_text)))

        query_text = str(query)
        solution_parts = []
        solution_parts.append(build_cipher_highscore_style_header(examples=normalized_examples, query=query_text))
        mapping, mapping_log = build_cipher_mapping_highscore_style(examples=normalized_examples)
        solution_parts.append(mapping_log)
        if len(mapping) == 0:
            return make_solver_result(solved=False, answer=None, solution='Task: cipher\nstatus=unsolved\nreason=empty_mapping', rule_name=rule_name, solver_name=solver_name)

        decoded_words, unknown_word_records, query_unknown_chars, decode_log = decode_cipher_query_highscore_style(query=query_text, mapping=mapping)
        solution_parts.append(decode_log)
        unknown_count = len(query_unknown_chars)
        rule_name = f'{base_rule_name}__unknown_count_{unknown_count}'
        query_words = query_text.split()
        initial_mapping = dict(mapping)

        for record in unknown_word_records:
            word_idx = record['word_idx']
            cipher_word = record['cipher_word']
            current_decoded_pattern, _ = decode_query_word_with_display(word=cipher_word, mapping=mapping)
            if '?' not in current_decoded_pattern:
                decoded_words[word_idx] = current_decoded_pattern
                continue

            # Reference first-pass candidate list: pattern + fixed letters + forward consistency.
            valid_candidates = get_consistent_vocab_candidates(
                cipher_word=cipher_word,
                decoded_pattern=current_decoded_pattern,
                vocab=cipher_selection_vocab,
                mapping=mapping,
            )

            # Reference final selection: remove candidates that map an unknown cipher
            # character to an already-used target/plain letter.
            unmapped_plain = {c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in set(mapping.values())}
            remaining_candidates = []
            for candidate in sorted(valid_candidates):
                bad = False
                for c_char, p_char in zip(cipher_word, candidate):
                    if c_char not in mapping and p_char not in unmapped_plain:
                        bad = True
                        break
                if not bad:
                    remaining_candidates.append(candidate)

            selected_word = choose_best_candidate(
                pattern=current_decoded_pattern,
                candidates=remaining_candidates,
                sentence_words=None,
                word_idx=None,
            )

            candidate_scan_log = build_candidate_word_scan_highscore_style(
                cipher_word=cipher_word,
                decoded_pattern=current_decoded_pattern,
                mapping=mapping,
                valid_candidates=valid_candidates,
                selected_word=selected_word,
                initial_mapping=initial_mapping,
                orig_display_dashed=record.get('display_dashed'),
            )
            solution_parts.append(candidate_scan_log)

            if selected_word == current_decoded_pattern:
                decoded_words[word_idx] = selected_word
                continue

            selected_update_log, new_pairs = build_selected_word_update_highscore_style(
                cipher_word=cipher_word,
                selected_word=selected_word,
                mapping=mapping,
            )
            solution_parts.append(selected_update_log)
            for c_char, p_char in new_pairs:
                mapping[c_char] = p_char
            decoded_words[word_idx] = selected_word

        answer = decode_sentence_with_mapping(query_words=query_words, mapping=mapping)
        solution_parts.append(f'I will now return the answer in \\boxed{{}}\nThe answer in \\boxed{{–}} is \\boxed{{{answer}}}')
        solution = '\n\n'.join(solution_parts)
        return make_solver_result(solved=True, answer=answer, solution=solution, rule_name=rule_name, solver_name=solver_name)
    except Exception as e:
        return make_solver_result(solved=False, answer=None, solution=f'Task: cipher\nstatus=solver_exception\nerror={type(e).__name__}: {e}', rule_name='cipher_solver_exception', solver_name=solver_name)

# ===== Extracted from notebook cell 64 =====
from collections import defaultdict

from dataclasses import dataclass

_NUMERIC_EXPR_RE = re.compile('^(\\d+)(\\D)(\\d+)$')

def parse_numeric_symbol_expr(expr):
    """
    고득점자 equation_numeric.py와 동일한 parser.

    input:
        55+61
        61-23
        48*52
        28:12
        22@75

    반환:
        left, operator, right

    주의:
        원본 repo는 음수 operand를 parse하지 않는다.
        즉 ^(\\d+)(\\D)(\\d+)$ 구조다.
    """
    if expr is None:
        return None
    expr = str(expr).strip()
    m = _NUMERIC_EXPR_RE.fullmatch(expr)
    if not m:
        return None
    return (m.group(1), m.group(2), m.group(3))

def numeric_quote(s):
    """
    고득점자 reasoning 로그에서 사용하는 〖...〗 포맷.
    """
    return f'〖{s}〗'

def numeric_rev(s):
    """
    고득점자 _rev(s)와 동일.
    음수면 부호는 유지하고 숫자 부분만 뒤집는다.
    """
    s = str(s)
    if s.startswith('-'):
        return '-' + s[1:][::-1]
    return s[::-1]

def numeric_common_candidates(a: int, b: int, sa: str, sb: str):
    """
    고득점자 _common_candidates와 동일한 common operation 후보.
    순서가 중요하다.
    """
    out = []
    out.append(('concatenation', sa + sb))
    out.append(('reverse concatenation', sb + sa))
    out.append(('addition', str(a + b)))
    out.append(('absolute difference', str(abs(a - b))))
    out.append(('negated absolute difference', str(-abs(a - b))))
    out.append(('subtraction (a-b)', str(a - b)))
    out.append(('reverse subtraction (b-a)', str(b - a)))
    out.append(('multiplication', str(a * b)))
    return out

def numeric_rare_candidates(a: int, b: int, sa: str, sb: str):
    """
    고득점자 _rare_candidates와 동일한 rare operation 후보.
    순서가 중요하다.
    """
    out = []
    out.append(('multiply+1', str(a * b + 1)))
    out.append(('multiply-1', str(a * b - 1)))
    out.append(('add+1', str(a + b + 1)))
    out.append(('add-1', str(a + b - 1)))
    out.append(('sub+1', str(a - b + 1)))
    out.append(('sub-1', str(a - b - 1)))
    if a != 0 and b != 0:
        big, small = (max(a, b), min(a, b))
        out.append(('max mod min', str(big % small)))
    if b != 0:
        out.append(('integer division (a/b)', str(a // b)))
        out.append(('modulo (a mod b)', str(a % b)))
    if a != 0:
        out.append(('reverse division (b/a)', str(b // a)))
        out.append(('reverse modulo (b mod a)', str(b % a)))
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = (int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1]))
        out.append(('digit absolute diff', str(abs(d1 - d3)) + str(abs(d2 - d4))))
        out.append(('digit add mod10', str((d1 + d3) % 10) + str((d2 + d4) % 10)))
        out.append(('digit sub mod10', str((d1 - d3) % 10) + str((d2 - d4) % 10)))
        out.append(('cross multiply', str(d1 * d3 + d2 * d4)))
        out.append(('cross multiply rev', str(d1 * d4 + d2 * d3)))
        out.append(('digit multiply', str(d1 * d3) + str(d2 * d4)))
        out.append(('digit multiply rev', str(d1 * d4) + str(d2 * d3)))
        out.append(('digit sum diff', str(d1 + d2 - (d3 + d4))))
        out.append(('digit sum sum', str(d1 + d2 + (d3 + d4))))
        out.append(('digit product diff', str(d1 * d2 - d3 * d4)))
        out.append(('digit product sum', str(d1 * d2 + d3 * d4)))
        det_val = d1 * d4 - d2 * d3
        out.append(('determinant', str(det_val)))
        out.append(('abs determinant', str(abs(det_val))))
    return out

def numeric_all_candidates(a: int, b: int, sa: str, sb: str):
    """
    common 먼저, rare 나중.
    """
    return numeric_common_candidates(a, b, sa, sb) + numeric_rare_candidates(a, b, sa, sb)


# ===== Extracted from notebook cell 65 =====
def numeric_expr_text(name: str, a: str, b: str) -> str:
    """
    고득점자 _expr(name, a, b)와 동일한 설명식 생성.
    """
    if name == 'addition':
        return f'{a} + {b}'
    if name == 'subtraction (a-b)':
        return f'{a} - {b}'
    if name == 'reverse subtraction (b-a)':
        return f'{b} - {a}'
    if name == 'multiplication':
        if len(a) >= 2:
            decomp = ' + '.join((str(int(d) * 10 ** (len(a) - 1 - i)) for i, d in enumerate(a)))
            return f'({decomp}) * {b}'
        return f'{a} * {b}'
    if name == 'absolute difference':
        return f'|{a} - {b}|'
    if name == 'negated absolute difference':
        return f'-|{a} - {b}|'
    if name == 'concatenation':
        return f'{a} || {b}'
    if name == 'reverse concatenation':
        return f'{b} || {a}'
    if name == 'multiply+1':
        if len(a) >= 2:
            decomp = ' + '.join((str(int(d) * 10 ** (len(a) - 1 - i)) for i, d in enumerate(a)))
            return f'({decomp}) * {b} + 1'
        return f'{a} * {b} + 1'
    if name == 'multiply-1':
        if len(a) >= 2:
            decomp = ' + '.join((str(int(d) * 10 ** (len(a) - 1 - i)) for i, d in enumerate(a)))
            return f'({decomp}) * {b} - 1'
        return f'{a} * {b} - 1'
    if name == 'add+1':
        return f'{a} + {b} + 1'
    if name == 'add-1':
        return f'{a} + {b} - 1'
    if name == 'sub+1':
        return f'{a} - {b} + 1'
    if name == 'sub-1':
        return f'{a} - {b} - 1'
    if name == 'integer division (a/b)':
        return f'{a} / {b}'
    if name == 'modulo (a mod b)':
        return f'{a} mod {b}'
    if name == 'reverse division (b/a)':
        return f'{b} / {a}'
    if name == 'reverse modulo (b mod a)':
        return f'{b} mod {a}'
    if name == 'max mod min':
        big, small = (a, b) if int(a) >= int(b) else (b, a)
        return f'max({a},{b}) mod min({a},{b}) = {big} mod {small}'
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = (a[0], a[1], b[0], b[1])
        if name == 'digit absolute diff':
            return f'|{d1}-{d3}| || |{d2}-{d4}|'
        if name == 'digit add mod10':
            return f'({d1}+{d3})%10 || ({d2}+{d4})%10'
        if name == 'digit sub mod10':
            return f'({d1}-{d3})%10 || ({d2}-{d4})%10'
        if name == 'cross multiply':
            return f'{d1}*{d3} + {d2}*{d4}'
        if name == 'cross multiply rev':
            return f'{d1}*{d4} + {d2}*{d3}'
        if name == 'digit multiply':
            return f'{d1}*{d3} || {d2}*{d4}'
        if name == 'digit multiply rev':
            return f'{d1}*{d4} || {d2}*{d3}'
        if name == 'digit sum diff':
            return f'({d1}+{d2}) - ({d3}+{d4})'
        if name == 'digit sum sum':
            return f'({d1}+{d2}) + ({d3}+{d4})'
        if name == 'digit product diff':
            return f'{d1}*{d2} - {d3}*{d4}'
        if name == 'digit product sum':
            return f'{d1}*{d2} + {d3}*{d4}'
        if name == 'determinant':
            return f'{d1}*{d4} - {d2}*{d3}'
        if name == 'abs determinant':
            return f'|{d1}*{d4} - {d2}*{d3}|'
    return ''

def numeric_expr_intermediate(name: str, a: str, b: str) -> str:
    """
    고득점자 _expr_intermediate(name, a, b)와 동일한 중간 계산식.
    """
    ia, ib = (int(a), int(b))
    if name in ('multiply+1', 'multiply-1', 'multiplication') and len(a) >= 2:
        places = [int(d) * 10 ** (len(a) - 1 - i) for i, d in enumerate(a)]
        decomp = ' + '.join((f'{p} * {b}' for p in places))
        evald = ' + '.join((str(p * ib) for p in places))
        product_sum = sum((p * ib for p in places))
        if name == 'multiply+1':
            return f'{decomp} + 1 = {evald} + 1 = {product_sum} + 1'
        if name == 'multiply-1':
            return f'{decomp} - 1 = {evald} - 1 = {product_sum} - 1'
        return f'{decomp} = {evald}'
    if len(a) == 2 and len(b) == 2:
        d1, d2, d3, d4 = (int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        if name == 'cross multiply':
            return f'{d1 * d3} + {d2 * d4}'
        if name == 'cross multiply rev':
            return f'{d1 * d4} + {d2 * d3}'
        if name == 'digit multiply':
            return f'{d1 * d3} || {d2 * d4}'
        if name == 'digit multiply rev':
            return f'{d1 * d4} || {d2 * d3}'
        if name == 'digit product diff':
            return f'{d1 * d2} - {d3 * d4}'
        if name == 'digit product sum':
            return f'{d1 * d2} + {d3 * d4}'
        if name == 'determinant':
            return f'{d1 * d4} - {d2 * d3}'
        if name == 'abs determinant':
            return f'|{d1 * d4} - {d2 * d3}|'
    return ''


# ===== Extracted from notebook cell 66 =====
@dataclass
class NumericFoundOp:
    op_name: str
    rev_ops: bool
    rev_res: bool
    fmt: str
    op_char: str

def numeric_apply_found_op(found: NumericFoundOp, a_str: str, b_str: str):
    """
    고득점자 _apply_op(found, a_str, b_str)와 동일.

    반환:
        result, explanation_lines
    """
    steps = []
    ta = a_str[::-1] if found.rev_ops else a_str
    tb = b_str[::-1] if found.rev_ops else b_str
    if found.rev_ops and found.rev_res:
        steps.append(f'reversed operands [{a_str}->{ta}, {b_str}->{tb}] and reversed result')
    elif found.rev_ops:
        steps.append(f'reversed operands [{a_str}->{ta}, {b_str}->{tb}]')
    elif found.rev_res:
        steps.append('reversed result')
    else:
        steps.append('identity')
    raw_result = ''
    for name, res in numeric_all_candidates(int(ta), int(tb), ta, tb):
        if name == found.op_name:
            raw_result = res
            break
    final = numeric_rev(raw_result) if found.rev_res else raw_result
    expr = numeric_expr_text(found.op_name, ta, tb)
    inter = numeric_expr_intermediate(found.op_name, ta, tb)
    if expr and inter:
        detail = f' {expr} = {inter} ='
    elif expr:
        detail = f' {expr} ='
    else:
        detail = ''
    val = f'{raw_result} -rev-> {final}' if found.rev_res else final
    steps.append(f'{found.op_name} f({ta}, {tb}) ={detail} {val}')
    if found.fmt == 'pre':
        final = found.op_char + final
        steps.append(f'Prefix operator: {final}')
    elif found.fmt == 'neg_suffix':
        if final.startswith('-'):
            old = final
            final = final[1:] + found.op_char
            steps.append(f'Result is negative - we add back the operator suffix 〖{found.op_char}〗: {old} -> 〖{final}〗')
        else:
            steps.append(f'Result is non-negative, no suffix needed: 〖{final}〗')
    elif found.fmt == 'neg_prefix':
        if final.startswith('-'):
            old = final
            final = found.op_char + final[1:]
            steps.append(f'Result is negative - we add back the operator prefix 〖{found.op_char}〗: {old} -> 〖{final}〗')
        else:
            steps.append(f'Result is non-negative, no prefix needed: 〖{final}〗')
    return (final, steps)


# ===== Extracted from notebook cell 67 =====
def build_numeric_symbol_highscore_reasoning(examples, query):
    """
    고득점자 GitHub reasoners/equation_numeric.py의 reasoning_equation_numeric()
    로직을 현재 notebook 구조에 맞게 이식한 함수.

    반환:
        solution, answer, rule_name
    """
    lines = []
    lines.append('We need to infer the transformation rule from the examples.')
    lines.append('I will put my final answer inside \\boxed{}.')
    lines.append('')
    lines.append('Examples:')
    parsed = []
    for expr, expected in examples:
        m = _NUMERIC_EXPR_RE.fullmatch(str(expr))
        if not m:
            continue
        a, op, b = (m.group(1), m.group(2), m.group(3))
        parsed.append((a, op, b, str(expected)))
        lines.append(f' {expr} = {expected}')
    if len(parsed) == 0:
        return (None, None, None)
    by_op = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))
    detected_fmts = {}
    transformed_groups = {}
    has_symbol_suffix = False
    has_symbol_prefix = False
    symbol_suffix_char = ''
    symbol_prefix_char = ''
    for op_char, group in by_op.items():
        any_neg_suffixed = op_char != '-' and any((out.endswith('-') and len(out) > 1 for _, _, out in group))
        any_neg_prefixed = op_char != '-' and any((out.startswith('-') and len(out) > 1 for _, _, out in group))
        any_suffixed = any((out.endswith(op_char) and len(out) > 1 for _, _, out in group))
        any_prefixed = any((out.startswith(op_char) and len(out) > 1 for _, _, out in group))
        fmt = 'num'
        transformed = list(group)
        if any_neg_suffixed:
            fmt = 'neg_suffix'
            transformed = [(a, b, '-' + out[:-1] if out.endswith('-') and len(out) > 1 else out) for a, b, out in group]
        elif any_neg_prefixed:
            fmt = 'neg_prefix'
        elif any_suffixed:
            fmt = 'neg_suffix'
            has_symbol_suffix = True
            symbol_suffix_char = op_char
            transformed = [(a, b, '-' + out[:-len(op_char)] if out.endswith(op_char) and len(out) > 1 else out) for a, b, out in group]
        elif any_prefixed:
            fmt = 'neg_prefix'
            has_symbol_prefix = True
            symbol_prefix_char = op_char
            transformed = [(a, b, '-' + out[len(op_char):] if out.startswith(op_char) and len(out) > 1 else out) for a, b, out in group]
        detected_fmts[op_char] = fmt
        transformed_groups[op_char] = transformed
    transformed_map = {}
    for oc, tgroup in transformed_groups.items():
        for a, b, tout in tgroup:
            transformed_map[a, oc, b] = tout
    all_inputs = []
    for a, _, b, _ in parsed:
        all_inputs.append(a)
        all_inputs.append(b)
    lines.append('')
    lines.append(f"The inputs are {', '.join(all_inputs)}")
    all_outputs = [out for _, _, _, out in parsed]
    lines.append('')
    lines.append(f"The outputs are {', '.join(all_outputs)}")
    if has_symbol_suffix:
        lines.append(f'Some outputs have the operator symbol as suffix 〖{symbol_suffix_char}〗.')
    if has_symbol_prefix:
        lines.append(f'Some outputs have the operator symbol as prefix 〖{symbol_prefix_char}〗.')
    if not has_symbol_suffix and (not has_symbol_prefix):
        lines.append('No outputs have a symbol prefix or suffix.')
    any_transformed = any((fmt != 'num' for fmt in detected_fmts.values()))
    if any_transformed:
        t_all = [transformed_map.get((a, op, b), out) for a, op, b, out in parsed]
        lines.append(f"We now consider the outputs to be {', '.join(t_all)}")
        if has_symbol_suffix:
            lines.append('We will add back the operator suffix if our answer is negative.')
        elif has_symbol_prefix:
            lines.append('We will add back the operator prefix if our answer is negative.')
    lines.append('')
    lines.append('Looking at the input of the examples')
    for a, op, b, out in parsed:
        lines.append(f'{a}{op}{b} -> {op}')
    op_names = list(by_op.keys())
    lines.append('')
    lines.append('The operators')
    for op in op_names:
        lines.append(op)
    q_match = _NUMERIC_EXPR_RE.fullmatch(str(query))
    q_op = q_match.group(2) if q_match else None
    lines.append('')
    lines.append('Looking at the question')
    if q_match:
        lines.append(f'{query} -> {q_op}')
    effective_q_op = q_op
    if q_op is not None and q_op not in by_op and by_op:
        most_common_op = max(by_op, key=lambda op: len(by_op[op]))
        lines.append(f'The question operator is not found in the examples. Investigating the most common example operator 〖{most_common_op}〗 instead. We will use absolute difference for the question operator.')
        effective_q_op = most_common_op
    elif q_op is not None and q_op in by_op:
        lines.append('The question operator is found in the examples.')
    found_ops = {}
    for op_char, group in sorted(by_op.items()):
        if effective_q_op is not None and op_char != effective_q_op and (len(by_op) > 1):
            continue
        detected_fmt = detected_fmts[op_char]
        group = transformed_groups[op_char]
        examples_str = ', '.join((f'{a}{op_char}{b} = {out}' for a, b, out in group))
        lines.append('')
        lines.append(f'Looking at operator 〖{op_char}〗 [{examples_str}]:')
        found = None
        candidate_sets = [('common', numeric_common_candidates), ('rare', numeric_rare_candidates)]
        n_ex = len(group)
        for set_name, cand_fn in candidate_sets:
            for rev_ops, rev_res in ((True, True), (False, False), (True, False), (False, True)):
                cycled = list(group)
                label = f'{set_name} operations'
                if rev_ops:
                    rev_parts = ', '.join((f'{ax}->{ax[::-1]} {bx}->{bx[::-1]}' for ax, bx, _ in cycled))
                    if rev_res:
                        label += f' reversed operands [{rev_parts}] and reversed result'
                    else:
                        label += f' reversed operands [{rev_parts}]'
                elif rev_res:
                    id_parts = ', '.join((f'{ax} {bx}' for ax, bx, _ in cycled))
                    label += f' identity operands [{id_parts}] reversed result'
                else:
                    id_parts = ', '.join((f'{ax} {bx}' for ax, bx, _ in cycled))
                    label += f' on identity [{id_parts}]'
                if rev_ops:
                    all_expected = ', '.join((f'({ax[::-1]},{bx[::-1]})->{exp}' for ax, bx, exp in cycled))
                else:
                    all_expected = ', '.join((f'({ax},{bx})->{exp}' for ax, bx, exp in cycled))
                lines.append(f' Trying {label} [expected {all_expected}]:')

                def _fmt_result(raw, a, b, detail, arrow):
                    fin = numeric_rev(raw) if rev_res else raw
                    val = f'{raw} -rev-> {fin}' if rev_res else fin
                    if arrow:
                        return f'f({a},{b}) ->{detail} {val}'
                    return f'f({a}, {b}) ={detail} {val}'
                ca_str, cb_str = (cycled[0][0], cycled[0][1])
                cta = ca_str[::-1] if rev_ops else ca_str
                ctb = cb_str[::-1] if rev_ops else cb_str
                try:
                    candidates = cand_fn(int(cta), int(ctb), cta, ctb)
                except Exception:
                    continue
                cand_idx = 0
                for cand_name, cand_res in candidates:
                    rotated = [cycled[(cand_idx + j) % n_ex] for j in range(n_ex)]
                    cand_idx += 1
                    parts = []
                    all_pass = True
                    for i, (ax, bx, exp_x) in enumerate(rotated):
                        rax = ax[::-1] if rev_ops else ax
                        rbx = bx[::-1] if rev_ops else bx
                        try:
                            raw = next((r for n, r in numeric_all_candidates(int(rax), int(rbx), rax, rbx) if n == cand_name))
                        except Exception:
                            all_pass = False
                            break
                        expr_x = numeric_expr_text(cand_name, rax, rbx)
                        inter_x = numeric_expr_intermediate(cand_name, rax, rbx)
                        if expr_x and inter_x:
                            detail_x = f' {expr_x} = {inter_x} ='
                        elif expr_x:
                            detail_x = f' {expr_x} ='
                        else:
                            detail_x = ''
                        fin = numeric_rev(raw) if rev_res else raw
                        status = 'match' if fin == exp_x else 'wrong'
                        if fin != exp_x:
                            all_pass = False
                        parts.append(_fmt_result(raw, rax, rbx, detail_x, arrow=i > 0) + f' {status}')
                        if fin != exp_x:
                            break
                    if all_pass:
                        if found:
                            parts.append('correct, but skipping')
                        else:
                            summary = []
                            if rev_ops:
                                summary.append('reversed operands')
                            if rev_res:
                                summary.append('reversed result')
                            summary.append(cand_name)
                            parts.append('correct, actions: ' + ', '.join(summary))
                    lines.append(f' {cand_name} ' + ', '.join(parts))
                    if not all_pass:
                        continue
                    if not found:
                        found = NumericFoundOp(op_name=cand_name, rev_ops=rev_ops, rev_res=rev_res, fmt=detected_fmt, op_char=op_char)
        if found:
            found_ops[op_char] = found
        else:
            if op_char == effective_q_op:
                return (None, None, None)
            lines.append(' No matching operation found.')
    if not q_match or effective_q_op not in found_ops:
        return (None, None, None)
    qa, qb = (q_match.group(1), q_match.group(3))
    lines.append('')
    lines.append(f'Applying to {query}:')
    if effective_q_op != q_op:
        lines.append(' We recall that the question operator is not found in the examples. We will use the absolute difference as the operator.')
        abs_diff_op = NumericFoundOp(op_name='absolute difference', rev_ops=False, rev_res=False, fmt=found_ops[effective_q_op].fmt, op_char=q_op or '')
        result_val, steps = numeric_apply_found_op(abs_diff_op, qa, qb)
    else:
        result_val, steps = numeric_apply_found_op(found_ops[effective_q_op], qa, qb)
    for step in steps:
        lines.append(f' {step}')
    lines.append(f' Result: 〖{result_val}〗')
    lines.append('')
    lines.append('I will now return the answer in \\boxed{}')
    lines.append(f'The answer in \\boxed{{–}} is \\boxed{{{result_val}}}')
    rule_name = f'highscore_equation_numeric__{effective_q_op}__{found_ops[effective_q_op].op_name}'
    return ('\n'.join(lines), result_val, rule_name)

def solve_numeric_symbol(examples, query, true_answer=None):
    """
    Numeric Symbol Solver.

    고득점자 GitHub equation_numeric.py 재현 버전.

    로직:
        1. examples를 left/operator/right/output으로 parse
        2. operator별 output prefix/suffix format 감지
        3. question operator가 examples에 있으면 해당 operator만 분석
        4. 없으면 가장 많은 example operator를 분석한 뒤 query에는 absolute difference 적용
        5. common operations 먼저 탐색
        6. rare operations 나중 탐색
        7. operand reverse / result reverse 조합 순서:
           (True, True), (False, False), (True, False), (False, True)
        8. 첫 full-match operation을 선택
    """
    solver_name = 'numeric_symbol_highscore_equation_numeric_solver'
    if query is None or examples is None or len(examples) == 0:
        return make_solver_result(solved=False, answer=None, solution='Task: numeric_symbol\nstatus=unsolved\nreason=query_or_examples_missing', rule_name=None, solver_name=solver_name)
    try:
        solution, answer, rule_name = build_numeric_symbol_highscore_reasoning(examples=examples, query=query)
        if solution is None or answer is None:
            return make_solver_result(solved=False, answer=None, solution=f'Task: numeric_symbol\nQuery: {query}\nstatus=unsolved\nreason=no_matching_highscore_equation_numeric_rule', rule_name='highscore_equation_numeric_no_match', solver_name=solver_name)
        return make_solver_result(solved=True, answer=answer, solution=solution, rule_name=rule_name, solver_name=solver_name)
    except Exception as e:
        return make_solver_result(solved=False, answer=None, solution=f'Task: numeric_symbol\nstatus=solver_exception\nerror={type(e).__name__}: {e}', rule_name='highscore_equation_numeric_exception', solver_name=solver_name)


# ===== Extracted from notebook cell 74 =====
@dataclass
class PureSymbolExample:
    a: tuple[str, str]
    op: str
    b: tuple[str, str]
    out: str

def quote_symbol(s):
    """
    고득점자 cryptarithm.py의 quote(s)와 동일한 출력 형식.
    """
    return f'〖{s}〗'

def box_each_symbol(s):
    """
    고득점자 cryptarithm.py의 _box(s)와 동일한 출력 형식.
    문자열의 각 문자를 〖〗로 감싼다.
    """
    return ''.join((f'〖{c}〗' for c in str(s)))

def parse_pure_symbol_expr_highscore(expr):
    """
    고득점자 cryptarithm parser와 동일한 5글자 구조.

    input:
        A B operator C D

    반환:
        PureSymbolExample용 component
    """
    expr = str(expr)
    if len(expr) != 5:
        return None
    return {'a': (expr[0], expr[1]), 'op': expr[2], 'b': (expr[3], expr[4])}

def make_pure_symbol_example_highscore(input_value, output_value):
    """
    example row를 고득점자 _Ex 구조로 변환한다.
    """
    parsed = parse_pure_symbol_expr_highscore(input_value)
    if parsed is None:
        return None
    return PureSymbolExample(a=parsed['a'], op=parsed['op'], b=parsed['b'], out=str(output_value))


# ===== Extracted from notebook cell 75 =====
def pure_symbol_concat_type_highscore(exs):
    """
    고득점자 cryptarithm.py의 _concat_type(exs)와 동일.

    반환:
        'fwd' : output = A1 A2 B1 B2
        'rev' : output = B1 B2 A1 A2
        None  : 둘 다 아님
    """
    if all((ex.out == ex.a[0] + ex.a[1] + ex.b[0] + ex.b[1] for ex in exs)):
        return 'fwd'
    if all((ex.out == ex.b[0] + ex.b[1] + ex.a[0] + ex.a[1] for ex in exs)):
        return 'rev'
    return None

def pure_symbol_apply_concat_type(q_a, q_b, concat_type):
    """
    query에 concat type 적용.
    """
    if concat_type == 'rev':
        return q_b[0] + q_b[1] + q_a[0] + q_a[1]
    return q_a[0] + q_a[1] + q_b[0] + q_b[1]


# ===== Extracted from notebook cell 76 =====
def build_pure_symbol_highscore_reasoning(original_examples, parsed_examples, query, q_a, q_op, q_b, concat_types, q_ct, answer):
    """
    고득점자 cryptarithm.py의 reasoning_cryptarithm() trace 형식을 재현한다.
    """
    lines = []
    lines.append('We need to infer the transformation rule from the examples.')
    lines.append('I will put my final answer inside \\boxed{}.')
    lines.append('')
    for original_ex, ex_parsed in zip(original_examples, parsed_examples):
        orig_inp = str(original_ex[0])
        orig_out = str(original_ex[1])
        lines.append(f'{quote_symbol(orig_inp)} = {quote_symbol(orig_out)}')
        a0 = quote_symbol(ex_parsed.a[0])
        a1 = quote_symbol(ex_parsed.a[1])
        b0 = quote_symbol(ex_parsed.b[0])
        b1 = quote_symbol(ex_parsed.b[1])
        op_q = quote_symbol(ex_parsed.op)
        out_boxed = box_each_symbol(orig_out)
        lines.append(f' input: {a0}{a1}{op_q}{b0}{b1}')
        lines.append(f' left:{a0}{a1}')
        lines.append(f' operator: {op_q}')
        lines.append(f' right:{b0}{b1}')
        lines.append(f' output: {out_boxed}')
        fwd = ex_parsed.a[0] + ex_parsed.a[1] + ex_parsed.b[0] + ex_parsed.b[1]
        rev = ex_parsed.b[0] + ex_parsed.b[1] + ex_parsed.a[0] + ex_parsed.a[1]
        is_fwd = orig_out == fwd
        is_rev = orig_out == rev
        lines.append(f" concatenation: {box_each_symbol(fwd)} {('match' if is_fwd else 'mismatch')}")
        lines.append(f" reverse concatenation: {box_each_symbol(rev)} {('match' if is_rev else 'mismatch')}")
        ct = concat_types.get(ex_parsed.op)
        if ct == 'fwd':
            op_type = 'concatenation'
        elif ct == 'rev':
            op_type = 'reverse concatenation'
        else:
            op_type = 'unknown'
        lines.append(f' operator: {quote_symbol(ex_parsed.op)}{op_type}')
        lines.append('')
    q_op_known = q_op in concat_types
    op_label = 'concatenation' if q_ct == 'fwd' else 'reverse concatenation'
    qa0 = quote_symbol(q_a[0])
    qa1 = quote_symbol(q_a[1])
    qb0 = quote_symbol(q_b[0])
    qb1 = quote_symbol(q_b[1])
    q_orig = str(query)
    lines.append(f'Question{quote_symbol(q_orig)}')
    lines.append(f' input: {qa0}{qa1}{quote_symbol(q_op)}{qb0}{qb1}')
    lines.append(f' left:{qa0}{qa1}')
    lines.append(f' operator:{quote_symbol(q_op)}')
    lines.append(f' right:{qb0}{qb1}')
    lines.append('')
    if q_op_known:
        lines.append(f'The question operator is {quote_symbol(q_op)}, which is {op_label}.')
    else:
        lines.append(f'The question operator is {quote_symbol(q_op)}, which is unknown.')
        lines.append('As the question operator is unknown, we default to concatenation.')
    lines.append('')
    lines.append(f' {op_label}({qa0}{qa1}, {qb0}{qb1}) = {box_each_symbol(answer)}')
    lines.append(f" output: {quote_symbol(answer)}-> {quote_symbol('{' + answer + '}')}")
    lines.append('')
    lines.append('I will now return the answer in \\boxed{}')
    lines.append(f'The answer in \\boxed{{–}} is \\boxed{{{answer}}}')
    return '\n'.join(lines)


# ===== Extracted from notebook cell 77 =====
def solve_pure_symbol(examples, query):
    """
    Pure Symbol solver.

    고득점자 GitHub cryptarithm.py 재현 버전.

    로직:
        1. 모든 input은 5글자 A B operator C D 구조여야 한다.
        2. 각 operator별 examples를 모은다.
        3. operator별 output이 left+right인지 right+left인지 확인한다.
        4. query operator가 examples에 있으면 해당 operator의 concat type 사용.
        5. query operator가 없거나 concat type을 못 찾으면 forward concat 기본값 사용.
        6. answer와 generated_cot 스타일 reasoning을 반환한다.
    """
    solver_name = 'pure_symbol_highscore_cryptarithm_solver'
    rule_name = 'highscore_cryptarithm_concat'
    if query is None or examples is None or len(examples) == 0:
        return make_solver_result(solved=False, answer=None, solution='Task: pure_symbol\nstatus=unsolved\nreason=query_or_examples_missing', rule_name=rule_name, solver_name=solver_name)
    try:
        parsed_examples = []
        for ex_input, ex_output in examples:
            parsed_ex = make_pure_symbol_example_highscore(input_value=ex_input, output_value=ex_output)
            if parsed_ex is None:
                return make_solver_result(solved=False, answer=None, solution='Task: pure_symbol\nstatus=unsolved\nreason=example_parse_failed_or_not_length_5', rule_name=rule_name, solver_name=solver_name)
            parsed_examples.append(parsed_ex)
        q = str(query)
        if len(q) != 5:
            return make_solver_result(solved=False, answer=None, solution='Task: pure_symbol\nstatus=unsolved\nreason=query_parse_failed_or_not_length_5', rule_name=rule_name, solver_name=solver_name)
        q_a = (q[0], q[1])
        q_op = q[2]
        q_b = (q[3], q[4])
        by_op = {}
        for parsed_ex in parsed_examples:
            by_op.setdefault(parsed_ex.op, []).append(parsed_ex)
        concat_types = {}
        for op, op_exs in by_op.items():
            ct = pure_symbol_concat_type_highscore(op_exs)
            if ct is not None:
                concat_types[op] = ct
        if q_op in by_op:
            q_ct = pure_symbol_concat_type_highscore(by_op[q_op])
            if q_ct is None:
                q_ct = 'fwd'
        else:
            q_ct = 'fwd'
        answer = pure_symbol_apply_concat_type(q_a=q_a, q_b=q_b, concat_type=q_ct)
        solution = build_pure_symbol_highscore_reasoning(original_examples=examples, parsed_examples=parsed_examples, query=query, q_a=q_a, q_op=q_op, q_b=q_b, concat_types=concat_types, q_ct=q_ct, answer=answer)
        return make_solver_result(solved=True, answer=answer, solution=solution, rule_name=f'{rule_name}_{q_ct}', solver_name=solver_name)
    except Exception as e:
        return make_solver_result(solved=False, answer=None, solution=f'Task: pure_symbol\nstatus=solver_exception\nerror={type(e).__name__}: {e}', rule_name=rule_name, solver_name=solver_name)


# ===== Extracted from notebook cell 88 =====
from typing import Dict, List, Literal, Optional, Sequence, Tuple

N_BITS = 8

SYM_FAMILIES = ('XOR', 'OR', 'AND')

ASYM_FAMILIES = ('AND-NOT', 'XOR-NOT', 'OR-NOT')

PAIR_FAMILIES = SYM_FAMILIES + ASYM_FAMILIES

UNARY_FAMILIES = ('I', 'NOT')

CONSTANT_FAMILIES = ('0', '1')

DEFAULT_FAMILY = 'DEFAULT'

SECTION_ORDER = ('Identity', 'NOT', 'Constant', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT')

_SECTION_TO_FAMILIES = {'Identity': ('I',), 'NOT': ('NOT',), 'Constant': ('0', '1')}

_FAMILY_TO_SECTION = {}

RuleFamily = Literal['I', 'NOT', '0', '1', 'XOR', 'OR', 'AND', 'AND-NOT', 'XOR-NOT', 'OR-NOT', 'DEFAULT']

@dataclass(frozen=True)
class RuleCandidate:
    family: RuleFamily
    primary: Optional[int]
    secondary: Optional[int]
    expr: str
    primary_stride: Optional[int] = None
    secondary_stride: Optional[int] = None
    primary_offset: Optional[int] = None
    secondary_offset: Optional[int] = None

    @property
    def is_default(self):
        return self.family == DEFAULT_FAMILY

@dataclass(frozen=True)
class Record:
    label: str
    col: str
    hash_: str
    matches: Tuple[int, ...]

def safe_parse_examples_if_string(examples):
    """
    examples가 이미 list면 그대로 사용하고,
    CSV 로드 등으로 문자열화되어 있으면 ast.literal_eval로 복원한다.
    """
    if isinstance(examples, str):
        try:
            return ast.literal_eval(examples)
        except Exception:
            return examples
    return examples

def _normalize_bits(value):
    """
    고득점자 repo의 _normalize_bits와 동일.
    문자열에서 0/1만 추출하고, 길이가 8이 아니면 빈 문자열 반환.
    """
    bits = ''.join((ch for ch in str(value) if ch in {'0', '1'}))
    if len(bits) != N_BITS:
        return ''
    return bits

def normalize_bit_examples_highscore(examples):
    """
    examples를 [(input_bits, output_bits), ...]로 정규화.
    """
    examples = safe_parse_examples_if_string(examples)
    if examples is None:
        return []
    normalized = []
    for ex in examples:
        if isinstance(ex, dict):
            x = ex.get('input')
            y = ex.get('output')
        else:
            x, y = ex
        x = _normalize_bits(x)
        y = _normalize_bits(y)
        if not x or not y:
            return []
        normalized.append((x, y))
    return normalized

def _column_bits(values, bit):
    return ''.join((v[bit] for v in values))

def _bit_not(bit):
    return '1' if bit == '0' else '0'

def _invert(bits):
    return ''.join((_bit_not(b) for b in bits))

def _column_hash(bits, total_examples):
    """
    고득점자 repo의 column hash.
    ones가 0 또는 전체 개수이면 'a', 아니면 hex count.
    """
    ones = bits.count('1')
    if ones == 0 or ones == total_examples:
        return 'a'
    return format(ones, 'x')

def _evaluate_binary(a, b, family):
    if family in ('AND', 'AND-NOT'):
        return '1' if a == '1' and b == '1' else '0'
    if family in ('OR', 'OR-NOT'):
        return '1' if a == '1' or b == '1' else '0'
    if family in ('XOR', 'XOR-NOT'):
        return '1' if a != b else '0'
    raise ValueError(f'Unsupported family {family}')

def _apply_family(a_bits, b_bits, family, invert_second=False):
    b_eff = _invert(b_bits) if invert_second else b_bits
    out = []
    for x, y in zip(a_bits, b_eff):
        out.append(_evaluate_binary(x, y, family))
    return ''.join(out)

def _find_match(candidates, fam, ep, es):
    """
    family / primary / secondary가 정확히 맞는 candidate 찾기.
    """
    for c in candidates:
        if c.family != fam:
            continue
        if c.primary == ep and (fam not in PAIR_FAMILIES or c.secondary == es):
            return c
    return None

def _exists_anywhere(all_matches, fam, ep, es):
    """
    같은 operand pair가 다른 output bit 위치에라도 존재하는지 확인.
    """
    for bit_cands in all_matches:
        if _find_match(bit_cands, fam, ep, es) is not None:
            return True
    return False

def _fail_suffix(all_matches, fam, ep, es):
    """
    y: wrong position
    x: not in operator
    """
    if _exists_anywhere(all_matches, fam, ep, es):
        return 'y'
    return 'x'

def make_bit_solver_result(solved, answer, solution, rule_name, solver_name):
    """
    기존 notebook의 make_solver_result와 호환.
    """
    if 'make_solver_result' in globals():
        return make_solver_result(solved=solved, answer=answer, solution=solution, rule_name=rule_name, solver_name=solver_name)
    return {'solved': solved, 'answer': answer, 'solution': solution, 'rule_name': rule_name, 'solver_name': solver_name}


# ===== Extracted from notebook cell 89 =====
def _compact_rule(c):
    """
    Compact display.
    Pair rule: 34
    Unary rule: 3
    Constant/default: expr
    """
    if c.primary is not None and c.secondary is not None:
        return f'{c.primary}{c.secondary}'
    if c.primary is not None:
        return str(c.primary)
    return c.family

def _format_list(cands, with_count=False, failed=None):
    if not cands:
        return 'none'
    if with_count:
        parts = []
        for i, c in enumerate(cands):
            if i == 0:
                parts.append(c.expr)
            else:
                parts.append(_compact_rule(c))
        return ' '.join(parts) + f': {len(cands)}'
    parts = [_compact_rule(c) for c in cands]
    if failed:
        parts.append(failed)
    return ' '.join(parts)

def _find_all_left_runs(all_matches):
    """
    output bit 0에서 시작하는 stride-consistent run.
    고득점자 repo는 stride (1,1)만 사용.
    """
    if not all_matches or not all_matches[0]:
        return []
    runs = []
    for start_cand in all_matches[0]:
        fam = start_cand.family
        strides = [(1, 1)]
        for p_step, s_step in strides:
            chain = [start_cand]
            cur_p = start_cand.primary
            cur_s = start_cand.secondary
            failed_next = None
            for b in range(1, len(all_matches)):
                ep = (cur_p + p_step) % N_BITS if cur_p is not None else None
                es = (cur_s + s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, ep, es)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, ep, es)
                    if ep is not None and es is not None:
                        failed_next = f'{ep}{es}{suffix}'
                    elif ep is not None:
                        failed_next = f'{ep}{suffix}'
                    break
                chain.append(found)
                cur_p, cur_s = (ep, es)
            runs.append((chain, failed_next))
    return runs

def _find_all_right_runs(all_matches):
    """
    output bit 7에서 끝나는 stride-consistent run.
    고득점자 repo는 stride (1,1)만 사용.
    """
    n = len(all_matches)
    if not all_matches or not all_matches[-1]:
        return []
    runs = []
    for end_cand in all_matches[-1]:
        fam = end_cand.family
        strides = [(1, 1)]
        for p_step, s_step in strides:
            chain = [end_cand]
            cur_p = end_cand.primary
            cur_s = end_cand.secondary
            failed_next = None
            for k in range(1, n):
                b = n - 1 - k
                pp = (cur_p - p_step) % N_BITS if cur_p is not None else None
                ps = (cur_s - s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, pp, ps)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, pp, ps)
                    if pp is not None and ps is not None:
                        failed_next = f'{pp}{ps}{suffix}'
                    elif pp is not None:
                        failed_next = f'{pp}{suffix}'
                    break
                chain.insert(0, found)
                cur_p, cur_s = (pp, ps)
            runs.append((chain, failed_next))
    return runs

def _lr_from_matches(all_matches):
    """
    Left / Right run 전체와 best를 계산.
    """
    all_left_runs = _find_all_left_runs(all_matches)
    all_right_runs = _find_all_right_runs(all_matches)
    left_run = max(all_left_runs, key=lambda t: len(t[0])) if all_left_runs else ([], None)
    right_run = max(all_right_runs, key=lambda t: len(t[0])) if all_right_runs else ([], None)
    left_lines = [_format_list(chain, failed=failed) for chain, failed in all_left_runs] if all_left_runs else ['none']
    left_best = _format_list(left_run[0], with_count=True)
    right_lines = [_format_list(list(reversed(chain)), failed=failed) for chain, failed in all_right_runs] if all_right_runs else ['none']
    right_best = _format_list(list(reversed(right_run[0])), with_count=True)
    return (left_lines, left_best, right_lines, right_best)

def _evaluate_rule(bits, rule):
    if rule.family == 'DEFAULT':
        return '1'
    if rule.family == '0':
        return '0'
    if rule.family == '1':
        return '1'
    if rule.family == 'I':
        assert rule.primary is not None
        return bits[rule.primary]
    if rule.family == 'NOT':
        assert rule.primary is not None
        return _bit_not(bits[rule.primary])
    if rule.family in PAIR_FAMILIES:
        assert rule.primary is not None
        assert rule.secondary is not None
        a = bits[rule.primary]
        b = bits[rule.secondary]
        if '-NOT' in rule.family:
            b = _bit_not(b)
        return _evaluate_binary(a, b, rule.family)
    raise ValueError(f'Unknown family {rule.family}')

def _emit_apply(lines, question_bits, vector):
    """
    고득점자 repo의 _emit_apply와 동일한 출력 형식.
    마지막 boxed answer까지 lines에 추가하고 answer를 반환.
    """
    lines.append(f'Applying to {question_bits}')
    lines.append('Input')
    for i, bit in enumerate(question_bits):
        lines.append(f'{i} {bit}')
    lines.append('Output')
    answer_bits = []
    for i, rule in enumerate(vector):
        if rule.family == 'DEFAULT':
            lines.append(f'{i} default 1 = 1')
            answer_bits.append('1')
            continue
        if rule.family in CONSTANT_FAMILIES:
            lines.append(f'{i} {rule.expr} = {rule.family}')
            answer_bits.append(rule.family)
            continue
        if rule.family == 'I':
            assert rule.primary is not None
            val = question_bits[rule.primary]
            lines.append(f'{i} {rule.expr} = {val}')
            answer_bits.append(val)
            continue
        if rule.family == 'NOT':
            assert rule.primary is not None
            val = question_bits[rule.primary]
            nval = _bit_not(val)
            lines.append(f'{i} {rule.expr} = NOT({val}) = {nval}')
            answer_bits.append(nval)
            continue
        assert rule.primary is not None
        assert rule.secondary is not None
        a = question_bits[rule.primary]
        b = question_bits[rule.secondary]
        if rule.family in SYM_FAMILIES:
            result = _evaluate_rule(question_bits, rule)
            lines.append(f'{i} {rule.expr} = {rule.family}({a},{b}) = {result}')
            answer_bits.append(result)
            continue
        base = rule.family.split('-')[0]
        result = _evaluate_rule(question_bits, rule)
        lines.append(f'{i} {rule.expr} = {base}({a},NOT({b})) = {result}')
        answer_bits.append(result)
    answer = ''.join(answer_bits)
    lines.append('')
    lines.append('I will now return the answer in \\boxed{}')
    lines.append(f'The answer in \\boxed{{–}} is \\boxed{{{answer}}}')
    return answer


# ===== Extracted from notebook cell 90 =====
def build_bit_highscore_tables(inputs, outputs):
    """
    고득점자 repo의 all_records / all_matches 생성 로직.
    """
    n_examples = len(outputs)
    output_columns = [_column_bits(outputs, i) for i in range(N_BITS)]
    input_columns = [_column_bits(inputs, i) for i in range(N_BITS)]
    input_inverted = [_invert(col) for col in input_columns]
    all_records = {name: [] for name in SECTION_ORDER}
    all_matches = {name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER}
    for out_idx, out_col in enumerate(output_columns):
        for i_col, in_col in enumerate(input_columns):
            if in_col == out_col:
                all_matches['Identity'][out_idx].append(RuleCandidate('I', i_col, None, f'I{i_col}'))
            if input_inverted[i_col] == out_col:
                all_matches['NOT'][out_idx].append(RuleCandidate('NOT', i_col, None, f'NOT{i_col}'))
        if out_col.count('1') == 0:
            all_matches['Constant'][out_idx].append(RuleCandidate('0', None, None, 'C0'))
        if out_col.count('1') == n_examples:
            all_matches['Constant'][out_idx].append(RuleCandidate('1', None, None, 'C1'))
    for label, col in zip([str(i) for i in range(N_BITS)], input_columns):
        matches = tuple((i for i, oc in enumerate(output_columns) if col == oc))
        all_records['Identity'].append(Record(label=label, col=col, hash_=_column_hash(col, n_examples), matches=matches))
    for label, col in zip([str(i) for i in range(N_BITS)], input_inverted):
        matches = tuple((i for i, oc in enumerate(output_columns) if col == oc))
        all_records['NOT'].append(Record(label=label, col=col, hash_=_column_hash(col, n_examples), matches=matches))
    for val in ('0', '1'):
        col = val * n_examples
        matches = tuple((i for i, oc in enumerate(output_columns) if col == oc))
        all_records['Constant'].append(Record(label=val, col=col, hash_=_column_hash(col, n_examples), matches=matches))
    for fam in ('XOR', 'OR', 'AND'):
        for circ_diff in range(1, N_BITS // 2 + 1):
            n_pairs = N_BITS // 2 if circ_diff == N_BITS // 2 else N_BITS
            for a in range(n_pairs):
                b = (a + circ_diff) % N_BITS
                lo, hi = (min(a, b), max(a, b))
                col = _apply_family(input_columns[lo], input_columns[hi], fam)
                matches = tuple((i for i, out_col in enumerate(output_columns) if col == out_col))
                all_records[fam].append(Record(label=f'{a}{b} {b}{a}', col=col, hash_=_column_hash(col, n_examples), matches=matches))
                for out_idx in matches:
                    all_matches[fam][out_idx].append(RuleCandidate(fam, a, b, f'{fam}{a}{b}'))
                    all_matches[fam][out_idx].append(RuleCandidate(fam, b, a, f'{fam}{b}{a}'))
    for fam in ('AND-NOT', 'XOR-NOT', 'OR-NOT'):
        for diff in range(1, N_BITS):
            for a in range(N_BITS):
                b = (a + diff) % N_BITS
                col = _apply_family(input_columns[a], input_columns[b], fam, invert_second=True)
                matches = tuple((i for i, out_col in enumerate(output_columns) if col == out_col))
                all_records[fam].append(Record(label=f'{a}{b}', col=col, hash_=_column_hash(col, n_examples), matches=matches))
                for out_idx in matches:
                    all_matches[fam][out_idx].append(RuleCandidate(fam, a, b, f'{fam}{a}{b}'))
    for name in ('Identity', 'NOT', 'Constant'):
        all_records[name].sort(key=lambda r: r.label)
    return {'n_examples': n_examples, 'output_columns': output_columns, 'input_columns': input_columns, 'input_inverted': input_inverted, 'all_records': all_records, 'all_matches': all_matches}


# ===== Extracted from notebook cell 91 =====
def build_bit_highscore_reasoning(inputs, outputs, question_bits):
    """
    고득점자 GitHub reasoners/bit_manipulation.py의 reasoning_bit_manipulation()
    로직을 현재 notebook 구조에 맞게 이식한 함수.

    반환:
        solution, answer, rule_name
    """
    table = build_bit_highscore_tables(inputs=inputs, outputs=outputs)
    n_examples = table['n_examples']
    output_columns = table['output_columns']
    input_columns = table['input_columns']
    all_records = table['all_records']
    all_matches = table['all_matches']
    lines = []
    lines.append('We need to deduce the transformation by matching the example outputs.')
    lines.append('I will put my final answer inside \\boxed{}.')
    lines.append('')
    for i, out in enumerate(outputs):
        lines.append(f'Output {i}: {out}')
        for bit in range(N_BITS):
            lines.append(f'{bit} {out[bit]}')
        lines.append('')
    lines.append('Output bit columns (with bitsum as hash)')
    for bit in range(N_BITS):
        lines.append(f'{bit} {output_columns[bit]} {_column_hash(output_columns[bit], n_examples)}')
    lines.append('')
    for i, inp in enumerate(inputs):
        lines.append(f'Input {i}: {inp}')
        for bit in range(N_BITS):
            lines.append(f'{bit} {inp[bit]}')
        lines.append('')
    lines.append('When matching output')
    lines.append('x: not in operator')
    lines.append('y: wrong position')
    lines.append('')
    section_lefts = []
    section_rights = []

    def _add_section(name):
        records = all_records[name]
        per_bit = all_matches[name]
        lines.append(name)
        prev_diff = None
        for rec in records:
            if len(rec.label) >= 2 and rec.label[0].isdigit() and rec.label[1].isdigit():
                diff = (int(rec.label[1]) - int(rec.label[0])) % N_BITS
                if prev_diff is not None and diff != prev_diff:
                    lines.append('')
                prev_diff = diff
            line = f'{rec.label} {rec.col} {rec.hash_}'
            if rec.matches:
                line += ' match ' + ' '.join((str(i) for i in rec.matches))
            lines.append(line)
        lines.append('')
        lines.append('Matching output')
        for i in range(N_BITS):
            cands = per_bit[i]
            if cands:

                def _compact(c):
                    if c.primary is not None and c.secondary is not None:
                        return f'{c.primary}{c.secondary}'
                    if c.primary is not None:
                        return str(c.primary)
                    return c.expr
                lines.append(f'{i} ' + ' '.join((_compact(c) for c in cands)))
            else:
                lines.append(f'{i} absent')
        lines.append('')
        left_lines, left_best, right_lines, right_best = _lr_from_matches(per_bit)
        section_lefts.append((name, left_best))
        section_rights.append((name, right_best))
        lines.append('Left')
        for ll in left_lines:
            lines.append(ll)
        lines.append(f'Best: {left_best}')
        lines.append('')
        lines.append('Right')
        for rl in right_lines:
            lines.append(rl)
        lines.append(f'Best: {right_best}')
        lines.append('')
    for name in all_records:
        _add_section(name)
    lines.append('Selecting')
    lines.append('')

    def _parse_count(val):
        if val == 'none':
            return 0
        try:
            return int(val.rsplit(': ', 1)[-1])
        except ValueError:
            return 0

    def _pick_winner(entries):
        best_name = None
        best_text = 'none'
        best_count = 0
        for name, val in entries:
            count = _parse_count(val)
            if count > best_count:
                best_count = count
                best_name = name
                best_text = val
        return (best_name, best_text, best_count)
    left_winner_name, left_winner_text, left_winner_count = _pick_winner(section_lefts)
    right_winner_name, right_winner_text, right_winner_count = _pick_winner(section_rights)

    def _get_section_run(winner_name, direction):
        if winner_name is None:
            return []
        per_bit = all_matches[winner_name]
        if direction == 'left':
            runs = _find_all_left_runs(per_bit)
        else:
            runs = _find_all_right_runs(per_bit)
        if not runs:
            return []
        best_chain, _ = max(runs, key=lambda t: len(t[0]))
        return best_chain
    left_run = _get_section_run(left_winner_name, 'left')
    right_run = _get_section_run(right_winner_name, 'right')
    lines.append('Lefts')
    for name, lb in section_lefts:
        lines.append(f'{name} {lb}')
    lines.append('')
    lines.append('Rights')
    for name, rb in section_rights:
        lines.append(f'{name} {rb}')
    lines.append('')
    lines.append(f'Left longest: {left_winner_count}')
    lines.append(f'Right longest: {right_winner_count}')
    lines.append('')

    def _matching_line(label, winner_name, entries):
        parts = []
        for name, _val in entries:
            parts.append(f"{name} {('yes' if name == winner_name else 'no')}")
        return f"{label} winner: {', '.join(parts)}"
    if right_winner_count > left_winner_count:
        lines.append(_matching_line('Right', right_winner_name, section_rights))
        lines.append(_matching_line('Left', left_winner_name, section_lefts))
        lines.append('')
        lines.append(f'Best right: {right_winner_text}')
        lines.append(f'Best left: {left_winner_text}')
    else:
        lines.append(_matching_line('Left', left_winner_name, section_lefts))
        lines.append(_matching_line('Right', right_winner_name, section_rights))
        lines.append('')
        lines.append(f'Best left: {left_winner_text}')
        lines.append(f'Best right: {right_winner_text}')
    lines.append('')
    left_len_final = left_winner_count
    right_len_final = right_winner_count
    if left_len_final + right_len_final > N_BITS:
        if right_len_final > left_len_final:
            left_len_final = N_BITS - right_len_final
            left_run = left_run[:left_len_final]
        else:
            right_len_final = N_BITS - left_len_final
            right_run = right_run[-right_len_final:] if right_len_final else []
    left_was_truncated = left_len_final < left_winner_count
    right_was_truncated = right_len_final < right_winner_count
    trunc_left = f'Truncated left: {_format_list(left_run, with_count=True)}'
    if left_was_truncated:
        trunc_left += ' truncated'
    trunc_right = f'Truncated right: {_format_list(list(reversed(right_run)), with_count=True)}'
    if right_was_truncated:
        trunc_right += ' truncated'
    if right_winner_count > left_winner_count:
        lines.append(trunc_right)
        lines.append(trunc_left)
    else:
        lines.append(trunc_left)
        lines.append(trunc_right)
    lines.append('')
    right_start_final = N_BITS - right_len_final
    lines.append('Tentative from right')
    for i in range(N_BITS - 1, -1, -1):
        if i >= right_start_final and right_run:
            lines.append(f'{i} {right_run[i - right_start_final].expr}')
        else:
            lines.append(f'{i} pending')
    lines.append('')
    lines.append('Tentative')
    for i in range(N_BITS):
        if i < left_len_final:
            lines.append(f'{i} {left_run[i].expr}')
        elif i >= right_start_final and right_run:
            lines.append(f'{i} {right_run[i - right_start_final].expr}')
        else:
            lines.append(f'{i} pending')
    lines.append('')

    def _extrap_from(run, bit, run_start_bit, side='left'):
        if not run:
            return None
        r = run[0]
        p = r.primary
        s = r.secondary
        if p is not None:
            p_off = (p - run_start_bit) % N_BITS
            ep = (p_off + bit) % N_BITS
        else:
            ep = None
        if s is not None:
            s_off = (s - run_start_bit) % N_BITS
            es = (s_off + bit) % N_BITS
        else:
            es = None
        if ep is not None and es is not None:
            return f'?{ep}{es}'
        if ep is not None:
            if side == 'left':
                return f'?{ep}?'
            else:
                return f'??{ep}'
        return None
    left_fam = left_run[0].family if left_run else None
    right_fam = right_run[0].family if right_run else None
    left_is_binary = left_fam in PAIR_FAMILIES if left_fam else False
    right_is_binary = right_fam in PAIR_FAMILIES if right_fam else False
    left_is_unary = left_fam in UNARY_FAMILIES if left_fam else False
    right_is_unary = right_fam in UNARY_FAMILIES if right_fam else False
    if right_winner_count > left_winner_count:
        preferred = []
        for i in range(N_BITS):
            if i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif i < left_len_final:
                preferred.append(left_run[i].expr)
            elif right_is_binary or right_is_unary:
                preferred.append(_extrap_from(right_run, i, right_start_final, 'right') or 'pending')
            else:
                preferred.append('pending')
        lines.append('Preferred from right')
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f'{i} {preferred[i]}')
        lines.append('')
        for i in range(N_BITS):
            if preferred[i] == 'pending':
                if left_is_binary or left_is_unary:
                    preferred[i] = _extrap_from(left_run, i, 0, 'left') or '?'
                else:
                    preferred[i] = '?'
            elif '?' in preferred[i][1:] and left_is_unary:
                el = _extrap_from(left_run, i, 0, 'left')
                if el:
                    merged = list(preferred[i])
                    el_chars = list(el)
                    for j in range(1, min(len(merged), len(el_chars))):
                        if merged[j] == '?' and el_chars[j] != '?':
                            merged[j] = el_chars[j]
                    preferred[i] = ''.join(merged)
        lines.append('Preferred from left')
        for i in range(N_BITS):
            lines.append(f'{i} {preferred[i]}')
        lines.append('')
    else:
        preferred = []
        for i in range(N_BITS):
            if i < left_len_final:
                preferred.append(left_run[i].expr)
            elif i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif left_is_binary or left_is_unary:
                preferred.append(_extrap_from(left_run, i, 0, 'left') or 'pending')
            else:
                preferred.append('pending')
        lines.append('Preferred from left')
        for i in range(N_BITS):
            lines.append(f'{i} {preferred[i]}')
        lines.append('')
        for i in range(N_BITS):
            if preferred[i] == 'pending':
                if right_is_binary or right_is_unary:
                    preferred[i] = _extrap_from(right_run, i, right_start_final, 'right') or '?'
                else:
                    preferred[i] = '?'
            elif '?' in preferred[i][1:] and right_is_unary:
                er = _extrap_from(right_run, i, right_start_final, 'right')
                if er:
                    merged = list(preferred[i])
                    er_chars = list(er)
                    for j in range(1, min(len(merged), len(er_chars))):
                        if merged[j] == '?' and er_chars[j] != '?':
                            merged[j] = er_chars[j]
                    preferred[i] = ''.join(merged)
        lines.append('Preferred from right')
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f'{i} {preferred[i]}')
        lines.append('')
    lines.append('Preferred')
    for i, pref in enumerate(preferred):
        if pref.startswith('?') and len(pref) == 3 and (pref[1] != '?') and (pref[2] != '?'):
            lines.append(f'{i} {pref} ?{pref[2]}{pref[1]}')
        else:
            lines.append(f'{i} {pref}')
    lines.append('')
    default_cand = RuleCandidate(DEFAULT_FAMILY, None, None, 'default 1')
    best = [default_cand] * N_BITS
    for i, rc in enumerate(left_run):
        best[i] = rc
    for i, rc in enumerate(right_run):
        best[right_start_final + i] = rc
    lines.append('Matching')
    pending_indices = []
    per_bit_cat = {name: {} for name in SECTION_ORDER}
    for i in range(N_BITS):
        pref = preferred[i]
        if not pref.startswith('?') or pref == '?':
            lines.append(f'{i} {best[i].expr}')
            continue
        pending_indices.append(i)
        digits_str = pref[1:]
        pref_digits = [int(d) for d in digits_str if d != '?']
        checks = []
        for section_name in SECTION_ORDER:
            cands = all_matches[section_name][i]
            if section_name in ('Identity', 'NOT'):
                found = [c for c in cands if c.primary in pref_digits]
                if found:
                    checks.append(section_name + ' ' + ' '.join((c.expr for c in found)))
                    per_bit_cat[section_name][i] = found
                else:
                    checks.append(f'{section_name} absent')
            elif section_name == 'Constant':
                if cands:
                    checks.append('Constant ' + ' '.join((c.expr for c in cands)))
                    per_bit_cat['Constant'][i] = list(cands)
                else:
                    checks.append('Constant absent')
            else:
                found_c = None
                orderings = []
                want_p = int(pref[1]) if len(pref) > 1 and pref[1] != '?' else None
                want_s = int(pref[2]) if len(pref) > 2 and pref[2] != '?' else None
                orderings.append((want_p, want_s))
                if want_p is not None and want_s is not None and (want_p != want_s):
                    orderings.append((want_s, want_p))
                for wp, ws in orderings:
                    for c in cands:
                        if (wp is None or c.primary == wp) and (ws is None or c.secondary == ws):
                            found_c = c
                            break
                    if found_c is not None:
                        break
                if found_c is not None:
                    checks.append(found_c.expr)
                    per_bit_cat[section_name][i] = [found_c]
                else:
                    checks.append(f'{section_name} absent')
        if pref.startswith('?') and len(pref) == 3 and (pref[1] != '?') and (pref[2] != '?'):
            pref_display = f'{pref} ?{pref[2]}{pref[1]}'
        else:
            pref_display = pref
        lines.append(f"{i} {pref_display} - {', '.join(checks)}")
    lines.append('')
    lines.append('Perfect match')
    chosen_cat = None
    for cat in SECTION_ORDER:
        is_perfect = chosen_cat is None and bool(pending_indices) and all((i in per_bit_cat[cat] for i in pending_indices))
        lines.append(f"{cat} {('yes' if is_perfect else 'no')}")
        if is_perfect:
            chosen_cat = cat
    lines.append('')
    pending_set = set(pending_indices)
    lines.append('Matched')
    for i in range(N_BITS):
        if i in pending_set:
            if chosen_cat and i in per_bit_cat[chosen_cat]:
                best[i] = per_bit_cat[chosen_cat][i][0]
                lines.append(f'{i} {best[i].expr}')
            else:
                all_cands = []
                for name in SECTION_ORDER:
                    if i in per_bit_cat[name]:
                        all_cands.extend(per_bit_cat[name][i])
                if all_cands:
                    lines.append(f'{i} ' + ' '.join((c.expr for c in all_cands)))
                    best[i] = all_cands[0]
                else:
                    lines.append(f'{i} none')
                    best[i] = default_cand
        else:
            lines.append(f'{i} {best[i].expr}')
    lines.append('')
    if all((r.is_default for r in best)):
        return (None, None, None)
    lines.append('Selected')
    for i, rule in enumerate(best):
        lines.append(f'{i} {rule.expr}')
    lines.append('')
    answer = _emit_apply(lines, question_bits, best)
    rule_name = 'highscore_bit_manipulation'
    return ('\n'.join(lines), answer, rule_name)


# ===== Extracted from notebook cell 92 =====
def solve_bit_manipulation(examples, query):
    """
    Bit Manipulation Solver.

    고득점자 GitHub reasoners/bit_manipulation.py 재현 버전.

    핵심:
        - whole-byte rotation/shift/NOT 선행 탐색 없음
        - output/input bit columns 생성
        - Identity / NOT / Constant / AND / OR / XOR / AND-NOT / OR-NOT / XOR-NOT
          섹션별 match table 생성
        - Left / Right longest run 선택
        - pending position을 preferred / matching / perfect match 로직으로 채움
        - 남는 경우 default 1
    """
    solver_name = 'bit_highscore_bit_manipulation_solver'
    try:
        normalized_examples = normalize_bit_examples_highscore(examples)
        question_bits = _normalize_bits(query)
        if not normalized_examples or not question_bits:
            return make_bit_solver_result(solved=False, answer=None, solution='Task: bit_manipulation\nstatus=unsolved\nreason=invalid_examples_or_query', rule_name='highscore_bit_invalid_input', solver_name=solver_name)
        inputs = [x for x, y in normalized_examples]
        outputs = [y for x, y in normalized_examples]
        if len(outputs) != len(inputs):
            return make_bit_solver_result(solved=False, answer=None, solution='Task: bit_manipulation\nstatus=unsolved\nreason=input_output_count_mismatch', rule_name='highscore_bit_count_mismatch', solver_name=solver_name)
        solution, answer, rule_name = build_bit_highscore_reasoning(inputs=inputs, outputs=outputs, question_bits=question_bits)
        if solution is None or answer is None:
            return make_bit_solver_result(solved=False, answer=None, solution=f'Task: bit_manipulation\nQuery: {question_bits}\nstatus=unsolved\nreason=no_highscore_bit_vector_selected', rule_name='highscore_bit_no_vector', solver_name=solver_name)
        result = make_bit_solver_result(solved=True, answer=answer, solution=solution, rule_name=rule_name, solver_name=solver_name)
        result['debug'] = {'main_stage': 'highscore_bit_column_matching', 'rule_name': rule_name}
        return result
    except Exception as e:
        return make_bit_solver_result(solved=False, answer=None, solution=f'Task: bit_manipulation\nstatus=solver_exception\nerror={type(e).__name__}: {e}', rule_name='highscore_bit_exception', solver_name=solver_name)


# ===== Extracted from notebook cell 100 =====
def run_solver_by_pattern(row):
    """
    pattern 값에 따라 알맞은 solver를 실행한다.
    """
    pattern = row['pattern']
    examples = row['examples']
    query = row['query']
    answer = row['answer']
    if pattern == 'roman_numeral':
        return solve_roman_numeral(examples, query)
    elif pattern == 'unit_conversion':
        return solve_unit_conversion(examples, query)
    elif pattern == 'formula_based':
        return solve_formula_based(examples, query)
    elif pattern == 'cipher':
        return solve_cipher(examples, query)
    elif pattern == 'bit_manipulation':
        return solve_bit_manipulation(examples, query)
    elif pattern == 'numeric_symbol':
        return solve_numeric_symbol(examples, query, answer)
    elif pattern == 'pure_symbol':
        return solve_pure_symbol(examples, query)
    else:
        return make_solver_result(solved=False, solver_name='unknown_solver')


# ===== Extracted from notebook cell 107 =====
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

COMMON_SYSTEM_PREFIX = 'You analyze the pattern of the quiz provided by the user, infer the correct answer, and provide it accurately. Refer to the assistant reasoning logs, analyze them carefully, and use them to determine the final answer.'

BIT_MANIPULATION_STRATEGY = '\n\nFor bit manipulation problems, use this strategy:\n1. Check the bitsum of every input/output example.\n2. Classify the bitsum pattern as preserved, decreased, increased, or mixed.\n3. Test simple whole-byte transformations such as rotations, shifts, reverse, and NOT-after-transform.\n4. If no whole-byte rule matches all examples, build input/output bit columns by reading each bit position vertically.\n5. Search exact Boolean relations for each output column using Constant, Identity, NOT, AND, OR, XOR, AND-NOT, OR-NOT, and XOR-NOT.\n6. Prefer repeated chain patterns across adjacent output bits.\n7. Fill uncovered output positions only with relations that match the examples exactly.\n8. Apply the selected output-bit rules to the query and concatenate the output bits.'

CIPHER_STRATEGY = "\n\nFor character substitution cipher problems, use this strategy:\n1. Align each cipher example with its plain-text output while ignoring spaces.\n2. Build the cipher-to-plain character mapping from the examples.\n3. Build the reverse plain-to-cipher mapping to keep the substitution one-to-one.\n4. Decode the query using the known mapping and mark unknown characters with '?'.\n5. For unresolved words, scan vocabulary candidates.\n6. Reject candidates by length mismatch, fixed-letter mismatch, or reverse mapping conflict.\n7. Select only candidates that pass all consistency checks.\n8. Update the mapping with selected candidates, re-decode the query, and produce the final plain-text sentence."

FORMULA_BASED_STRATEGY = "\n\nFor formula-based falling-distance problems, use this strategy:\n1. Treat the falling-distance rule as d = k * t^2, where k is the hidden proportional constant.\n2. For each example, compute t^2 using explicit multiplication steps.\n3. For each example, compute k = d / t^2 and show the division calculation trace.\n4. Convert each example's k into a 3-decimal template value.\n5. Collect all k values, sort them, and select the median k.\n6. Apply the median k to the query using d = median_k * query_t^2.\n7. Show the query t^2 calculation, the multiplication by median k, and the running partial sums.\n8. Use the calculated prediction as the final output and preserve the exact generated numeric format."

UNIT_CONVERSION_STRATEGY = "\n\nFor unit conversion problems, use this strategy:\n1. Treat the conversion as a linear factor rule: output = input * factor.\n2. For each example, compute factor = output / input and show the division calculation trace.\n3. Convert each example's factor into a 3-decimal template value.\n4. Collect all factor values, sort them, and select the median factor.\n5. Apply the median factor directly to the query input.\n6. Show the query multiplication by decomposing the median factor into decimal place values.\n7. Add the partial products step by step to obtain the final prediction.\n8. Use the calculated prediction as the final output and preserve the exact generated numeric format."

ROMAN_NUMERAL_STRATEGY = '\n\nFor Roman numeral problems, use this strategy:\n1. Identify the task as Arabic-to-Roman numeral conversion.\n2. Use the largest-value-first subtraction method.\n3. Use standard Roman numeral subtractive notation such as IV, IX, XL, XC, CD, and CM.\n4. Use extended overline notation when needed, including V̅, X̅, L̅, C̅ and extended subtractive forms.\n5. Check the examples by converting each input number and comparing the predicted Roman numeral with the expected output.\n6. For each conversion, show the subtraction calculation, appended Roman symbol, remaining value, parts, and result.\n7. Apply the same conversion calculation to the query.\n8. Preserve the exact Roman numeral output format.'

NUMERIC_SYMBOL_STRATEGY = '\n\nFor numeric-symbol transformation problems, use this strategy:\n1. Parse each expression into left operand, operator symbol, and right operand.\n2. Use same-operator examples as the primary evidence for the query operator.\n3. Exclude examples with different operator symbols from rule fitting because each symbol may use a different rule.\n4. Scan candidate transformation pipelines with operand preprocessing, core operation, result postprocessing, padding, and optional symbol wrapping.\n5. For each candidate, count how many same-operator examples it matches and reject candidates that cannot be applied to the query.\n6. Select the highest-matching candidate using match ratio, match count, and rule priority as tie-breakers.\n7. Verify the selected rule on every same-operator example using debug-style calculation logs.\n8. Apply the same selected pipeline to the query and preserve the exact output formatting, including signs, leading zeros, prefixes, suffixes, and operator symbols when used.'

PURE_SYMBOL_STRATEGY = '\n\nFor pure symbolic transformation problems, use this strategy:\n1. Treat every character as a symbol and parse each input into A, B, operator, C, and D positions.\n2. Group examples by operator symbol and use same-operator examples as the primary evidence for the query operator.\n3. If same-operator evidence is insufficient, scan all examples for a global structural rule.\n4. Test simple structural candidates such as left+right, right+left, left only, right only, reverse(left+right), and position-select rules over A, B, C, and D.\n5. For each candidate, count matched examples, mismatched examples, match ratio, and rule priority.\n6. Select a rule only when it matches all required evidence under the chosen evidence scope.\n7. Verify the selected rule on the relevant examples using debug-style calculation logs.\n8. Apply the same selected rule to the query positions and preserve the exact symbol output format.'

PATTERN_SYSTEM_PROMPTS = {'bit_manipulation': COMMON_SYSTEM_PREFIX + BIT_MANIPULATION_STRATEGY, 'cipher': COMMON_SYSTEM_PREFIX + CIPHER_STRATEGY, 'formula_based': COMMON_SYSTEM_PREFIX + FORMULA_BASED_STRATEGY, 'unit_conversion': COMMON_SYSTEM_PREFIX + UNIT_CONVERSION_STRATEGY, 'roman_numeral': COMMON_SYSTEM_PREFIX + ROMAN_NUMERAL_STRATEGY, 'numeric_symbol': COMMON_SYSTEM_PREFIX + NUMERIC_SYMBOL_STRATEGY, 'pure_symbol': COMMON_SYSTEM_PREFIX + PURE_SYMBOL_STRATEGY}

DEFAULT_SYSTEM_PROMPT = COMMON_SYSTEM_PREFIX + '\n\nFor unknown rule-induction problems, compare the examples carefully, infer the most consistent hidden rule, verify it against all examples, and apply the rule to the query.'

def get_system_prompt_by_pattern(pattern):
    """
    pattern별 system.content를 반환한다.

    현재 실험에서는 system message를 messages에 넣지 않는다.
    이 함수는 기존 코드 호환용으로 유지한다.
    """
    pattern = str(pattern).strip()
    return PATTERN_SYSTEM_PROMPTS.get(pattern, DEFAULT_SYSTEM_PROMPT)


# ===== Extracted from notebook cell 108 =====
def get_solver_solution(row):
    """
    solver_solution이 있으면 사용하고,
    없으면 fallback reasoning을 사용한다.
    """
    if pd.notna(row.get('solver_solution')) and str(row.get('solver_solution')).strip() != '':
        return str(row['solver_solution']).strip()
    return 'Task: unknown_rule_induction\nstatus=solver_solution_missing\naction=infer the hidden transformation rule from examples and apply it to the query'

def get_training_answer(row):
    """
    학습용 assistant.content에 들어갈 answer를 반환한다.

    기본:
        row["answer"] 사용

    unit_conversion:
        solver_answer를 학습 정답으로 사용한다.

    formula_based:
        solver_answer를 학습 정답으로 사용한다.

    numeric_symbol:
        고득점자 reasoning 내부 예측값을 그대로 학습 정답으로 사용한다.
        예: prediction/raw solver output이 6}이면 assistant.content도 \\boxed{6}} 형태가 된다.

    pure_symbol:
        metric extractor가 raw {, }를 처리하므로 escape하지 않는다.
    """
    pattern = str(row.get('pattern', '')).strip()
    if pattern in {'unit_conversion', 'formula_based', 'numeric_symbol', 'bit_manipulation'}:
        solver_answer = row.get('solver_answer')
        if pd.notna(solver_answer) and str(solver_answer).strip() != '' and (str(solver_answer).strip().lower() != 'none'):
            return str(solver_answer).strip()
    return str(row['answer']).strip()

def make_boxed_answer(answer):
    """
    metric extract_final_answer는 answer 안의 }도 처리하도록 설계되어 있다.
    따라서 pure_symbol/numeric_symbol 모두 escape하지 않고 raw answer를 그대로 넣는다.

    예:
        answer = 6}
        boxed  = \\boxed{6}}
    """
    return f'\\boxed{{{str(answer)}}}'

def build_training_messages(row, use_reasoning_content=True):
    """
    Training message 생성.

    이번 실험 포맷:
        - system message 없음
        - user.content = original prompt + metric과 동일한 PROMPT_SUFFIX
        - assistant.reasoning_content = solver_solution
        - assistant.content = boxed training answer

    중요:
        - metric의 실제 추론 입력도 user prompt 뒤에 PROMPT_SUFFIX를 붙인다.
        - unit_conversion / formula_based / numeric_symbol은 solver_answer를 학습 정답으로 사용한다.
    """
    pattern = str(row.get('pattern', '')).strip()
    prompt = str(row['prompt']).strip()
    user_content = prompt + PROMPT_SUFFIX
    answer = get_training_answer(row)
    boxed_answer = make_boxed_answer(answer)
    solution = get_solver_solution(row)
    if use_reasoning_content:
        assistant_message = {'role': 'assistant', 'reasoning_content': solution, 'content': boxed_answer}
    else:
        assistant_message = {'role': 'assistant', 'content': f'<think>\n{solution}\n</think>\n{boxed_answer}'}
    return [{'role': 'user', 'content': user_content}, assistant_message]

sample_idx = 0


# ===== Extracted from notebook cell 118 =====
def print_distribution(df, title='Distribution'):
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)
    print('\n[Pattern Count]')
    print(df['pattern'].value_counts().to_string())
    print('\n[Solver Solved by Pattern]')
    print(df.groupby('pattern')['solver_solved'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'total', 'sum': 'solved', 'mean': 'coverage'}).to_string())
    print('\n[Solver Correct by Pattern]')
    print(df.groupby('pattern')['solver_correct'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'total', 'sum': 'correct', 'mean': 'correct_rate'}).to_string())
    print('\n[Rule Count]')
    print(df[df['solver_solved']].groupby(['pattern', 'solver_rule_name']).size().sort_values(ascending=False).to_string())


# ===== Extracted from notebook cell 120 =====
def sample_group(group, n=None, random_state=42):
    """
    group에서 최대 n개만 샘플링한다.

    n=None이면 전체 사용.
    n=0이면 빈 DataFrame 반환.
    len(group) <= n이면 전체 사용.
    """
    if n is None:
        return group
    if n <= 0:
        return group.iloc[0:0].copy()
    if len(group) <= n:
        return group
    return group.sample(n=n, random_state=random_state)

def prepare_solver_status_columns(df):
    """
    solver 결과 상태 컬럼을 일관되게 만든다.

    생성 컬럼:
        - row_id
        - solver_answer_str
        - answer_str
        - is_correct
        - correct_solved
        - wrong_solved
        - unsolved
    """
    df = df.copy()
    if 'row_id' not in df.columns:
        df['row_id'] = df.index
    df['solver_answer_str'] = df['solver_answer'].astype(str).str.strip()
    df['answer_str'] = df['answer'].astype(str).str.strip()
    df['is_correct'] = df['solver_answer_str'] == df['answer_str']
    df['solver_solved'] = df['solver_solved'].astype(bool)
    df['correct_solved'] = df['solver_solved'] & df['is_correct']
    df['wrong_solved'] = df['solver_solved'] & ~df['is_correct']
    df['unsolved'] = ~df['solver_solved']
    return df

def filter_by_mode(pattern_df, mode):
    """
    mode에 따라 pattern_df를 필터링한다.

    mode:
        - all
        - correct
        - wrong
        - unsolved
    """
    if mode == 'all':
        return pattern_df.copy()
    if mode == 'correct':
        return pattern_df[pattern_df['correct_solved']].copy()
    if mode == 'wrong':
        return pattern_df[pattern_df['wrong_solved']].copy()
    if mode == 'unsolved':
        return pattern_df[pattern_df['unsolved']].copy()
    raise ValueError(f'Unknown mode: {mode}')

def select_pattern_subset(df, pattern, mode='all', n=None, group_by_rule=False, random_state=42):
    """
    특정 pattern에서 원하는 조건으로 데이터를 선택한다.

    Parameters
    ----------
    df:
        processed_train_df

    pattern:
        선택할 pattern 이름

    mode:
        all / correct / wrong / unsolved

    n:
        group_by_rule=False이면 해당 mode 전체에서 최대 n개 선택
        group_by_rule=True이면 rule별 최대 n개 선택

    group_by_rule:
        True이면 solver_rule_name별로 나눠서 샘플링
        False이면 전체에서 샘플링

    random_state:
        샘플링 고정 seed
    """
    pattern_df = df[df['pattern'] == pattern].copy()
    filtered_df = filter_by_mode(pattern_df, mode=mode)
    if len(filtered_df) == 0:
        return filtered_df
    if group_by_rule:
        filtered_df = filtered_df.copy()
        filtered_df['_rule_key'] = filtered_df['solver_rule_name'].fillna('NO_RULE')
        sampled_df = filtered_df.groupby('_rule_key', group_keys=False).apply(lambda g: sample_group(g, n, random_state)).drop(columns=['_rule_key']).reset_index(drop=True)
        return sampled_df
    return sample_group(filtered_df, n=n, random_state=random_state).reset_index(drop=True)

def build_selected_dataset_by_config(df, pattern_configs, random_state=42, shuffle=True, drop_duplicates_by_row_id=True):
    """
      ============================================================
      Pattern Sampling Config
      ============================================================
      사용 방법:

      pattern_configs는 pattern별로 어떤 데이터를 얼마나 뽑을지 정하는 설정이다.

      각 pattern 아래에는 여러 개의 조건을 list로 넣을 수 있다.

      사용 가능한 mode:
        "all"      : 해당 pattern 전체에서 샘플링
        "correct"  : solver_solved=True 이고 solver_answer == answer 인 것만 샘플링
        "wrong"    : solver_solved=True 이고 solver_answer != answer 인 것만 샘플링
        "unsolved" : solver_solved=False 인 것만 샘플링

      n:
        n=None : 조건에 맞는 데이터 전체 사용
        n=0    : 하나도 사용하지 않음
        n=100  : 최대 100개만 샘플링

      group_by_rule:
        False : 조건에 맞는 전체 데이터에서 n개 샘플링
        True  : solver_rule_name별로 각각 최대 n개씩 샘플링

      예시:
        {"mode": "all", "n": 600, "group_by_rule": False}
            -> 해당 pattern 전체에서 최대 600개 샘플링

        {"mode": "correct", "n": 3, "group_by_rule": True}
            -> 정답과 일치한 solved 데이터 중 solver_rule_name별 최대 3개씩 샘플링

        {"mode": "unsolved", "n": None, "group_by_rule": False}
            -> solve 안 된 데이터 전체 사용

        {"mode": "wrong", "n": 50, "group_by_rule": False}
            -> solver가 풀었지만 정답과 다른 오답 solved 데이터 중 최대 50개 샘플링

      주의:
        - 특정 pattern을 아예 제외하고 싶으면 pattern_configs에 그 pattern key를 넣지 않으면 된다.
        - n=0으로 넣어도 제외되지만, 보통은 key를 빼는 방식이 더 깔끔하다.
        - 같은 row가 여러 조건에 동시에 선택될 수 있으므로,
          build_selected_dataset_by_config(..., drop_duplicates_by_row_id=True)가 중복을 제거한다.
      ============================================================
    """
    df = prepare_solver_status_columns(df)
    selected_parts = []
    for pattern, configs in pattern_configs.items():
        for config in configs:
            mode = config.get('mode', 'all')
            n = config.get('n', None)
            group_by_rule = config.get('group_by_rule', False)
            part = select_pattern_subset(df=df, pattern=pattern, mode=mode, n=n, group_by_rule=group_by_rule, random_state=random_state)
            part = part.copy()
            part['select_pattern'] = pattern
            part['select_mode'] = mode
            part['select_group_by_rule'] = group_by_rule
            part['select_n'] = n
            selected_parts.append(part)
    if len(selected_parts) == 0:
        return df.iloc[0:0].copy()
    selected_df = pd.concat(selected_parts, ignore_index=True)
    if drop_duplicates_by_row_id and 'row_id' in selected_df.columns:
        before = len(selected_df)
        selected_df = selected_df.drop_duplicates(subset=['row_id'], keep='first').copy()
        after = len(selected_df)
        print('Dropped duplicate rows:', before - after)
    if shuffle:
        selected_df = selected_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return selected_df


# ===== Extracted from notebook cell 121 =====
from sklearn.model_selection import train_test_split

def _empty_like(df):
    """
    df와 같은 컬럼 구조를 가진 빈 DataFrame 반환.
    """
    return df.iloc[0:0].copy()

def _safe_concat(parts, like_df):
    """
    parts가 비어 있어도 안전하게 concat한다.
    """
    if parts is None or len(parts) == 0:
        return _empty_like(like_df)
    parts = [part for part in parts if part is not None and len(part) > 0]
    if len(parts) == 0:
        return _empty_like(like_df)
    return pd.concat(parts, ignore_index=False)

def split_remaining_val_test_rule_aware(remaining_df, target_patterns=('bit_manipulation', 'numeric_symbol'), val_ratio=0.5, random_state=42):
    """
    remaining_df를 val/test로 나누되,
    target_patterns의 solver_rule_name이 val/test에 최소 1개씩 들어가도록 우선 배정한다.

    조건:
        - 같은 pattern + solver_name + solver_rule_name 기준으로 그룹화
        - 그룹 내 샘플이 2개 이상이면 val/test에 1개씩 배정
        - 1개뿐이면 val에 우선 배정
        - 나머지는 랜덤 split

    방어 처리:
        - remaining_df가 비어 있으면 val/test 모두 빈 DataFrame 반환
        - val_parts 또는 test_parts가 비어 있어도 에러 없이 빈 DataFrame 반환
        - stratify가 불가능한 소수 클래스가 있으면 stratify=None으로 fallback
    """
    remaining_df = remaining_df.copy()
    if len(remaining_df) == 0:
        return (_empty_like(remaining_df), _empty_like(remaining_df))
    key_cols = ['pattern', 'solver_name', 'solver_rule_name']
    missing_cols = [col for col in key_cols if col not in remaining_df.columns]
    if missing_cols:
        raise ValueError(f'remaining_df에 필요한 컬럼이 없습니다: {missing_cols}')
    target_df = remaining_df[remaining_df['pattern'].isin(target_patterns) & remaining_df['solver_rule_name'].notna() & remaining_df['solver_name'].notna()].copy()
    non_target_df = remaining_df.drop(index=target_df.index).copy()
    val_parts = []
    test_parts = []
    used_indices = set()
    grouped = target_df.groupby(key_cols, dropna=False)
    for _, group in grouped:
        group = group.sample(frac=1, random_state=random_state)
        if len(group) >= 2:
            val_parts.append(group.iloc[[0]])
            test_parts.append(group.iloc[[1]])
            used_indices.update(group.iloc[[0, 1]].index)
        elif len(group) == 1:
            val_parts.append(group.iloc[[0]])
            used_indices.add(group.index[0])
    reserved_df = target_df.loc[~target_df.index.isin(used_indices)].copy()
    rest_parts = []
    if len(reserved_df) > 0:
        rest_parts.append(reserved_df)
    if len(non_target_df) > 0:
        rest_parts.append(non_target_df)
    rest_df = _safe_concat(rest_parts, like_df=remaining_df)
    if len(rest_df) > 0:
        if len(rest_df) == 1:
            val_parts.append(rest_df)
        else:
            stratify_col = None
            if 'pattern' in rest_df.columns and rest_df['pattern'].nunique() > 1:
                pattern_counts = rest_df['pattern'].value_counts()
                if pattern_counts.min() >= 2:
                    stratify_col = rest_df['pattern']
            rest_val_df, rest_test_df = train_test_split(rest_df, test_size=1 - val_ratio, random_state=random_state, shuffle=True, stratify=stratify_col)
            val_parts.append(rest_val_df)
            test_parts.append(rest_test_df)
    val_df = _safe_concat(val_parts, like_df=remaining_df)
    test_df = _safe_concat(test_parts, like_df=remaining_df)
    if len(val_df) > 0:
        val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        val_df = val_df.reset_index(drop=True)
    if len(test_df) > 0:
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        test_df = test_df.reset_index(drop=True)
    return (val_df, test_df)

def split_train_val_test_from_selection_rule_aware(original_df, selected_df, val_ratio=0.5, random_state=42):
    """
    selected_df → train
    remaining_df → rule-aware val/test split

    방어 처리:
        - row_id가 없으면 index 기반으로 생성
        - remaining_df가 비어 있으면 val/test는 빈 DataFrame 반환
    """
    original_df = original_df.copy()
    selected_df = selected_df.copy()
    if 'row_id' not in original_df.columns:
        original_df['row_id'] = original_df.index
    if 'row_id' not in selected_df.columns:
        selected_df['row_id'] = selected_df.index
    selected_row_ids = set(selected_df['row_id'])
    train_final_df = selected_df.copy()
    remaining_df = original_df[~original_df['row_id'].isin(selected_row_ids)].copy()
    print('Selected rows:', len(train_final_df))
    print('Remaining rows:', len(remaining_df))
    if len(remaining_df) == 0:
        print('remaining_df is empty. val/test will be empty.')
        return (train_final_df, _empty_like(original_df), _empty_like(original_df))
    val_final_df, test_final_df = split_remaining_val_test_rule_aware(remaining_df=remaining_df, target_patterns=('bit_manipulation', 'numeric_symbol'), val_ratio=val_ratio, random_state=random_state)
    return (train_final_df, val_final_df, test_final_df)


# ===== Extracted from notebook cell 122 =====
def split_train_val_from_selection(original_df, selected_df):
    selected_row_ids = set(selected_df['row_id'])
    train_final_df = selected_df.copy()
    val_final_df = original_df[~original_df['row_id'].isin(selected_row_ids)].copy()
    return (train_final_df, val_final_df)


# ===== VSCode/repo helper utilities =====

def initialize_cipher_vocab_from_dataframe(df: pd.DataFrame, pattern_col: str = "pattern", answer_col: str = "answer") -> None:
    """Initialize global cipher vocabulary/statistics from the training dataframe.

    The Colab notebook built cipher_answer_vocab from train_df at top-level. In the repo,
    call this once before running solve_cipher so the module has the same resources.
    """
    global cipher_answer_vocab, cipher_external_vocab, cipher_noun_vocab
    global cipher_display_vocab, cipher_selection_vocab, CIPHER_STYLE_VOCAB_TABLE_TEXT
    global cipher_unigram_counts, cipher_bigram_counts

    cipher_answer_vocab = set()
    if pattern_col in df.columns:
        cipher_answer_series = df.loc[df[pattern_col] == "cipher", answer_col].astype(str)
    else:
        cipher_answer_series = df[answer_col].astype(str)

    for text in cipher_answer_series:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        cipher_answer_vocab.update(words)

    cipher_answer_vocab = set(sorted(cipher_answer_vocab))
    cipher_external_vocab = set()
    cipher_noun_vocab = set()
    cipher_display_vocab = set(sorted(cipher_answer_vocab))
    cipher_selection_vocab = set(sorted(cipher_answer_vocab))
    CIPHER_STYLE_VOCAB_TABLE_TEXT = format_vocab_table(cipher_selection_vocab, cols=8)

    cipher_unigram_counts = Counter()
    cipher_bigram_counts = Counter()
    for text in cipher_answer_series:
        words = re.findall(r"[a-zA-Z]+", str(text).lower())
        for w in words:
            cipher_unigram_counts[w] += 1
        for a, b in zip(words, words[1:]):
            cipher_bigram_counts[(a, b)] += 1


def parse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add pattern/examples/query columns using the pattern-aware parsers.

    If the input CSV is the original Kaggle train.csv, it usually has only
    id/prompt/answer. In that case pattern is inferred from prompt. If it is a
    high-score matched file with a `type` column, that column is copied to pattern.
    """
    df = df.copy()

    if "pattern" not in df.columns:
        if "type" in df.columns:
            df["pattern"] = df["type"]
        else:
            df["pattern"] = df["prompt"].apply(detect_pattern)

    parsed = df.apply(lambda row: pd.Series(parse_prompt_by_pattern(row)), axis=1)
    parsed.columns = ["examples", "query"]
    df[["examples", "query"]] = parsed
    return df


def solve_dataframe(df: pd.DataFrame, initialize_cipher: bool = True) -> pd.DataFrame:
    """Run all pattern solvers and add solver_* columns."""
    df = df.copy()
    if "examples" not in df.columns or "query" not in df.columns:
        df = parse_dataframe(df)

    if initialize_cipher:
        initialize_cipher_vocab_from_dataframe(df)

    df["solver_result"] = df.apply(run_solver_by_pattern, axis=1)
    df["solver_solved"] = df["solver_result"].apply(lambda x: bool(x.get("solved", False)))
    df["solver_answer"] = df["solver_result"].apply(lambda x: x.get("answer"))
    df["solver_solution"] = df["solver_result"].apply(lambda x: x.get("solution"))
    df["solver_rule_name"] = df["solver_result"].apply(lambda x: x.get("rule_name"))
    df["solver_name"] = df["solver_result"].apply(lambda x: x.get("solver_name"))
    df["solver_correct"] = df.apply(
        lambda row: metric_like_match(row.get("answer"), row.get("solver_answer")),
        axis=1,
    )
    return df


def add_training_messages(df: pd.DataFrame, use_reasoning_content: bool = True) -> pd.DataFrame:
    """Add messages column for SFT training."""
    df = df.copy()
    df["messages"] = df.apply(
        lambda row: build_training_messages(row, use_reasoning_content=use_reasoning_content),
        axis=1,
    )
    return df


def build_sft_dataframe(df: pd.DataFrame, use_reasoning_content: bool = True) -> pd.DataFrame:
    """Parse, solve, evaluate, and add training messages."""
    solved_df = solve_dataframe(df)
    return add_training_messages(solved_df, use_reasoning_content=use_reasoning_content)
